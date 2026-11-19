"""URL Feature Extractor for PhishingShield.

This module is the single source of truth for URL feature extraction across
training, evaluation, and production inference. It extracts exactly 16 lexical
and structural features from any given URL string.
"""

from __future__ import annotations

import collections
import ipaddress
import math
import re
from typing import Any, Iterable
import urllib.parse

import pandas as pd
import tldextract

# Strictly ordered list of exactly 16 feature names.
# Both model training and inference pipelines depend on this exact ordering.
FEATURES = [
    "url_length",
    "has_ip_address",
    "dot_count",
    "https_flag",
    "url_entropy",
    "token_count",
    "subdomain_count",
    "query_param_count",
    "tld_length",
    "path_length",
    "has_hyphen_in_domain",
    "number_of_digits",
    "tld_popularity",
    "suspicious_file_extension",
    "domain_name_length",
    "percentage_numeric_chars",
]

# Set of popular TLDs as per project specification.
POPULAR_TLDS = frozenset({"com", "org", "net", "edu", "gov", "in", "co", "uk"})

# High-risk executable/script file extensions.
SUSPICIOUS_EXTENSIONS = (
    ".exe",
    ".zip",
    ".rar",
    ".scr",
    ".bat",
    ".js",
    ".apk",
)

# Reusable offline TLDExtract instance (does not attempt network fetches in CI/Docker).
_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=())

# Regex pattern for scheme check
_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")


def _safe_float(val: float) -> float:
    """Ensure a float value is finite (not NaN or Inf)."""
    if math.isnan(val) or math.isinf(val):
        return 0.0
    return float(val)


def extract_features(url: Any) -> dict[str, int | float]:
    """Extract exactly 16 numeric features from a URL string.

    Feature Definitions:
    1. url_length: Length of normalized URL string.
    2. has_ip_address: 1 if hostname is IPv4/IPv6, else 0.
    3. dot_count: Total count of '.' in normalized URL.
    4. https_flag: 1 if original URL explicitly used HTTPS scheme, else 0.
    5. url_entropy: Shannon entropy H = -sum(p_i * log2(p_i)) of characters.
    6. token_count: Count of alphanumeric tokens delimited by non-alphanumeric chars.
    7. subdomain_count: Number of dot-separated subdomain components from tldextract.
    8. query_param_count: Number of query parameters parsed from query string.
    9. tld_length: Length of extracted public suffix (including multipart separator).
    10. path_length: Character length of URL path component.
    11. has_hyphen_in_domain: 1 if registered domain name contains '-', else 0.
    12. number_of_digits: Count of numeric digits (0-9) in normalized URL.
    13. tld_popularity: 1 if public suffix is in popular set, else 0.
    14. suspicious_file_extension: 1 if path ends with .exe, .zip, .rar, etc., else 0.
    15. domain_name_length: Length of registered domain name without suffix.
    16. percentage_numeric_chars: Percentage (0-100) of digit characters in URL.

    Args:
        url: URL string, or None/empty/malformed object.

    Returns:
        dict: Mapping from each feature name in `FEATURES` to its numeric value.
    """
    default_row: dict[str, int | float] = {feat: 0 for feat in FEATURES}
    default_row["url_entropy"] = 0.0
    default_row["percentage_numeric_chars"] = 0.0

    if url is None:
        return default_row

    raw_str = str(url).strip()
    if not raw_str:
        return default_row

    try:
        # Check HTTPS flag on raw input before normalization
        has_https = 1 if raw_str.lower().startswith("https://") else 0

        # Normalization: prepend http:// if no URI scheme is present
        if _SCHEME_RE.match(raw_str):
            normalized_url = raw_str
        else:
            normalized_url = "http://" + raw_str

        # Parse URL components
        parsed = urllib.parse.urlsplit(normalized_url)

        # 1. url_length
        url_length = len(normalized_url)

        # 2. has_ip_address
        hostname = parsed.hostname or ""
        host_clean = hostname.strip("[]")
        has_ip = 0
        if host_clean:
            try:
                ipaddress.ip_address(host_clean)
                has_ip = 1
            except ValueError:
                has_ip = 0

        # 3. dot_count
        dot_count = normalized_url.count(".")

        # 4. https_flag
        https_flag = has_https

        # 5. url_entropy
        length = len(normalized_url)
        if length > 0:
            char_counts = collections.Counter(normalized_url)
            entropy = -sum((c / length) * math.log2(c / length) for c in char_counts.values())
            url_entropy = _safe_float(entropy)
        else:
            url_entropy = 0.0

        # 6. token_count: non-alphanumeric character delimiter rule
        tokens = [t for t in re.split(r"[^a-zA-Z0-9]+", normalized_url) if t]
        token_count = len(tokens)

        # Extract domain components via tldextract
        ext = _EXTRACTOR(normalized_url)

        # 7. subdomain_count
        if has_ip or not ext.subdomain:
            subdomain_count = 0
        else:
            subdomain_count = len([p for p in ext.subdomain.split(".") if p])

        # 8. query_param_count
        if parsed.query:
            query_params = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
            query_param_count = len(query_params)
        else:
            query_param_count = 0

        # 9. tld_length
        suffix = ext.suffix or ""
        tld_length = len(suffix)

        # 10. path_length
        path = parsed.path or ""
        path_length = len(path)

        # 11. has_hyphen_in_domain
        domain_name = ext.domain or ""
        if has_ip:
            has_hyphen_in_domain = 0
        else:
            has_hyphen_in_domain = 1 if ("-" in domain_name) else 0

        # 12. number_of_digits
        number_of_digits = sum(c.isdigit() for c in normalized_url)

        # 13. tld_popularity
        suffix_lower = suffix.lower()
        if suffix_lower in POPULAR_TLDS:
            tld_popularity = 1
        elif suffix_lower and any(part in POPULAR_TLDS for part in suffix_lower.split(".") if part):
            tld_popularity = 1
        else:
            tld_popularity = 0

        # 14. suspicious_file_extension
        path_lower = path.lower()
        if any(path_lower.endswith(ext_candidate) for ext_candidate in SUSPICIOUS_EXTENSIONS):
            suspicious_file_extension = 1
        else:
            suspicious_file_extension = 0

        # 15. domain_name_length
        domain_name_length = 0 if has_ip else len(domain_name)

        # 16. percentage_numeric_chars
        if url_length > 0:
            percentage_numeric_chars = _safe_float((number_of_digits / url_length) * 100.0)
        else:
            percentage_numeric_chars = 0.0

        return {
            "url_length": url_length,
            "has_ip_address": has_ip,
            "dot_count": dot_count,
            "https_flag": https_flag,
            "url_entropy": url_entropy,
            "token_count": token_count,
            "subdomain_count": subdomain_count,
            "query_param_count": query_param_count,
            "tld_length": tld_length,
            "path_length": path_length,
            "has_hyphen_in_domain": has_hyphen_in_domain,
            "number_of_digits": number_of_digits,
            "tld_popularity": tld_popularity,
            "suspicious_file_extension": suspicious_file_extension,
            "domain_name_length": domain_name_length,
            "percentage_numeric_chars": percentage_numeric_chars,
        }
    except Exception:
        # Fallback to safe zero defaults on any unhandled parsing failure
        return default_row


def extract_batch(urls: Iterable[Any]) -> pd.DataFrame:
    """Extract features for a collection of URLs into a pandas DataFrame.

    Preserves the exact order of `FEATURES` as columns, with one row per
    input URL. Handles empty collections without errors.

    Args:
        urls: Iterable of URL strings or raw objects.

    Returns:
        pd.DataFrame: 16-column DataFrame containing extracted numeric features.
    """
    rows = [extract_features(u) for u in urls]
    if not rows:
        return pd.DataFrame(columns=FEATURES)

    df = pd.DataFrame(rows, columns=FEATURES)
    return df
