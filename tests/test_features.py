"""Unit tests for the URL Feature Extractor."""

import math
from pathlib import Path

import pandas as pd
import pytest

from src.features.extractor import FEATURES, extract_batch, extract_features

EXPECTED_FEATURES = [
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

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_urls.csv"


def test_features_length_and_order():
    """Test 1 & 2: FEATURES contains exactly 16 elements in the required order."""
    assert len(FEATURES) == 16
    assert FEATURES == EXPECTED_FEATURES


def test_extract_features_dict_structure():
    """Test 3 & 4: extract_features returns a dictionary with keys matching FEATURES order."""
    res = extract_features("https://example.com")
    assert isinstance(res, dict)
    assert list(res.keys()) == FEATURES


def test_feature_values_numeric_and_finite():
    """Test 5 & 6: All feature values must be numeric and finite."""
    urls = [
        "https://example.com",
        "http://192.168.0.1/login",
        "http://[::1]/",
        "not a url",
        "",
        None,
        "https://sub.domain-test.co.uk/path/file.exe?user=1&id=2#section",
    ]
    for url in urls:
        features = extract_features(url)
        for name, val in features.items():
            assert isinstance(val, (int, float)), f"{name} is not numeric for url: {url}"
            assert not math.isnan(val), f"{name} is NaN for url: {url}"
            assert not math.isinf(val), f"{name} is Inf for url: {url}"


def test_https_detection():
    """Test 7 & 8: HTTPS detection flag correctly identifies https scheme."""
    https_res = extract_features("https://example.com")
    http_res = extract_features("http://example.com")
    no_scheme_res = extract_features("example.com")

    assert https_res["https_flag"] == 1
    assert http_res["https_flag"] == 0
    assert no_scheme_res["https_flag"] == 0


def test_ip_address_detection():
    """Test 9 & 10: IPv4 and IPv6 detection works."""
    ipv4_res = extract_features("http://192.168.0.1/login")
    assert ipv4_res["has_ip_address"] == 1
    assert ipv4_res["domain_name_length"] == 0

    ipv6_res = extract_features("http://[::1]/")
    assert ipv6_res["has_ip_address"] == 1
    assert ipv6_res["domain_name_length"] == 0

    normal_res = extract_features("https://example.com")
    assert normal_res["has_ip_address"] == 0


def test_url_length():
    """Test 11: URL length is correctly calculated on normalized URL."""
    res = extract_features("https://example.com/test")
    assert res["url_length"] == len("https://example.com/test")

    # When scheme is missing, prepends http://
    res_no_scheme = extract_features("example.com")
    assert res_no_scheme["url_length"] == len("http://example.com")


def test_dot_count():
    """Test 12: Dot count matches number of dots in normalized URL."""
    res = extract_features("https://a.b.c.example.com/path.html")
    assert res["dot_count"] == 5


def test_subdomain_count():
    """Test 13: Subdomain count works via tldextract."""
    single_sub = extract_features("https://blog.example.com")
    assert single_sub["subdomain_count"] == 1

    multi_sub = extract_features("https://a.b.c.example.com")
    assert multi_sub["subdomain_count"] == 3

    no_sub = extract_features("https://example.com")
    assert no_sub["subdomain_count"] == 0


def test_query_param_count():
    """Test 14: Query parameter count correctly parses parameters."""
    with_query = extract_features("https://example.com/search?q=test&lang=en&page=2")
    assert with_query["query_param_count"] == 3

    no_query = extract_features("https://example.com/search")
    assert no_query["query_param_count"] == 0


def test_tld_popularity():
    """Test 15: TLD popularity works for specified popular suffixes."""
    assert extract_features("https://example.com")["tld_popularity"] == 1
    assert extract_features("https://example.org")["tld_popularity"] == 1
    assert extract_features("https://example.co.uk")["tld_popularity"] == 1
    assert extract_features("https://example.xyz")["tld_popularity"] == 0
    assert extract_features("https://example.top")["tld_popularity"] == 0


def test_suspicious_file_extension():
    """Test 16: Suspicious extensions are detected in path without query."""
    assert extract_features("https://example.com/malware.exe")["suspicious_file_extension"] == 1
    assert extract_features("https://example.com/archive.zip?download=1")["suspicious_file_extension"] == 1
    assert extract_features("https://example.com/script.js")["suspicious_file_extension"] == 1
    assert extract_features("https://example.com/app.apk")["suspicious_file_extension"] == 1
    assert extract_features("https://example.com/index.html")["suspicious_file_extension"] == 0
    assert extract_features("https://example.com/page.php?file=test.exe")["suspicious_file_extension"] == 0


def test_numeric_character_calculations():
    """Test 17: Digit count and percentage calculations."""
    res = extract_features("https://example.com/user1234")
    assert res["number_of_digits"] == 4
    expected_pct = (4 / len("https://example.com/user1234")) * 100.0
    assert abs(res["percentage_numeric_chars"] - expected_pct) < 1e-4


def test_url_entropy():
    """Test 18: Shannon entropy is nonnegative and finite."""
    res = extract_features("https://example.com/very/complex/path/with/random/chars?id=123")
    assert res["url_entropy"] > 0
    assert not math.isnan(res["url_entropy"])
    assert not math.isinf(res["url_entropy"])


def test_edge_cases_do_not_raise():
    """Test 19, 20, 21, 22, 23: None, empty string, malformed, unicode, and long URLs do not raise."""
    assert extract_features(None)["url_length"] == 0
    assert extract_features("")["url_length"] == 0
    assert extract_features("   ")["url_length"] == 0

    malformed = extract_features("not a url")
    assert isinstance(malformed, dict)
    assert len(malformed) == 16

    unicode_url = extract_features("https://münchen.de/über-uns")
    assert isinstance(unicode_url, dict)
    assert unicode_url["has_hyphen_in_domain"] == 0

    long_url = "https://example.com/" + ("a" * 5000)
    long_res = extract_features(long_url)
    assert long_res["url_length"] == len(long_url)
    assert long_res["path_length"] == 5001


def test_batch_extraction():
    """Test 24, 25, 26, 27: Batch extraction row counts, columns, empty batch, and consistency."""
    sample_urls = [
        "https://example.com",
        "http://192.168.0.1/login",
        "https://bank-security.com/portal",
        "",
        None,
    ]
    df = extract_batch(sample_urls)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(sample_urls)
    assert list(df.columns) == FEATURES

    # Empty batch
    empty_df = extract_batch([])
    assert isinstance(empty_df, pd.DataFrame)
    assert len(empty_df) == 0
    assert list(empty_df.columns) == FEATURES

    # Check consistency between extract_features and extract_batch
    single = extract_features("https://example.com")
    from_batch = df.iloc[0].to_dict()
    for col in FEATURES:
        assert single[col] == pytest.approx(from_batch[col])


def test_sample_urls_fixture():
    """Test extraction on all URLs in the fixture file."""
    assert FIXTURE_PATH.exists(), f"Missing fixture file at {FIXTURE_PATH}"
    df_fixture = pd.read_csv(FIXTURE_PATH)
    assert "url" in df_fixture.columns
    assert len(df_fixture) >= 25

    extracted = extract_batch(df_fixture["url"])
    assert len(extracted) == len(df_fixture)
    assert list(extracted.columns) == FEATURES
    assert extracted.isna().sum().sum() == 0
