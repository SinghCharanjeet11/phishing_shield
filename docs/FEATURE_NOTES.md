# PhishingShield: Feature Validation & Dataset Inspection Notes

**Author:** Arman (Data & ML Engineer)  
**Date:** November 2026  
**Dataset Source:** LegitPhish from Mendeley Data (DOI: `10.17632/hx4m73v2sf.2`)  
**SHA-256:** `8685e7901c3b4e73eb43c106cb166cc932dd14f76960a032c713c2c32f09dbcd`  

---

## 1. Dataset Inspection Summary (T1.1)

### Core Statistics
* **Dataset File:** `data/raw/legitphish.csv` (downloaded from Mendeley Data, gitignored)
* **Total Rows in Raw File:** **101,219**
* **Total Columns:** **18**
* **URL Column Name:** `URL` (string)
* **Label Column Name:** `ClassLabel` (float/int)
* **Null Values:**
  * `ClassLabel`: 1 record with missing label (`https://www.chocolatleroux.com`, row index 65,118) — dropped.
  * All feature columns: 0 nulls.
* **Exact Duplicate URLs:** **346 duplicate entries** (648 total duplicate rows)
  * Conflicting labels among duplicates: **0** (no conflicting labels).
  * Deduplication policy: First instance retained; 346 duplicate rows removed before train/test splitting.
* **Final Clean Deduplicated Records:** **100,872**

### Ground Truth Label Encoding
* **Published / Mendeley Author Convention:**
  * `0.0` = **Phishing** (63,678 records in raw)
  * `1.0` = **Legitimate** (37,540 records in raw)
* **Internal PhishingShield Convention:**
  * `1` = **Phishing**
  * `0` = **Legitimate**
* **Mapping Applied:**
  $$\text{Internal Label} = \begin{cases} 1 & \text{if } \text{ClassLabel} = 0 \\ 0 & \text{if } \text{ClassLabel} = 1 \end{cases}$$

### Team Chat Update (Inspection Findings)
> **Team Update from Arman:**  
> Verified raw dataset `legitphish.csv`: 101,219 rows, 18 columns (`URL` and `ClassLabel`). Dropped 1 null label and deduplicated 346 duplicate URLs before splitting, yielding 100,872 clean unique URLs. Published labels are 0=phishing, 1=legit; mapped internally to 1=phishing, 0=legit. Train/test split is 80/20 stratified (80,697 train, 20,175 test). All features are recomputed directly from raw URLs via our shared extractor.

---

## 2. Feature Extraction Validation & Comparison (T1.5)

To evaluate our shared feature extractor (`src/features/extractor.py`), we drew a random sample of **200 URLs** from `legitphish.csv` and compared our extracted features against the precomputed features provided in the dataset.

The PhishingShield system enforces **one single source of truth**: all features used for model training, evaluation, and live FastAPI prediction are computed using our shared Python extractor, completely decoupling runtime inference from any precomputed dataset artifacts.

### Summary of Comparison across 200 Sampled URLs

| Feature Name | Published Dataset Column | Our Extractor Column | 200-Sample Agreement | Match % | Primary Reason for Observed Differences |
|---|---|---|---|---|---|
| URL Length | `url_length` | `url_length` | 197 / 200 | 98.5% | Normalization: URLs lacking scheme prepend `http://` (+7 chars). |
| Has IP Address | `has_ip_address` | `has_ip_address` | 200 / 200 | 100.0% | Exact match using standard Python `ipaddress` parsing. |
| Dot Count | `dot_count` | `dot_count` | 200 / 200 | 100.0% | Exact character count `normalized_url.count('.')`. |
| HTTPS Flag | `https_flag` | `https_flag` | 200 / 200 | 100.0% | Binary check for explicit `https://` prefix. |
| URL Entropy | `url_entropy` | `url_entropy` | 197 / 200 | 98.5% | Shannon entropy formula matches; minor differences on normalized scheme additions. |
| Token Count | `token_count` | `token_count` | 92 / 200 | 46.0% | **Delimiter difference:** We split on all non-alphanumeric `[^a-zA-Z0-9]+`; dataset authors used a different delimiter subset. |
| Subdomain Count | `subdomain_count` | `subdomain_count` | 101 / 200 | 50.5% | **TLD parsing:** We use `tldextract` with Public Suffix List rules; dataset used dot-splitting. |
| Query Parameter Count | `query_param_count` | `query_param_count` | 1 / 200 | 0.5% | **Parsing logic:** We use `urllib.parse.parse_qsl`; dataset authors appear to have defaulted to 1 or counted special characters. |
| TLD Length | `tld_length` | `tld_length` | 101 / 200 | 50.5% | Multipart ccTLD handling (e.g. `.co.uk` = 5 chars in our extractor vs 2 chars in simple split). |
| Path Length | `path_length` | `path_length` | 197 / 200 | 98.5% | Standard URI path length from `urllib.parse.urlsplit`. |
| Hyphen in Domain | `has_hyphen_in_domain` | `has_hyphen_in_domain` | 199 / 200 | 99.5% | Binary check for `'-'` in the registered domain component. |
| Digit Count | `number_of_digits` | `number_of_digits` | 200 / 200 | 100.0% | Exact character count `sum(c.isdigit())`. |
| TLD Popularity | `tld_popularity` | `tld_popularity` | 183 / 200 | 91.5% | Our exact set `{com, org, net, edu, gov, in, co, uk}` vs dataset's internal TLD list. |
| Suspicious Extension | `suspicious_file_extension` | `suspicious_file_extension` | 195 / 200 | 97.5% | Path ends with `.exe`, `.zip`, `.rar`, `.scr`, `.bat`, `.js`, `.apk`. |
| Domain Name Length | `domain_name_length` | `domain_name_length` | 103 / 200 | 51.5% | Registered domain name length via `tldextract` vs full hostname length. |
| Percentage Numeric Chars | `percentage_numeric_chars` | `percentage_numeric_chars` | 199 / 200 | 99.5% | Numeric percentage $0.0 - 100.0\%$ with division-by-zero protection. |

### Detailed Analysis of Differences

1. **`query_param_count` Discrepancy:**
   * *Example:* `http://117.205.108.72:37255/bin.sh` has zero query parameters (`our_qp = 0`). The dataset recorded `dataset_qp = 1`.
   * *Confirmed Explanation:* Our implementation strictly complies with RFC 3986 URL parsing via `urllib.parse.parse_qsl`. The dataset authors used a non-standard heuristic that attributed query parameters to URLs without a `?` symbol. Our extractor provides accurate parsing for production inference.

2. **`token_count` Discrepancy:**
   * *Example:* `https://www.youtube.com/watch?v=2Uj1A9AguFs` yielded 6 tokens with our regex `[^a-zA-Z0-9]+` (`www`, `youtube`, `com`, `watch`, `v`, `2Uj1A9AguFs`) vs 8 in the dataset.
   * *Confirmed Explanation:* Dataset authors split on specific custom symbols and may have treated uppercase/lowercase casing or underscores differently. Our rule is deterministic, documented, and symmetrical across training and inference.

3. **`tld_length` and `domain_name_length` Discrepancy:**
   * *Example:* For `bbc.co.uk`, `tldextract` identifies the public suffix as `co.uk` (length 5) and domain as `bbc` (length 3). Naive dot splitting produces TLD `uk` (length 2) and domain `co` (length 2).
   * *Confirmed Explanation:* We deliberately use `tldextract` to correctly isolate public suffixes without downloading runtime lists. This provides superior semantic correctness for security analysis.

---

## 3. Simulated Drift Dataset (`current_batch.csv`) (T1.2b)

To validate Charan's Evidently data drift monitoring pipeline (`src/monitoring/drift.py`), we generate a **simulated shifted batch** using `make_shifted_batch()` in `src/data/prepare.py`.

### Shift Specifications
* **Source Pool:** Exclusively the **held-out test split** (`data/processed/test.csv`). The training set is **never** sampled to prevent leakage and guarantee that reference and current distributions are independent.
* **Output Path:** `data/processed/current_batch.csv` (gitignored).
* **Sample Size ($n$):** **3,000 rows**.
* **Target Schema:** Exactly 18 columns matching `test.csv`: `url, label, url_length, has_ip_address, dot_count, https_flag, url_entropy, token_count, subdomain_count, query_param_count, tld_length, path_length, has_hyphen_in_domain, number_of_digits, tld_popularity, suspicious_file_extension, domain_name_length, percentage_numeric_chars`.
* **Class Distribution Shift:**
  * Baseline Test Split: $\approx 62.9\%$ Phishing, $37.1\%$ Legitimate.
  * Shifted Batch Target: $\approx \mathbf{80.0\%}$ **Phishing** (2,400 rows) and $\mathbf{20.0\%}$ **Legitimate** (600 rows).
* **Feature Distribution Bias:**
  * For phishing samples, sampling weights are proportional to:
    $$w_i = \text{clip}(\text{url\_length}_i, 1, \infty) \times (1 + \text{query\_param\_count}_i)$$
  * This deliberately biases the batch towards longer, query-heavy phishing URLs, reflecting advanced evasion techniques.
* **Deterministic Reproducibility:** Controlled using `random_state=42`.
* **Sampling Replacement:** When the candidate pool exceeds required rows (as with held-out test split), sampling occurs **without replacement**.

### Important Architectural Notice on Drift Simulation
> **MLOps Notice:** This batch represents a **simulated shift** created specifically for CI and monitoring pipeline demonstration. It must not be reported as live production traffic. The Evidently drift detector evaluates distributional differences between reference training data and this shifted evaluation batch.
