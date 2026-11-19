# Arman: Data & ML Engineer

**Branch:** `feature/data-ml`
**Merges to `main` via PR only. Ask one teammate to review each PR.**

**Workload: ~17 h (~42% of the team, the largest share)**

## Your mission
Own everything from the raw CSV to "a URL becomes 16 numbers". You build the **one feature extractor that the whole project depends on**, so it ships first. Later you own evaluation, the drift batch, the model performance dashboard in the UI, testing of the running app, the README and the demo script.

You are on the critical path for the first 2 hours: Charan (training) and Deeksha (API) are both waiting on `extractor.py`.

---

## Files you own

| File | Purpose |
|---|---|
| `src/data/prepare.py` | Load, clean, dedupe, standardize labels, split, save processed splits |
| `src/features/extractor.py` | URL to features. **Single source of truth** for training and inference |
| `src/evaluation/evaluate.py` | Metrics, confusion matrix, writes `artifacts/reports/evaluation.json` |
| `tests/test_features.py` | Unit tests for the extractor |
| `tests/fixtures/sample_urls.csv` | ~30 hand-picked URLs (mix of phishing and legit) for tests. Small, committed to git |
| `docs/FEATURE_NOTES.md` | Our features vs the dataset's published features; known differences |
| `frontend/dashboard.py` | Model performance dashboard section of the Streamlit UI (`render_dashboard(...)`), called from Deeksha's `frontend/app.py` |
| `README.md` | Final project README (Day 2) |
| `docs/architecture.png` (or .md diagram) | Architecture diagram (Day 2) |

---

## Tasks

### T1.1 Dataset inspection (Day 1, 09:00-09:45), BLOCKS EVERYONE
- [ ] Download LegitPhish from Mendeley to `data/raw/legitphish.csv` (gitignored).
- [ ] Report to the team (post in group chat and record in `docs/FEATURE_NOTES.md`):
  - actual column names, especially the **URL column** and **label column** (plan assumes `ClassLabel`, verify)
  - real row count, class counts, nulls, number of duplicate URLs
  - **actual label encoding** (published: 0 = phishing, 1 = legitimate; verify on a few known URLs)
- [ ] **30-minute rule:** if LegitPhish is not usable by 09:30, stop and tell the team. Do *not* silently switch to UCI. UCI has precomputed features that cannot be derived from a raw URL, which breaks the whole inference design. The team decides together.

### T1.2 `src/data/prepare.py` (Day 1, 09:45-11:00)
- [ ] `load_dataset(path="data/raw/legitphish.csv") -> DataFrame` with exactly two columns: `url` (str), `label` (int, **1 = phishing, 0 = legitimate**).
- [ ] Drop nulls and empty URLs, strip whitespace, **drop duplicate URLs before splitting** (prevents train/test leakage).
- [ ] `make_splits(df, test_size=0.2, random_state=42)` returns a stratified split.
- [ ] Running `python -m src.data.prepare` writes:
  - `data/processed/train.csv` and `data/processed/test.csv`, columns: `url, label, <16 FEATURES>`
- [ ] Print a short summary when run (rows before/after dedupe, class balance per split).

### T1.2b Drift "current" batch (Day 1, 16:00-18:00), USED BY Charan's `drift.py`
- [ ] `make_shifted_batch(test_df, n=3000, random_state=42) -> DataFrame` in `prepare.py`: a deliberately shifted batch built **only from the test split** (never train), e.g. phishing-heavy resample (~80% phishing) plus a bias towards longer, query-heavy URLs.
- [ ] `python -m src.data.prepare` also writes `data/processed/current_batch.csv` (same columns as `test.csv`, gitignored).
- [ ] Document in `docs/FEATURE_NOTES.md` exactly how the batch is shifted, and that it is **simulated**.
- [ ] Optional: a small hand-collected list of fresh URLs as a second "real new batch".

### T1.3 `src/features/extractor.py` (Day 1, 09:45-11:00, deliver to the team by ~11:00)
- [ ] `FEATURES`: list of the 16 names, in this **exact order** (do not reorder; the model depends on it):
  `url_length, has_ip_address, dot_count, https_flag, url_entropy, token_count, subdomain_count, query_param_count, tld_length, path_length, has_hyphen_in_domain, number_of_digits, tld_popularity, suspicious_file_extension, domain_name_length, percentage_numeric_chars`
- [ ] `extract_features(url) -> dict` (keys = `FEATURES`, values numeric).
- [ ] `extract_batch(urls) -> pd.DataFrame` (columns in `FEATURES` order).
- [ ] Normalize: prepend `http://` if there is no scheme.
- [ ] Use `tldextract.TLDExtract(suffix_list_urls=())` so it works offline (Docker/CI have no internet guarantee).
- [ ] `tld_popularity` = 1 if TLD is in `{com, org, net, edu, gov, in, co, uk}`, else 0.
- [ ] `suspicious_file_extension` = 1 if the path ends with `.exe .zip .rar .scr .bat .js .apk`.
- [ ] **Never raises.** Handles `None`, `""`, malformed URLs, IPv6 hosts, very long URLs, unicode. Bad input returns a valid all-numeric feature row.
- [ ] Write one docstring line per feature describing exactly how it is computed.

### T1.4 `tests/test_features.py` (Day 1, with T1.3)
- [ ] Output keys and order equal `FEATURES`.
- [ ] Known values on simple URLs (e.g. `https://example.com` has `https_flag == 1`, `has_ip_address == 0`).
- [ ] IP URL (`http://192.168.0.1/login`) gives `has_ip_address == 1`.
- [ ] Edge cases do not raise: `None`, `""`, `"not a url"`, `"http://[::1]/"`, a 5,000-character URL.
- [ ] `extract_batch` returns the right shape and column order.
- [ ] Tests use **only** `tests/fixtures/` or inline strings, never `data/raw/` (CI has no dataset).

### T1.5 Feature validation notes (Day 1, 14:00-16:00)
- [ ] On ~200 sampled URLs, compare our extractor's output with the dataset's precomputed columns where names overlap.
- [ ] Record differences in `docs/FEATURE_NOTES.md` (we are not byte-for-byte compatible, and that is fine; document it).

### T1.6 `src/evaluation/evaluate.py` (Day 1, 14:00-18:00)
- [ ] `compute_metrics(y_true, y_pred, y_proba) -> dict` returning `accuracy, precision, recall, f1, roc_auc, false_positive_rate`, computed **for the phishing class (label 1)**.
  **Charan imports this in `train.py`.** Agree on the function signature with them by 11:00 and deliver it by 13:00.
- [ ] `python -m src.evaluation.evaluate` loads `artifacts/model/model.joblib` and `data/processed/test.csv` and writes `artifacts/reports/evaluation.json` using this schema (your dashboard in T1.8 reads it):
```json
{
  "model_name": "RandomForest",
  "model_version": "1",
  "evaluated_at": "2026-11-20T15:00:00Z",
  "test_rows": 20000,
  "metrics": {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "roc_auc": 0.0, "false_positive_rate": 0.0},
  "confusion_matrix": {"tn": 0, "fp": 0, "fn": 0, "tp": 0},
  "classification_report": { "...": "sklearn output_dict=True" }
}
```
- [ ] Propose the **model selection rule** to Charan (see "Decisions to validate" below).

### T1.7 README, diagram, demo script (Day 2, 13:00-18:00)
- [ ] `README.md`: what it is, architecture diagram, setup (Windows and Linux), how to train / evaluate / drift / run API / run UI / run tests / docker compose, model comparison table with real numbers, limitations ("lexical heuristics, not a definitive security verdict").
- [ ] Architecture diagram: dataset → prepare → extractor → train (MLflow) → model artifact → FastAPI → Streamlit, plus Docker / CI / drift.
- [ ] Demo script: who says what, in the order of the 8-step demo in `PROJECT_OVERVIEW.md` §14.
- [ ] Collect screenshots from teammates (MLflow, UI, CI, drift).

### T1.8 Streamlit model performance dashboard (Day 2, 09:00-11:00)
- [ ] `frontend/dashboard.py` exposes `render_dashboard(api_url: str, evaluation_path: str = "artifacts/reports/evaluation.json")`. Deeksha's `app.py` calls it; keeping it in its own file avoids merge conflicts on `app.py`.
- [ ] Shows: metrics of the selected model (from `evaluation.json`), confusion matrix (table or heatmap), comparison table of all candidate models (from the API's `/model-info` → `candidates`), model name, version, MLflow run ID, trained date.
- [ ] If `evaluation.json` is missing or the API is down: show a friendly message, never a traceback.
- [ ] Only HTTP calls to the API plus reading `evaluation.json`. Do not load the model or compute features in the UI.

### T1.9 Integration testing of the running app (Day 2, 15:00-16:00)
- [ ] Against the app running locally with `docker compose up`: a known phishing URL, a known legitimate URL, an empty input, an IP-address URL, a 3,000-character URL, an API restart.
- [ ] Record results (pass/fail + screenshot) in `docs/TEST_REPORT.md` and report bugs to their owner immediately.

---

## Interfaces you provide (others depend on these: do not change them without telling the team)

| Interface | Consumer |
|---|---|
| `FEATURES`, `extract_features`, `extract_batch` | Charan (training, drift), Deeksha (API) |
| `load_dataset`, `make_splits`, `data/processed/{train,test}.csv` | Charan (training, drift) |
| `compute_metrics(...)` | Charan (training) |
| `evaluation.json` schema | Your own dashboard, README |
| `make_shifted_batch`, `data/processed/current_batch.csv` | Charan (drift report) |
| `render_dashboard(...)` in `frontend/dashboard.py` | Deeksha (`frontend/app.py`) |

## Interfaces you consume
- `artifacts/model/model.joblib` and `model_metadata.json` from **Charan** (for `evaluate.py`; see `TASKS_CHARAN_MLOPS_CI.md`).
- `/model-info` API response from **Deeksha** (for the dashboard's comparison table; see `TASKS_DEEKSHA_APP_DEPLOY.md`).

## Decisions to validate with the team (my proposals)
1. **Model selection rule:** among candidates with phishing-class `false_positive_rate <= 0.10`, choose the highest **F1**; break ties on **recall**. Never select on accuracy alone.
2. Prediction threshold 0.5 on `P(phishing)` for the PHISHING/LEGITIMATE label (risk levels are separate, owned by Deeksha).

## Definition of done (Arman)
- [ ] `python -m src.data.prepare` produces both processed splits from the real CSV
- [ ] `pytest tests/test_features.py -v` passes locally **and in CI**
- [ ] `evaluation.json` generated for the selected model
- [ ] `current_batch.csv` generated and its shift documented
- [ ] Dashboard renders in the Streamlit app (locally and in Docker) with real metrics
- [ ] `TEST_REPORT.md` filled in for the local Docker app
- [ ] `FEATURE_NOTES.md` and `README.md` complete, with real numbers
- [ ] No dataset or processed files committed

## Your demo moments
Step 3 (compare evaluation metrics on the dashboard) and the overall narration / README walkthrough.
