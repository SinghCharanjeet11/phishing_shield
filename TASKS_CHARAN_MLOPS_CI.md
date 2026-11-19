# Charan: MLOps & CI Engineer

**Branch:** `feature/mlops-ci`
**Merges to `main` via PR only. Ask one teammate to review each PR.**

**Workload: ~15.5 h (~34% of the team)**

## Your mission
Own the reproducible training pipeline: train and compare models, track everything in MLflow, save a **versioned** model with metadata, generate the drift report, and make **GitHub Actions** run our tests on every push/PR. You also set up the repo skeleton so everyone starts from the same structure.

(Originally the overview had a separate DevOps member. With 3 people, **CI** and the **AWS account/instance setup** move to you; **Docker and deploying the app on EC2** move to Deeksha.)

---

## Files you own

| File | Purpose |
|---|---|
| Repo skeleton | All folders from `PROJECT_OVERVIEW.md` §6, every `src/*` package with `__init__.py` |
| `.gitignore` | Exact list from overview §6 |
| `requirements.txt` (pinning) | Pin versions after the first clean install |
| `src/training/train.py` | Train LR + RF (+ XGBoost optional), MLflow logging, pick best, save model + metadata |
| `src/monitoring/drift.py` | Evidently drift report from a **separate / shifted** batch |
| `tests/test_model.py` | Tiny synthetic-data training test: save, load, metadata, predict_proba |
| `.github/workflows/ci.yml` | Install, run pytest, then Docker build (Docker build step after Deeksha's Dockerfile exists) |
| AWS setup (no files in git) | EC2 instance, security group, budget alert, SSH access for Deeksha |

---

## Tasks

### T2.1 Repo skeleton and setup (Day 1, 09:00-09:45), BLOCKS EVERYONE
- [ ] Create the GitHub repo, push the skeleton to `main`, create all 3 feature branches.
- [ ] Empty `__init__.py` in `src/` and every subpackage (`data, features, training, evaluation, monitoring, api`) plus `tests/`.
- [ ] `.gitignore` with: `.venv/ __pycache__/ *.pyc .env mlruns/ mlflow.db data/raw/* data/processed/* artifacts/model/* artifacts/reports/* .pytest_cache/ .DS_Store`
- [ ] Add `.gitkeep` to the ignored folders so they exist after clone (use `!` rules so `.gitkeep` is not ignored).
- [ ] Protect `main` (PR required, CI must pass once CI exists).

### T2.2 Pin dependencies (Day 1, by 11:00)
- [ ] After the first clean install on Python 3.11, pin versions (`pip freeze`) and commit.
- [ ] **Pin Evidently explicitly** and write the drift code against *that* version's API only.
- [ ] Check that `xgboost` installs cleanly on all three laptops; if not, drop it from requirements.

### T2.3 `src/training/train.py` (Day 1, 10:00-13:00)
- [ ] While waiting for Arman's extractor, develop against a **synthetic DataFrame** with the 16 `FEATURES` columns, then swap in `data/processed/train.csv` / `test.csv`.
- [ ] MLflow: `mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))`, `mlflow.set_experiment("PhishingShield")`, `mlflow.sklearn.autolog()`.
- [ ] One MLflow run per model, with a clear run name (`logreg-baseline`, `random-forest`, `xgboost`).
- [ ] Models (each one is a sklearn `Pipeline` that accepts a DataFrame with `FEATURES` columns):
  - Logistic Regression: `StandardScaler` + `LogisticRegression(max_iter=1000, class_weight="balanced")`
  - Random Forest: `RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)`
  - XGBoost: optional, only if T2.3 is done before 13:00
- [ ] Log test metrics from Arman's `compute_metrics()` to every run (do not write your own metric code; reuse Arman's).
- [ ] Log params: dataset name, row counts, feature list, seed, git commit hash if available.
- [ ] Select the best model using the **agreed selection rule** (proposal in Arman's file: FPR ≤ 0.10, then max F1, tie → recall).
- [ ] Save to `artifacts/model/`:
  - `model.joblib`: the fitted Pipeline; `predict_proba(X)[:, 1]` **must be P(phishing)**
  - `model_metadata.json` with this schema (Deeksha's `/model-info` returns it):
```json
{
  "model_name": "RandomForest",
  "version": "1",
  "mlflow_run_id": "abc123...",
  "mlflow_experiment": "PhishingShield",
  "trained_at": "2026-11-20T12:30:00Z",
  "dataset": "LegitPhish",
  "train_rows": 80000,
  "test_rows": 20000,
  "features": ["url_length", "...16 in FEATURES order..."],
  "threshold": 0.5,
  "metrics": {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "roc_auc": 0.0, "false_positive_rate": 0.0},
  "candidates": [
    {"model_name": "LogisticRegression", "mlflow_run_id": "...", "metrics": {"...": 0.0}},
    {"model_name": "RandomForest", "mlflow_run_id": "...", "metrics": {"...": 0.0}}
  ]
}
```
- [ ] **Versioning:** if a `model_metadata.json` already exists, the new version = old version + 1, and the old `model.joblib` + metadata are copied to `artifacts/model/archive/v<old>/` before overwriting. (Optional extra: register the model in the MLflow Model Registry.)
- [ ] `MODEL_DIR` env var (default `artifacts/model`) controls where the model is written. **Deeksha's API reads the same env var.**
- [ ] Run: `python -m src.training.train` from the repo root.

### T2.4 Milestone 1 integration (Day 1, 13:00-14:00, with Arman and Deeksha)
- [ ] Real data → extractor → trained model → `/predict` returns a real prediction. Hand the trained `artifacts/model/` folder to Deeksha (zip over chat/drive; it is gitignored).

### T2.5 `tests/test_model.py` (Day 1, 14:00-16:00)
- [ ] Build ~200 rows of synthetic `FEATURES` data inside the test, train a tiny model, save to a `tmp_path` `MODEL_DIR`.
- [ ] Assert: files exist, metadata has every required key, `features == FEATURES`, reloaded model gives `predict_proba` in `[0, 1]` with shape `(n, 2)`, version increments on a second save.
- [ ] **No access to `data/raw`, `data/processed`, `mlflow.db` or the real model** (CI does not have them). Use a temporary MLflow URI (`file:` under `tmp_path`) if the test touches MLflow.
- [ ] Factor save/load helpers so they can be tested without running the full training (e.g. `save_model(model, metadata, model_dir)` / `load_model(model_dir)`). **Deeksha imports `load_model` in the API.**

### T2.6 GitHub Actions CI (Day 1, 10:00-13:00 skeleton, finished Day 2 morning)
- [ ] `.github/workflows/ci.yml`: trigger on push and PR to `main`.
- [ ] Steps: checkout → setup Python 3.11 (pip cache) → `pip install -r requirements.txt` → `pytest tests/ -v`.
- [ ] Workflow fails if any test fails.
- [ ] After Deeksha's `Dockerfile` lands: add a `docker build` job that runs **only after tests pass** (`needs: test`).
- [ ] Add the CI badge to the README (give it to Arman).

### T2.7 `src/monitoring/drift.py` (Day 1, 16:00-18:00, finished Day 2 11:00-13:00)
- [ ] **Reference** = features from `data/processed/train.csv` (sample is fine, e.g. 5k rows).
- [ ] **Current** = `data/processed/current_batch.csv`, the shifted batch **built by Arman** (`make_shifted_batch` in `prepare.py`, from the test split, clearly **simulated**). Never use the training set as current data.
- [ ] Until Arman's batch exists, test the report with a quick in-script perturbation of a test-split sample.
- [ ] All features come from Arman's code (never reimplement features).
- [ ] Output `artifacts/reports/drift_report.html` (and a small `drift_summary.json`: number of drifted features, share drifted, timestamp).
- [ ] Run: `python -m src.monitoring.drift`.
- [ ] Wording in the report/README: "simulated drift on a shifted batch", **not** "production drift detection".

### T2.8 Verification (Day 2, 11:00-13:00)
- [ ] Fresh clone → install → `prepare` → `train` → `evaluate` → `drift` works end to end with no manual steps.
- [ ] MLflow UI shows both runs with params, metrics and artifacts: `mlflow ui --host 127.0.0.1 --port 5000` (**local only, never exposed on EC2**).

### T2.9 AWS account and instance setup (Day 1, 16:00-18:00 or Day 2, 11:00-13:00), OPTIONAL
- [ ] Set an AWS **budget alert** first (e.g. $5).
- [ ] Launch Ubuntu LTS, **t3.small**, 15-20 GB disk; create the key pair and share it privately with Deeksha (**never commit `.pem` files or AWS keys**).
- [ ] Security group: **22 only from our IP**, **8501 public**, **8000 NOT public**, **MLflow (5000) never open**.
- [ ] Hand over to Deeksha by Day 2, 13:00: public IP, SSH command, confirmation that the ports are set as above.
- [ ] After the demo: **stop the instance** (or terminate it once the team agrees).

---

## Interfaces you provide

| Interface | Consumer |
|---|---|
| `artifacts/model/model.joblib` + `model_metadata.json` (schema above) | Deeksha (API `/predict`, `/model-info`), Arman (`evaluate.py`) |
| `load_model(model_dir)` helper and the `MODEL_DIR` env var | Deeksha (API, test fixtures) |
| `drift_report.html` / `drift_summary.json` | Deeksha (optional link in UI), Arman (README) |
| CI workflow | Everyone |
| Running EC2 instance + SSH access | Deeksha (deployment) |

## Interfaces you consume
- `FEATURES`, `extract_batch`, `load_dataset`, `make_splits`, `compute_metrics`, processed CSVs and `current_batch.csv` from **Arman** (`TASKS_ARMAN_DATA_ML.md`).
- `Dockerfile` from **Deeksha** (for the CI Docker build step).

## Definition of done (Charan)
- [ ] `python -m src.training.train` trains ≥ 2 models, both visible in MLflow, best saved with metadata containing the run ID
- [ ] Version increments and archives the previous model on retrain
- [ ] `pytest tests/test_model.py -v` passes locally and in CI
- [ ] CI is green on `main` and runs the Docker build after tests
- [ ] `drift_report.html` generated from a separate batch
- [ ] No `mlruns/`, `mlflow.db` or model binaries committed
- [ ] (Optional) EC2 instance ready for Deeksha with the correct security group and a budget alert

## Your demo moments
Step 1 (train both models), step 2 (MLflow runs), step 4 (versioned model + metadata), step 6 (push → GitHub Actions), step 7 (drift report).
