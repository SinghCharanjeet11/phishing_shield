# PhishingShield: Project Overview

> Read this file first. It describes what we are building, the decisions already made, the target structure, and the order of work. If anything here conflicts with what you find in the repo, tell us before changing direction.

## 1. What we are building

**PhishingShield** (display name in UI may be "PhiShark - AI Phishing Analyst") is an MLOps-focused phishing URL detector. A user pastes a URL; the system predicts **phishing vs legitimate**, a phishing probability, and a risk level.

It is a **48-hour college project by a team of 4**. The goal is a visible, working **MLOps lifecycle**, not a fancy UI:

train and compare models -> track in MLflow -> save versioned model -> serve via API -> frontend -> Docker -> CI -> drift report -> (optional) AWS EC2

### Definition of done
- At least 2 trained and compared models (Logistic Regression baseline, Random Forest primary; XGBoost only if time permits)
- Reproducible training with MLflow experiment tracking
- Saved, versioned model artifact plus `model_metadata.json` (includes the MLflow run ID)
- Working prediction API (FastAPI)
- Functional frontend (Streamlit)
- Dockerized app (docker compose)
- GitHub Actions CI running tests
- A data drift report (Evidently), generated from a *separate/shifted* batch, not the training set
- Deployed on AWS EC2 if time allows (optional, last task)

### Explicit non-goals
No Kubernetes, Airflow, SageMaker, deep learning, or complex cloud infra. Don't spend time on elaborate UI.

## 2. Team and prerequisites

Team of 4. Already available: Python/scikit-learn, Docker, MLflow experience, AWS account, GitHub.

| Member | Role | Owns |
|---|---|---|
| 1 | Data/ML | dataset, cleaning, feature extraction, evaluation |
| 2 | MLOps | MLflow, model artifacts, metadata, drift monitoring |
| 3 | Backend/UI | FastAPI, Streamlit, API integration |
| 4 | DevOps | Docker, compose, GitHub Actions, AWS, README |

Branches: `feature/data-ml`, `feature/mlops`, `feature/backend-frontend`, `feature/devops`. Merge to `main` via pull requests only.

## 3. Tech stack

Python **3.11**, pandas, numpy, scikit-learn, (xgboost optional), mlflow, fastapi, uvicorn, streamlit, joblib, tldextract, evidently (optional, pin the version), requests, pydantic, pytest, httpx.

`requirements.txt` (start with the core, add xgboost/evidently once the basics work):
```
pandas
numpy
scikit-learn
xgboost
mlflow
fastapi
uvicorn[standard]
streamlit
joblib
tldextract
evidently
requests
pydantic
python-multipart
pytest
httpx
```
After the first successful install, run `pip freeze` and pin versions so all four machines, Docker and CI match.

## 4. Dataset

- **Primary: LegitPhish** (Mendeley Data), roughly 101k labelled URLs (about 63.7k phishing, 37.5k legitimate). Treat these numbers as unverified until we inspect the file.
- **Fallback: UCI Phishing Websites** (11,055 rows, 30 features). Switch to it if getting LegitPhish working takes more than 30 minutes.
- **Never mix datasets.** Features and label definitions differ. Train and evaluate on one consistent dataset.
- Place the CSV at `data/raw/legitphish.csv`. It is **gitignored**; each teammate downloads it separately.
- **Label encoding:** published encoding is 0 = phishing, 1 = legitimate (verify against the actual file, including the label column name; the plan assumed `ClassLabel`). **Standardize internally to 1 = phishing, 0 = legitimate.**
- Note: UCI features are precomputed and not derivable from a raw URL, so using UCI as the fallback would require changing the feature/inference approach. Raise this with us before switching.

## 5. ML strategy

1. Load CSV, drop nulls, **drop exact duplicate URLs before splitting** (avoids leakage).
2. **Recompute features from the raw URL** with our own extractor, rather than using the dataset's precomputed columns, so the same code serves training and inference.
3. Standardize labels (1 = phishing).
4. Stratified **80/20** train/test split, fixed random seed (e.g. 42).
5. Train Logistic Regression (baseline) and Random Forest. Optionally XGBoost.
6. Metrics: accuracy, precision, recall, F1, ROC-AUC, confusion matrix, classification report.
7. **Model selection prioritizes phishing recall and F1, with a reasonable false-positive rate. Do not pick purely by accuracy.**
8. Save the best model with joblib to `artifacts/model/`, plus `model_metadata.json` (model name, MLflow run ID, metrics, feature list, training date, version).
9. Log everything to MLflow (`mlflow.set_experiment("PhishingShield")`, `mlflow.sklearn.autolog()`, tracking URI `sqlite:///mlflow.db`).

### Features (16, in this exact order, defined once as `FEATURES`)
`url_length, has_ip_address, dot_count, https_flag, url_entropy, token_count, subdomain_count, query_param_count, tld_length, path_length, has_hyphen_in_domain, number_of_digits, tld_popularity, suspicious_file_extension, domain_name_length, percentage_numeric_chars`

Implementation notes for `src/features/extractor.py`:
- `extract_features(url) -> dict` and `extract_batch(urls) -> DataFrame` (columns in `FEATURES` order). **This is the single source of truth for training and prediction.**
- Normalize URL (prepend `http://` if no scheme) before parsing.
- Use `tldextract.TLDExtract(suffix_list_urls=())` so it works offline (important in Docker/CI).
- `tld_popularity` = membership in a small common-TLD set (com, org, net, edu, gov, in, co, uk).
- `suspicious_file_extension` = path ends with .exe, .zip, .rar, .scr, .bat, .js, .apk.
- Handle weird input safely (empty string, None, malformed URL, IPv6, very long URL) without raising.
- These are lexical/structural heuristics, not definitive security signals. Say so in the UI and README.
- The starter extractor is not a byte-for-byte reproduction of the dataset authors' features. Compare its output with the published features on a sample of URLs and document the differences.

## 6. Target repo structure

```
PhishingShield/
├── data/
│   ├── raw/legitphish.csv          # gitignored
│   └── processed/                  # gitignored
├── artifacts/
│   ├── model/                      # gitignored (model.joblib, model_metadata.json)
│   └── reports/                    # gitignored (evaluation.json, drift_report.html)
├── src/
│   ├── __init__.py
│   ├── data/prepare.py             # load, clean, dedupe, label standardization, split
│   ├── features/extractor.py       # URL -> features (shared by train and inference)
│   ├── training/train.py           # train LR + RF, log to MLflow, save best model + metadata
│   ├── evaluation/evaluate.py      # metrics, confusion matrix -> artifacts/reports/evaluation.json
│   ├── monitoring/drift.py         # Evidently report -> artifacts/reports/drift_report.html
│   └── api/main.py                 # FastAPI service
├── frontend/app.py                 # Streamlit app (calls the API over HTTP)
├── tests/
│   ├── test_features.py
│   ├── test_api.py
│   └── test_model.py
├── .github/workflows/ci.yml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
├── .env.example
├── README.md
└── PROJECT_OVERVIEW.md             # this file
```
Every package directory under `src/` needs an `__init__.py`.

`.gitignore` must include: `.venv/ __pycache__/ *.pyc .env mlruns/ mlflow.db data/raw/* data/processed/* artifacts/model/* artifacts/reports/* .pytest_cache/ .DS_Store`

## 7. API contract (FastAPI, `src/api/main.py`)

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | API health/welcome |
| `/health` | GET | Status, including whether the model is loaded |
| `/predict` | POST | Classify a URL |
| `/model-info` | GET | Model version, metrics, MLflow run ID |

Request: `{"url": "https://example.com"}`

Response (values illustrative only):
```json
{
  "prediction": "LEGITIMATE",
  "phishing_probability": 0.08,
  "risk_level": "LOW",
  "model_version": "1"
}
```
- Load the model once at startup from `artifacts/model/`; return a clear 503 if it is missing rather than crashing.
- Validate input with pydantic; return 422/400 on empty or invalid URLs.
- Risk levels from probability (e.g. LOW < 0.4, MEDIUM 0.4 to 0.7, HIGH > 0.7; tune and document).
- Optionally return the extracted feature values so the UI can show the URL feature analysis.
- Run: `uvicorn src.api.main:app --reload --port 8000` (docs at `/docs`).

## 8. Frontend (Streamlit, `frontend/app.py`)

Sections: URL input, prediction result, phishing probability/risk, URL feature analysis, model performance dashboard (metrics + confusion matrix from `evaluation.json`), model version/experiment info.
- Talks to the API over HTTP; **read the API base URL from the `API_URL` env var** (default `http://127.0.0.1:8000`).
- Run: `streamlit run frontend/app.py`
- Keep it simple. Function over polish.

## 9. Monitoring

`src/monitoring/drift.py`: use training features as the reference and a **separate batch** of new/simulated URL features as current data. Write `artifacts/reports/drift_report.html`. Do not claim production drift detection from a training-vs-training report. Pin the Evidently version, since its API changed significantly between releases.

## 10. Docker and CI

- `Dockerfile`: `python:3.11-slim`, install requirements, copy `src/`, `frontend/`, `artifacts/`, set `PYTHONPATH=/app`, expose 8000 and 8501.
- `docker-compose.yml`: services `api` (port 8000) and `frontend` (port 8501, `API_URL=http://api:8000`, `depends_on: api`). Mount `./artifacts:/app/artifacts` so the model is available.
- `ci.yml`: on push/PR to `main`; checkout, Python 3.11, install, `pytest tests/ -v`. Add a Docker build step after tests pass. Fail the workflow on test failure.
- **CI must not depend on gitignored files** (dataset, trained model). Tests must use small in-repo fixtures, a tiny model trained on synthetic data inside the test, or mocks.

## 11. AWS EC2 (last, optional)

Ubuntu LTS, t3.small, 15-20 GB storage, Docker Compose. Security group: SSH (22) restricted to our IP; 8501 public for the demo; **8000 not public**; **never expose MLflow publicly**. Never commit AWS keys. Set a budget alert and stop the instance after the demo.
```
sudo apt update
sudo apt install -y docker.io docker-compose-v2 git
sudo systemctl enable --now docker
git clone <REPO_URL> && cd PhishingShield
sudo docker compose up --build -d
```
**Known gotcha:** `artifacts/model/` is gitignored, so a fresh `git clone` on EC2 has no model. Before `docker compose up`, copy the model there (`scp` the folder, or pull it from S3), or retrain on the instance. Decide this early; don't discover it during the demo.

## 12. Order of work (do not skip ahead)

**Do not build the frontend before the model can make predictions.**

**Milestone 1 (first ~4 hours):** a raw URL goes through `extract_features`, then a trained model, and gets classified.

Day 1 (data, ML, API):
- 09:00 shared setup: clone, venv, install, confirm dataset format and label column
- 10:00-13:00 in parallel: M1 cleaning + extractor, M2 training + MLflow, M3 FastAPI skeleton, M4 Docker + CI skeleton
- 13:00-14:00 integrate extractor + model + API
- 14:00-16:00 test predictions, model loading, API responses, metrics
- 16:00-18:00 Streamlit start; evaluation and drift reports

Day 2 (MLOps, AWS, demo):
- 09:00-11:00 finish Streamlit, FastAPI, compose
- 11:00-13:00 verify MLflow runs, artifacts, tests, drift report
- 13:00-15:00 AWS EC2 deployment
- 15:00-16:00 integration testing on the deployed app (predictions, errors)
- 16:00-18:00 README, architecture diagram, screenshots, demo prep

## 13. Setup commands

Windows PowerShell:
```
git clone <REPO_URL>
cd PhishingShield
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
Useful commands:
```
python -m src.training.train
python -m src.evaluation.evaluate
python -m src.monitoring.drift
mlflow ui --host 127.0.0.1 --port 5000      # keep local only
uvicorn src.api.main:app --reload --port 8000
streamlit run frontend/app.py
pytest tests/ -v
docker build -t phishing-shield .
docker compose up --build
```
(Run modules from the repo root with `python -m ...` so `src.*` imports resolve.)

## 14. Demo sequence (what the faculty should see)

1. Train Logistic Regression and Random Forest
2. Show both runs in MLflow
3. Compare evaluation metrics
4. Show the saved, versioned model and metadata
5. Run the app with Docker
6. Push a code change and show GitHub Actions running tests
7. Generate a drift report from a changed URL-feature batch
8. Show the deployed app on AWS

## 15. Working rules

- Keep it simple and shippable; prefer the smallest thing that satisfies the definition of done.
- One feature extractor, used everywhere. Never reimplement feature logic in the API or frontend.
- Never commit the dataset, `mlruns/`, `mlflow.db`, model binaries, `.env`, or any credentials.
- Write or update tests with each module. Run `pytest` before declaring something done.
- Don't invent dataset facts (column names, counts, label encoding). Inspect the real file first and report what you find.
- If a step is blocked (dataset format, AWS permissions, Evidently API), say so and propose the fallback instead of silently changing scope.
- Ask before changing the stack, dataset, or folder structure.

## 16. Current status

Nothing is implemented yet. First tasks: create the repo skeleton above, add `.gitignore` and `requirements.txt`, inspect the dataset, then build `src/features/extractor.py` and its tests.
