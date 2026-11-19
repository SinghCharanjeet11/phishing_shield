# Deeksha: Backend, Frontend & Deployment Engineer

**Branch:** `feature/app-deploy`
**Merges to `main` via PR only. Ask one teammate to review each PR.**

**Workload: ~10 h (~25% of the team, the lightest share)**

## Your mission
Turn the trained model into something people can use: the **FastAPI** prediction service, the main **Streamlit** screen, and the **Docker / docker compose** setup that runs the app locally for the demo.

(Originally the overview had a separate DevOps member. With 3 people, **Docker** moves to you and **CI** goes to Charan. The **model performance dashboard** and **testing of the running app** go to Arman. The optional AWS EC2 deployment is **not part of this project**.)

**Rule from the overview:** do not build the Streamlit UI before the model can make predictions through the API (Milestone 1). Until then, focus on the API skeleton and Docker.

---

## Files you own

| File | Purpose |
|---|---|
| `src/api/main.py` | FastAPI app: `/`, `/health`, `/predict`, `/model-info` |
| `src/api/schemas.py` | Pydantic request/response models (optional; can live in `main.py`) |
| `tests/test_api.py` | API tests using a tiny synthetic model (no real model in CI) |
| `frontend/app.py` | Streamlit UI (talks to the API over HTTP only); calls Arman's `render_dashboard()` from `frontend/dashboard.py` |
| `Dockerfile` | `python:3.11-slim` image for API + frontend |
| `docker-compose.yml` | `api` + `frontend` services |
| `.env.example` | `API_URL`, `MODEL_DIR`, `MLFLOW_TRACKING_URI` (no secrets) |

---

## Tasks

### T3.1 FastAPI skeleton (Day 1, 10:00-13:00)
- [ ] Endpoints:

| Endpoint | Method | Returns |
|---|---|---|
| `/` | GET | Welcome message + link to `/docs` |
| `/health` | GET | `{"status": "ok", "model_loaded": true, "model_version": "1"}` |
| `/predict` | POST | Prediction (contract below) |
| `/model-info` | GET | Contents of `model_metadata.json` (Charan's schema) |

- [ ] Request: `{"url": "https://example.com"}`, validated with pydantic: reject empty / whitespace-only / over-long (e.g. > 2,048 chars) with **422**.
- [ ] Response:
```json
{
  "url": "https://example.com",
  "prediction": "LEGITIMATE",
  "phishing_probability": 0.08,
  "risk_level": "LOW",
  "model_version": "1",
  "features": {"url_length": 19, "...": "all 16 features"}
}
```
- [ ] `prediction` = `"PHISHING"` if `phishing_probability >= threshold` (read `threshold` from metadata, default 0.5), else `"LEGITIMATE"`.
- [ ] `risk_level`: **LOW** < 0.4, **MEDIUM** 0.4-0.7, **HIGH** > 0.7. Keep the thresholds in one constant and document them.
- [ ] Load the model **once at startup** (FastAPI lifespan) from `MODEL_DIR` (default `artifacts/model`) using **Charan's `load_model()`**.
- [ ] If the model is missing: the app still starts, `/health` reports `model_loaded: false`, and `/predict` and `/model-info` return **503** with a clear message. Never crash.
- [ ] Features come from **Arman's `extract_features()` only**. Never reimplement feature logic in the API.
- [ ] Until Arman and Charan deliver, develop against a stub model trained on random data with the 16 `FEATURES` columns.
- [ ] Run: `uvicorn src.api.main:app --reload --port 8000` (docs at `/docs`).

### T3.2 `tests/test_api.py` (Day 1, 14:00-16:00)
- [ ] Use FastAPI `TestClient` (httpx).
- [ ] Fixture: train a tiny model on synthetic data, save it into `tmp_path` with Charan's `save_model()`, point `MODEL_DIR` at it (monkeypatch), then create the client.
- [ ] Test: `/` 200; `/health` 200 with `model_loaded: true`; `/predict` valid URL returns all response fields, probability in `[0, 1]`, risk level is one of LOW/MEDIUM/HIGH; empty URL returns 422; missing body returns 422; **no model → `/predict` returns 503**; `/model-info` returns metadata keys.
- [ ] **Never depend on the real model or dataset** (CI does not have them).

### T3.3 Dockerfile + docker compose (Day 1, 10:00-13:00 skeleton, finished Day 2 09:00-11:00)
- [ ] `Dockerfile`: `python:3.11-slim`, copy `requirements.txt` first and `pip install --no-cache-dir` (layer caching), then copy `src/`, `frontend/`; `ENV PYTHONPATH=/app`; `EXPOSE 8000 8501`.
- [ ] Add a `.dockerignore` (`.venv`, `data/`, `mlruns/`, `mlflow.db`, `.git`, `__pycache__`).
- [ ] `docker-compose.yml`:
  - `api`: same image, `uvicorn src.api.main:app --host 0.0.0.0 --port 8000`, port 8000, volume `./artifacts:/app/artifacts`, healthcheck on `/health`.
  - `frontend`: same image, `streamlit run frontend/app.py --server.address 0.0.0.0 --server.port 8501`, port 8501, `API_URL=http://api:8000`, `depends_on: api`.
- [ ] Tell Charan as soon as the Dockerfile builds, so they can add the CI Docker build step.
- [ ] Verify: `docker compose up --build` → UI at `localhost:8501` gets real predictions.

### T3.4 Streamlit frontend (Day 1, 16:00-18:00, finished Day 2 09:00-11:00), ONLY AFTER MILESTONE 1
- [ ] `API_URL` from env var (default `http://127.0.0.1:8000`). All data comes **from the API over HTTP**.
- [ ] Sections you build:
  1. URL input + "Analyze" button
  2. Result: PHISHING / LEGITIMATE, probability, colored risk badge
  3. URL feature analysis: table of the 16 features from the `/predict` response
- [ ] Below these, call `render_dashboard(API_URL)` from Arman's `frontend/dashboard.py` (metrics, confusion matrix, model comparison, model/experiment info). Until Arman delivers it, leave a placeholder line.
- [ ] Friendly errors: API down, 503 (no model), 422 (bad URL). The UI must never show a raw traceback.
- [ ] Visible disclaimer: "Lexical/structural heuristics, not a definitive security verdict."
- [ ] Function over polish. No time on styling.
- [ ] Tell Arman when `docker compose up` works so they can run the integration tests (Arman's T1.9). Fix the bugs they report.

---

## Interfaces you provide

| Interface | Consumer |
|---|---|
| API contract (`/predict` request/response, `/model-info`, `/health`) | Frontend (you), README (Arman), demo |
| `Dockerfile`, `docker-compose.yml` | Charan (CI Docker build step), everyone for running locally |
| `.env.example` | Everyone |
| `/model-info` response (includes `candidates`) | Arman (dashboard) |

## Interfaces you consume
- `extract_features`, `FEATURES` from **Arman** (`TASKS_ARMAN_DATA_ML.md`)
- `render_dashboard(...)` in `frontend/dashboard.py` from **Arman**
- `load_model()`, `save_model()`, `MODEL_DIR`, `model_metadata.json` schema from **Charan** (`TASKS_CHARAN_MLOPS_CI.md`)

## Definition of done (Deeksha)
- [ ] All 4 endpoints work with the real model; `/docs` shows them
- [ ] Missing model → 503, bad input → 422, no crashes
- [ ] `pytest tests/test_api.py -v` passes locally and in CI
- [ ] `docker compose up --build` serves the UI with real predictions
- [ ] Streamlit shows prediction, probability, risk and features, with Arman's dashboard embedded

## Your demo moments
Step 5 (run the app with Docker) and the live URL predictions.
