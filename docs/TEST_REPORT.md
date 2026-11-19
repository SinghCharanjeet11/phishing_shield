# PhishingShield: Integration Test Report (T1.9)

**Author:** Arman (Data & ML Engineer)  
**Date:** November 2026  
**Execution Environment:** Local Windows Environment & Docker Architecture  
**Model Tested:** Random Forest Classifier (v1, `artifacts/model/model.joblib`)  

---

## 1. Executive Summary

This report documents empirical integration tests evaluating the PhishingShield system across six mandatory operational scenarios. In compliance with project guidelines, all test results reflect actual observed behavior. Local container and Python execution were verified directly. The app is demonstrated locally with Docker compose; there is no cloud deployment.

---

## 2. Integration Test Matrix

| Environment | Scenario | Input / Test Case | Expected Behavior | Actual Observed Result | Status | Evidence / Notes |
|---|---|---|---|---|---|---|
| Local App | **Scenario 1:** Known Phishing URL | `https://keraekken-loagginnusa.godaddysites.com/` | Detect phishing, high risk probability | Prediction: **PHISHING**, Probability: `0.9450`, Risk: **HIGH** | **PASS** | Correctly flags typosquatting domain and suspicious pattern. |
| Local App | **Scenario 2:** Known Legitimate URL | `https://www.ekohealth.com` | Detect legitimate, low risk probability | Prediction: **LEGITIMATE**, Probability: `0.0000`, Risk: **LOW** | **PASS** | High-authority domain correctly classified with zero false-alarm probability. |
| Local App | **Scenario 3:** Empty / Malformed Input | `""` (Empty string) / `None` / Whitespace | Safe handling; 0-feature row fallback; API 422 error | Extractor returns all zeros; no unhandled exception. API validates input with Pydantic 422 response. | **PASS** | Robust against null/empty edge cases; UI displays user-friendly input validation warning. |
| Local App | **Scenario 4:** IP-Address URL | `http://192.168.0.1/login` | Extract `has_ip_address = 1`; flag phishing risk | `has_ip_address: 1`, Prediction: **PHISHING**, Probability: `1.0000`, Risk: **HIGH** | **PASS** | Direct IP hostname triggers maximum phishing probability. |
| Local App | **Scenario 5:** Over-long URL | 3,000-character URL string (`https://example.com/path?x=...`) | Compute features without memory leak or buffer crash | URL parsed cleanly; length = 3,025; execution completes in < 5ms. | **PASS** | Regex and tldextract handle large strings safely without CPU lock or overflow. |
| Local App | **Scenario 6:** API Restart / Missing Model | Model artifact removed or service restarted | 503 response on `/predict`; dashboard renders notice without traceback | `/health` returns `model_loaded: false`; `/predict` returns 503; Dashboard displays friendly warning. | **PASS** | Graceful degradation verified; no Python traceback exposed to end users. |

---

## 3. Detailed Scenario Observations & Defect Analysis

### Observation on Scenario 2 (Lexical Artifacts on Deep Subpaths)
* **Finding:** While typical legitimate homepages (e.g. `https://www.ekohealth.com`, `https://www.voabambara.com`) reliably score `P(phishing) = 0.0000`, deeply nested subpaths on legitimate websites (e.g., `https://docs.python.org/3/library/urllib.parse.html`) trigger elevated phishing scores ($P \approx 0.91$).
* **Root Cause:** In the raw LegitPhish dataset, legitimate URLs were predominantly sampled as domain roots, whereas phishing URLs were captured from URLHaus with active attack subpaths and scripts. As a result, high `path_length` and `token_count` are strong statistical correlates of phishing in this training set.
* **Mitigation & Documentation:** This confirms our core limitation statement: *URL lexical heuristics are a triage tool, not a definitive security verdict.* In production, lexical scoring must be paired with domain reputation lookup or DNS telemetry.

### Verification of Scenario 6 (Graceful API Degradation)
* When the FastAPI backend is stopped or `model.joblib` is absent:
  1. The API `/health` endpoint remains accessible and returns:
     ```json
     {"status": "ok", "model_loaded": false, "model_version": null}
     ```
  2. The `/predict` endpoint returns HTTP 503:
     ```json
     {"detail": "Trained model is not available. Please try again later."}
     ```
  3. The Streamlit performance dashboard (`frontend/dashboard.py`) catches the offline status and presents a clean notification:
     > *"Model comparison information is temporarily unavailable (API offline or unreachable)."*
  4. Zero unhandled tracebacks are visible to the end user.
