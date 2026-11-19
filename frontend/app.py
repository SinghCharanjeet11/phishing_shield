import os
import re
import requests
import streamlit as st

from dashboard import render_dashboard
# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="PhishingShield | URL Security",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;500;600;700&family=Raleway:wght@500;600;700;800&family=Space+Grotesk:wght@600;700&display=swap');

/* Body text: Open Sans (icon fonts and code blocks keep their own fonts) */
.stApp, .stApp *:not([data-testid="stIconMaterial"]):not([class*="material"]):not(code):not(pre) {
    font-family: 'Open Sans', sans-serif;
}

/* Headings and display values: Raleway */
.stApp h1, .stApp h1 *, .stApp h2, .stApp h2 *, .stApp h3, .stApp h3 *,
.stApp h4, .stApp h4 *, .scanner-title, .status-value, .result-value {
    font-family: 'Raleway', sans-serif !important;
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, #123b3b 0%, transparent 30%),
        radial-gradient(circle at 90% 20%, #172b55 0%, transparent 35%),
        #080e1a;
    color: #f8fafc;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

/* Hide Streamlit branding */
#MainMenu, footer {
    visibility: hidden;
}

/* Hero */
.hero {
    padding: 24px 0 8px 0;
    margin-bottom: 28px;
}

/* Hero heading: Space Grotesk */
.stApp .hero h1, .stApp .hero h1 * {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700;
    letter-spacing: -1px;
}

/* Hide the link icon Streamlit adds next to headings */
[data-testid="stHeaderActionElements"], .stApp h1 a, .stApp h2 a, .stApp h3 a {
    display: none !important;
}

.hero h1 {
    color: #ffffff;
    font-size: clamp(34px, 5vw, 58px);
    font-weight: 800;
    margin: 0 0 12px 0;
    line-height: 1.15;
}

.hero p {
    color: #b7cce0;
    font-size: 16px;
    line-height: 1.8;
}

/* Dashboard cards */
.status-card {
    background: linear-gradient(145deg, #14243a, #0e192a);
    border: 1px solid #2a405a;
    border-radius: 16px;
    padding: 23px;
}

.status-label {
    color: #91a9c3;
    font-size: 14px;
    font-weight: 600;
}

.status-value {
    color: #ffffff;
    font-size: 23px;
    font-weight: 800;
    margin-top: 17px;
}

/* Scanner */
.scanner {
    background: linear-gradient(145deg, #101e31, #0d1726);
    border: 1px solid #29415d;
    border-radius: 20px;
    padding: 30px;
    margin-top: 24px;
}

.scanner-title {
    color: #ffffff;
    font-size: 24px;
    font-weight: 800;
    margin-bottom: 8px;
}

.scanner-subtitle {
    color: #91b3d2;
    font-size: 14px;
    margin-bottom: 20px;
}

/* Inputs */
.stTextInput input {
    background: #101c2e !important;
    color: white !important;
    border: 1px solid #36516f !important;
    border-radius: 10px !important;
    padding: 15px !important;
}

.stTextInput input:focus {
    border-color: #22d3ee !important;
    box-shadow: 0 0 0 2px rgba(34,211,238,0.15) !important;
}

/* Buttons */
.stTextInput input::placeholder {
    color: #6f86a0 !important;
}

.stButton button, .stFormSubmitButton button {
    background: linear-gradient(90deg, #06b6d4, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 13px 25px !important;
    font-weight: 800 !important;
    transition: all 0.25s ease;
}

.stButton button p, .stFormSubmitButton button p {
    color: white !important;
}

.stButton button:hover, .stFormSubmitButton button:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(6,182,212,0.3);
}

/* Result panels */
.result-card {
    background: #111f32;
    border: 1px solid #304966;
    border-radius: 17px;
    padding: 25px;
    margin: 15px 0;
}

.result-heading {
    font-size: 14px;
    color: #91a9c3;
    font-weight: 600;
}

.result-value {
    color: white;
    font-size: 30px;
    font-weight: 800;
    margin-top: 10px;
}

.safe {
    color: #4ade80;
}

.danger {
    color: #fb7185;
}

.warning {
    color: #fbbf24;
}

/* Footer */
.footer {
    border-top: 1px solid #263c54;
    margin-top: 50px;
    padding-top: 22px;
    color: #8296ad;
    text-align: center;
    font-size: 12px;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Readable text on the dark background */
[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stWidgetLabel"] p {
    color: #c9d8e8 !important;
}

[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {
    color: #91a9c3 !important;
}

[data-testid="stMetricValue"], [data-testid="stMetricValue"] * {
    color: #ffffff !important;
}

[data-testid="stTable"] th, [data-testid="stTable"] td {
    color: #dbe7f3 !important;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# API HELPERS
# --------------------------------------------------

def get_api_health():
    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=3
        )

        if response.status_code == 200:
            return response.json()

        return None

    except requests.RequestException:
        return None


def analyze_url(url):
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"url": url},
            timeout=30
        )

        if response.status_code == 200:
            return response.json(), None

        try:
            error = response.json().get("detail", "Unknown API error")
        except ValueError:
            error = response.text

        return None, f"API Error ({response.status_code}): {error}"

    except requests.Timeout:
        return None, "The API request timed out. Please try again."

    except requests.RequestException:
        return None, "Cannot connect to the backend. Please start FastAPI."


def valid_url(url):
    if not url or len(url) > 2048:
        return False

    pattern = r"^https?://[^\s]+$"
    return bool(re.match(pattern, url, re.IGNORECASE))


# --------------------------------------------------
# HERO SECTION
# --------------------------------------------------

st.markdown("""
<div class="hero">
<h1>Secure your digital world</h1>
<p>Analyze suspicious URLs, identify phishing threats and make safer decisions before visiting a website.</p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LIVE API STATUS
# --------------------------------------------------

health = get_api_health()

api_connected = health is not None
model_loaded = bool(health and health.get("model_loaded"))
model_version = (
    health.get("model_version") or "Pending"
    if health else "Unavailable"
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="status-card">
        <div class="status-label">API</div>
        <div class="status-value">{"Online" if api_connected else "Offline"}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="status-card">
        <div class="status-label">Model</div>
        <div class="status-value">{"Loaded" if model_loaded else "Not loaded"}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="status-card">
        <div class="status-label">Model version</div>
        <div class="status-value">{model_version}</div>
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# URL SCANNER
# --------------------------------------------------

st.markdown("""
<div class="scanner">
<div class="scanner-title">URL Security Scanner</div>
<div class="scanner-subtitle">Enter a website address below to analyze its security risk.</div>
</div>
""", unsafe_allow_html=True)

with st.form("url_scanner", clear_on_submit=False):

    url = st.text_input(
        "Website URL",
        placeholder="https://example.com",
        max_chars=2048
    )

    submitted = st.form_submit_button(
        "ANALYZE WEBSITE",
        use_container_width=True
    )


# --------------------------------------------------
# ANALYSIS RESULTS
# --------------------------------------------------

if submitted:

    url = url.strip()

    if not valid_url(url):
        st.error(
            "Please enter a valid URL beginning with http:// or https://"
        )

    elif not api_connected:
        st.error(
            "Backend is offline. Start FastAPI and try again."
        )

    else:
        with st.spinner("Analyzing website security..."):

            result, error = analyze_url(url)

        if error:
            st.error(error)

            if not model_loaded:
                st.warning(
                    "The ML model is not available yet. "
                    "Once your team completes model training and integration, "
                    "website predictions will be available."
                )

        else:
            prediction = result.get("prediction", "UNKNOWN")
            probability = float(
                result.get("phishing_probability", 0)
            )
            risk = result.get("risk_level", "UNKNOWN")
            version = result.get("model_version", "Unknown")
            features = result.get("features", {})

            probability_percent = round(probability * 100, 2)

            is_phishing = prediction == "PHISHING"

            st.markdown("---")

            st.markdown("## Security Analysis Report")

            st.caption(f"Scanned URL: {url}")

            if is_phishing:
                st.error("PHISHING DETECTED")
            elif prediction == "LEGITIMATE":
                st.success("WEBSITE CLASSIFIED AS LEGITIMATE")
            else:
                st.warning("UNKNOWN PREDICTION")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-heading">Prediction</div>
                    <div class="result-value">
                        {prediction}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-heading">Risk level</div>
                    <div class="result-value">
                        {risk}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-heading">Phishing probability</div>
                    <div class="result-value">
                        {probability_percent}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("### Phishing Probability")

            st.progress(
                min(max(probability, 0.0), 1.0),
                text=f"{probability_percent}% phishing probability"
            )

            st.markdown("### Extracted Features")

            if features:
                st.dataframe(
                    [
                        {"Feature": key, "Value": value}
                        for key, value in features.items()
                    ],
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No feature details returned by the API.")

            st.caption(f"Model version: {version}")


# --------------------------------------------------
# INFORMATION SECTION
# --------------------------------------------------

st.markdown("---")

st.markdown("## Why PhishingShield?")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### Smart Detection

    Machine-learning-based URL classification
    designed to identify suspicious websites.
    """)

with col2:
    st.markdown("""
    ### Fast Analysis

    FastAPI-powered prediction interface
    for quick URL security checks.
    """)

with col3:
    st.markdown("""
    ### Model Monitoring

    Version-aware model integration and
    monitoring support.
    """)

# --------------------------------------------------
# MODEL PERFORMANCE DASHBOARD
# --------------------------------------------------

render_dashboard(API_URL)
# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("""
<div class="footer">
    <b>PhishingShield</b> &nbsp; | &nbsp;
    Intelligent Phishing URL Detection System
    <br><br>
    Built with Python, Streamlit, FastAPI and Machine Learning
    <br><br>
    Lexical/structural heuristics, not a definitive security verdict.
</div>
""", unsafe_allow_html=True)
