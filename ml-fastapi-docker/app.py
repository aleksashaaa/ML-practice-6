"""Credit Risk Studio — a Streamlit front-end for the German credit model.

Run the API first:
    uvicorn main:app --reload
Then start the UI:
    streamlit run app.py
"""

from __future__ import annotations

import html
import os
from typing import Any

import requests
import streamlit as st

import features

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")

STEPS = ("Applicant", "Loan", "Finances", "Obligations", "Review")
STEP_ICONS = ("01", "02", "03", "04", "05")
STEP_GROUP = {0: "Applicant", 1: "Loan", 2: "Finances", 3: "Obligations"}

FIELD_ORDER = (
    "purpose",
    "credit_amount",
    "duration_months",
    "checking_account",
    "savings_account",
    "credit_history",
    "installment_rate",
    "existing_credits",
    "age",
    "personal_status_sex",
    "job",
    "employment_since",
    "foreign_worker",
    "telephone",
    "other_debtors",
    "other_installment_plans",
    "property",
    "housing",
    "residence_since",
    "maintenance_for",
)

st.set_page_config(
    page_title="Credit Risk Studio",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --violet-950: #2e1065;
    --violet-900: #3b0f7a;
    --violet-700: #5b21b6;
    --violet-600: #6d28d9;
    --violet-500: #7c3aed;
    --violet-400: #8b5cf6;
    --violet-200: #ddd6fe;
    --violet-100: #ede9fe;
    --violet-50: #f7f4ff;
    --ink: #2c2350;
    --muted: #7a6f9b;
    --line: #ece5ff;
    --good: #059669;
    --good-soft: #e7f8f1;
    --bad: #e11d48;
    --bad-soft: #ffe9ef;
    --shadow: 0 26px 60px -38px rgba(76, 29, 149, 0.55);
}

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Manrope', 'Segoe UI', system-ui, sans-serif;
    color: var(--ink);
    font-size: 18px;
}

.stApp {
    background:
        radial-gradient(900px 420px at 8% -8%, #efe4ff 0%, rgba(239, 228, 255, 0) 60%),
        radial-gradient(760px 420px at 96% 4%, #f3e8ff 0%, rgba(243, 232, 255, 0) 55%),
        linear-gradient(180deg, #fbf9ff 0%, #f6f1ff 45%, #fcfbff 100%);
    background-attachment: fixed;
}

[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer, [data-testid="stDecoration"],
[data-testid="stStatusWidget"], [data-testid="stAppDeployButton"] {
    display: none !important;
}

/* Keep the sidebar reopen control visible (it lives in the toolbar). */
[data-testid="stToolbar"] { background: transparent !important; padding-right: 8px; }
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
}
[data-testid="stSidebarCollapseButton"] button,
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="stExpandSidebarButton"] {
    background: #fff !important;
    border: 1px solid #e6ddff !important;
    border-radius: 11px !important;
    color: var(--violet-600) !important;
    width: 36px;
    height: 36px;
    display: grid !important;
    place-items: center;
    box-shadow: 0 10px 22px -16px rgba(76, 29, 149, 0.7);
}
[data-testid="stSidebarCollapseButton"] button:hover,
[data-testid="stSidebarCollapsedControl"] button:hover,
[data-testid="stExpandSidebarButton"]:hover {
    background: #faf7ff !important;
    border-color: #c9b8f8 !important;
}

.block-container {
    max-width: 1200px;
    padding-top: 1.6rem;
    padding-bottom: 3.5rem;
}

/* ---------- Hero ---------- */
.hero {
    position: relative;
    overflow: hidden;
    border-radius: 30px;
    padding: 38px 42px 34px;
    color: #fff;
    background: linear-gradient(125deg, #3b0f7a 0%, #6d28d9 48%, #a855f7 100%);
    box-shadow: 0 40px 80px -48px rgba(76, 29, 149, 0.95);
    margin-bottom: 1.6rem;
}
.hero::after {
    content: "";
    position: absolute;
    width: 420px; height: 420px;
    right: -120px; top: -170px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(255,255,255,0.28) 0%, rgba(255,255,255,0) 65%);
}
.hero::before {
    content: "";
    position: absolute;
    width: 260px; height: 260px;
    right: 180px; bottom: -170px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(255,255,255,0.16) 0%, rgba(255,255,255,0) 70%);
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    border-radius: 999px;
    background: rgba(255,255,255,0.16);
    border: 1px solid rgba(255,255,255,0.28);
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.hero h1 {
    position: relative;
    z-index: 1;
    margin: 16px 0 8px;
    font-size: 2.55rem;
    line-height: 1.1;
    font-weight: 800;
    letter-spacing: -0.02em;
}
.hero p {
    position: relative;
    z-index: 1;
    margin: 0 0 20px;
    max-width: 640px;
    font-size: 1.1rem;
    color: rgba(255,255,255,0.88);
}
.hero-chips { position: relative; z-index: 1; display: flex; flex-wrap: wrap; gap: 10px; }
.hero-chips span {
    padding: 7px 14px;
    border-radius: 999px;
    background: rgba(255,255,255,0.14);
    border: 1px solid rgba(255,255,255,0.24);
    font-size: 0.9rem;
    font-weight: 600;
}

/* ---------- Stepper ---------- */
.stepper {
    background: #fff;
    border: 1px solid var(--line);
    border-radius: 22px;
    padding: 18px 24px 16px;
    box-shadow: var(--shadow);
    margin-bottom: 1.4rem;
}
.stepper-row { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 6px; }
.stepper-item { display: flex; flex-direction: column; align-items: center; gap: 7px; min-width: 0; text-align: center; }
.stepper-dot {
    flex: 0 0 auto;
    width: 38px; height: 38px;
    border-radius: 50%;
    display: grid; place-items: center;
    font-size: 0.82rem; font-weight: 800;
    background: var(--violet-50);
    color: #a79bc9;
    border: 1.5px solid #e7dfff;
    transition: all .2s ease;
}
.stepper-item.done .stepper-dot { background: var(--violet-100); color: var(--violet-600); border-color: var(--violet-200); }
.stepper-item.active .stepper-dot {
    background: linear-gradient(135deg, #7c3aed, #a855f7);
    color: #fff; border-color: transparent;
    box-shadow: 0 10px 20px -10px rgba(124,58,237,.9);
}
.stepper-text { display: flex; flex-direction: column; min-width: 0; }
.stepper-text b { font-size: 0.9rem; font-weight: 700; color: var(--muted); line-height: 1.2; }
.stepper-item.active .stepper-text b, .stepper-item.done .stepper-text b { color: var(--ink); }
.stepper-text small { display: none; }
.stepper-progress { margin-top: 14px; height: 6px; border-radius: 999px; background: #f1ebff; overflow: hidden; }
.stepper-progress > div {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #8b5cf6, #c084fc);
    transition: width .3s ease;
}

/* ---------- Cards ---------- */
.block-container div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #fff;
    border: 1px solid var(--line);
    border-radius: 24px;
    box-shadow: var(--shadow);
    padding: 1.35rem 1.4rem;
}
.block-container div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stVerticalBlockBorderWrapper"] {
    box-shadow: none;
    padding: 0.6rem 0.8rem;
}

.section-title { font-size: 1.4rem; font-weight: 800; letter-spacing: -0.01em; margin: 0 0 4px; }
.section-sub { color: var(--muted); font-size: 1rem; margin: 0 0 18px; }
.group-label {
    display: inline-block;
    font-size: 0.78rem; font-weight: 800; letter-spacing: .1em; text-transform: uppercase;
    color: var(--violet-600); background: var(--violet-50);
    border: 1px solid var(--violet-100);
    padding: 5px 11px; border-radius: 999px; margin-bottom: 10px;
}

/* ---------- Inputs ---------- */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    border-radius: 13px !important;
    border-color: #e6ddff !important;
    background: #fdfcff !important;
}
div[data-baseweb="select"] > div:hover,
div[data-baseweb="input"] > div:hover { border-color: #b9a6f7 !important; }
div[data-baseweb="select"] > div { min-height: 46px; }
.stSelectbox label p, .stNumberInput label p, .stSlider label p {
    font-weight: 700 !important;
    color: #40346b !important;
    font-size: 1rem !important;
}
div[data-baseweb="select"], div[data-baseweb="input"] { font-size: 1rem !important; }
[data-testid="stExpander"] summary { font-size: 1rem !important; font-weight: 700 !important; }
.stSlider [data-testid="stTickBarMin"], .stSlider [data-testid="stTickBarMax"] { color: #a79bc9; }

/* ---------- Buttons ---------- */
.stButton > button, .stLinkButton > a {
    border-radius: 14px !important;
    font-family: 'Manrope', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.98rem !important;
    padding: 0.66rem 1.25rem !important;
    transition: transform .15s ease, box-shadow .15s ease, background .15s ease;
}
[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #6d28d9, #a855f7) !important;
    border: none !important;
    color: #fff !important;
    box-shadow: 0 18px 34px -18px rgba(109, 40, 217, 0.95) !important;
}
[data-testid="stBaseButton-primary"]:hover {
    background: linear-gradient(135deg, #5b21b6, #9333ea) !important;
    transform: translateY(-1px);
}
[data-testid="stBaseButton-secondary"] {
    background: #fff !important;
    color: var(--violet-700) !important;
    border: 1px solid #ddd0ff !important;
}
[data-testid="stBaseButton-secondary"]:hover { border-color: #b9a6f7 !important; background: #faf7ff !important; }
.stLinkButton > a { border: 1px solid #ddd0ff !important; color: var(--violet-700) !important; background: #fff !important; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid var(--line); }
[data-testid="stSidebar"] .block-container { padding-top: 1.6rem; }
[data-testid="stSidebar"] .stButton > button { padding: 0.5rem 0.9rem !important; font-size: 0.92rem !important; }
.brand { display: flex; align-items: center; gap: 12px; margin-bottom: 4px; }
.brand-mark {
    width: 42px; height: 42px; border-radius: 13px;
    display: grid; place-items: center; font-size: 1.15rem;
    background: linear-gradient(135deg, #6d28d9, #c084fc);
    box-shadow: 0 12px 24px -12px rgba(109,40,217,.9);
}
.brand b { font-size: 1.08rem; display: block; line-height: 1.2; }
.brand small { color: var(--muted); font-size: 0.82rem; }
.status {
    display: inline-flex; align-items: center; gap: 8px;
    font-size: 0.88rem; font-weight: 700;
    padding: 6px 12px; border-radius: 999px; margin: 10px 0 4px;
}
.status.ok { color: var(--good); background: var(--good-soft); }
.status.down { color: var(--bad); background: var(--bad-soft); }
.status .dot { width: 8px; height: 8px; border-radius: 50%; background: currentColor; }
.sidebar-heading {
    font-size: 0.78rem; font-weight: 800; letter-spacing: .11em; text-transform: uppercase;
    color: #a79bc9; margin: 22px 0 8px;
}
.summary { display: flex; flex-direction: column; gap: 9px; }
.srow { display: flex; align-items: baseline; justify-content: space-between; gap: 14px; font-size: 0.94rem; }
.srow span { color: var(--muted); }
.srow b { font-weight: 700; color: var(--ink); text-align: right; }
.sdivider { height: 1px; background: #f2edff; margin: 6px 0; }

/* ---------- Review ---------- */
.review-card {
    background: #fff; border: 1px solid var(--line); border-radius: 20px;
    padding: 18px 20px; box-shadow: var(--shadow); height: 100%;
}
.review-card h4 { margin: 0 0 12px; font-size: 1.08rem; font-weight: 800; display: flex; align-items: center; gap: 8px; }
.review-card h4 em {
    font-style: normal; font-size: 0.74rem; font-weight: 800; letter-spacing: .09em;
    text-transform: uppercase; color: var(--violet-600);
    background: var(--violet-50); border: 1px solid var(--violet-100);
    border-radius: 999px; padding: 3px 9px;
}
.review-card .srow { font-size: 0.92rem; padding: 4px 0; border-bottom: 1px dashed #f3eeff; }
.review-card .srow:last-child { border-bottom: none; }

/* ---------- Result ---------- */
.verdict {
    border-radius: 24px;
    padding: 26px 28px;
    border: 1px solid transparent;
    height: 100%;
}
.verdict.good { background: linear-gradient(135deg, #ecfdf5, #f6fffb); border-color: #b7f0d8; }
.verdict.bad { background: linear-gradient(135deg, #fff1f5, #fff8fa); border-color: #ffc9d8; }
.verdict .eyebrow {
    font-size: .72rem; font-weight: 800; letter-spacing: .11em; text-transform: uppercase;
    color: var(--muted); margin-bottom: 8px;
}
.verdict h2 { margin: 0 0 8px; font-size: 2rem; font-weight: 800; letter-spacing: -0.02em; }
.verdict.good h2 { color: #047857; }
.verdict.bad h2 { color: #be123c; }
.verdict p { margin: 0 0 18px; color: #5c5480; font-size: 1rem; max-width: 520px; }
.pill-row { display: flex; flex-wrap: wrap; gap: 8px; }
.pill {
    display: inline-flex; align-items: center; gap: 7px;
    padding: 7px 13px; border-radius: 999px;
    background: rgba(255,255,255,0.8); border: 1px solid #ece5ff;
    font-size: 0.88rem; font-weight: 700; color: #4b3f75;
}
.pill b { color: var(--ink); }

.score-wrap { display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 10px; padding: 6px 0; }.score-ring {
    width: 176px; height: 176px; border-radius: 50%;
    background: conic-gradient(var(--ring) calc(var(--p) * 1%), #efe9ff 0);
    display: grid; place-items: center;
    box-shadow: inset 0 0 0 1px rgba(109,40,217,.04);
}
.score-inner {
    width: 138px; height: 138px; border-radius: 50%; background: #fff;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    box-shadow: 0 12px 30px -22px rgba(76,29,149,.8);
}
.score-inner b { font-size: 2.2rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1; }
.score-inner span { font-size: 0.8rem; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; margin-top: 6px; }
.score-note { font-size: 0.84rem; color: var(--muted); font-weight: 600; }
@media (min-width: 901px) {
    .score-wrap { min-height: 340px; height: 100%; }
}

.risk-scale { position: relative; height: 10px; border-radius: 999px; margin: 22px 0 6px;
    background: linear-gradient(90deg, #34d399, #fbbf24 55%, #fb7185); }
.risk-scale .marker {
    position: absolute; top: 50%; width: 16px; height: 16px; border-radius: 50%;
    background: #fff; border: 3px solid #3b0f7a; transform: translate(-50%, -50%);
    box-shadow: 0 4px 12px rgba(59,15,122,.35);
}
.risk-scale-labels { display: flex; justify-content: space-between; font-size: 0.8rem; color: var(--muted); font-weight: 600; }

/* ---------- Factors ---------- */
.factor { margin-bottom: 15px; }
.factor-top { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; margin-bottom: 6px; }
.factor-label { font-size: 0.94rem; font-weight: 700; color: #40346b; }
.factor-label small { font-weight: 600; color: var(--muted); }
.factor-value { font-size: 0.88rem; font-weight: 800; font-variant-numeric: tabular-nums; }
.factor-value.up { color: var(--bad); }
.factor-value.down { color: var(--good); }
.factor-track { height: 9px; border-radius: 999px; background: #f4efff; overflow: hidden; }
.factor-fill { height: 100%; border-radius: 999px; }
.factor-fill.up { background: linear-gradient(90deg, #fb7185, #e11d48); }
.factor-fill.down { background: linear-gradient(90deg, #6ee7b7, #059669); }
.legend { display: flex; flex-wrap: wrap; gap: 16px; margin: 4px 0 18px; font-size: 0.86rem; color: var(--muted); font-weight: 600; }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 6px; }

/* ---------- Metrics ---------- */
.metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 14px; }
.metric {
    background: #fff; border: 1px solid var(--line); border-radius: 18px;
    padding: 16px 18px; box-shadow: var(--shadow); height: 100%;
}
.metric span { display: block; font-size: 0.8rem; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; color: #a79bc9; margin-bottom: 6px; }
.metric b { font-size: 1.7rem; font-weight: 800; letter-spacing: -0.02em; color: var(--violet-900); }
.cm { border-collapse: separate; border-spacing: 6px; width: 100%; max-width: 440px; font-size: 0.92rem; }
.cm th { font-size: 0.78rem; text-transform: uppercase; letter-spacing: .07em; color: #a79bc9; font-weight: 800; padding: 4px 6px; text-align: center; }
.cm th.rowhead { text-align: left; }
.cm td { padding: 12px 10px; text-align: center; border-radius: 12px; font-weight: 800; font-size: 1.08rem; }
.cm td.rowhead { text-align: left; font-weight: 700; color: var(--muted); font-size: 0.88rem; background: transparent; padding-left: 0; }
.cm td.hit { background: #f1ecff; color: var(--violet-900); }
.cm td.miss { background: #fff4f7; color: #b91c4a; }
.footnote { color: #a79bc9; font-size: 0.84rem; margin-top: 18px; line-height: 1.6; }
.empty-state {
    border: 1px dashed #d9ccff; border-radius: 20px; padding: 26px;
    text-align: center; color: var(--muted); background: #fdfcff; font-weight: 600;
}
div[data-testid="stAlert"] { border-radius: 16px; }

@media (max-width: 900px) {
    .hero { padding: 28px 24px; }
    .hero h1 { font-size: 2rem; }
    .stepper-text b { font-size: 0.8rem; }
    .stepper-dot { width: 34px; height: 34px; }
    .block-container [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; gap: 12px !important; }
    .block-container [data-testid="stHorizontalBlock"] > div {
        flex: 1 1 100% !important;
        width: 100% !important;
        min-width: 100% !important;
    }
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def init_state() -> None:
    if "step" not in st.session_state:
        st.session_state["step"] = 0
    if "form" not in st.session_state:
        st.session_state["form"] = dict(features.DEFAULT_INPUT)
    if "form_version" not in st.session_state:
        st.session_state["form_version"] = 0


def reset_state() -> None:
    st.session_state["form"] = dict(features.DEFAULT_INPUT)
    st.session_state["form_version"] += 1
    st.session_state["step"] = 0
    st.session_state.pop("result", None)
    st.session_state.pop("result_payload", None)


def sync_feature(widget_key: str, name: str) -> None:
    st.session_state["form"][name] = st.session_state[widget_key]


def goto(step: int) -> None:
    st.session_state["step"] = max(0, min(step, len(STEPS) - 1))
    st.session_state.pop("result", None)


@st.cache_data(ttl=10, show_spinner=False)
def api_available(url: str) -> bool:
    try:
        return requests.get(f"{url}/", timeout=2).ok
    except requests.exceptions.RequestException:
        return False


@st.cache_data(ttl=60, show_spinner=False)
def fetch_metrics(url: str) -> dict[str, Any] | None:
    try:
        response = requests.get(f"{url}/metrics", timeout=4)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None


def escape(value: Any) -> str:
    return html.escape(str(value))


def hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <span class="hero-badge">UCI Statlog · German Credit Data</span>
            <h1>Credit Risk Studio</h1>
            <p>Score a credit application in four short steps. Friendly categories,
               a clear risk verdict and a transparent breakdown of what drives it.</p>
            <div class="hero-chips">
                <span>Logistic Regression</span>
                <span>20 applicant features</span>
                <span>1,000 real applications</span>
                <span>Explainable by design</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stepper(step: int) -> None:
    parts: list[str] = []
    for index, name in enumerate(STEPS):
        state = "active" if index == step else ("done" if index < step else "")
        marker = "✓" if index < step else STEP_ICONS[index]
        parts.append(
            f'<div class="stepper-item {state}">'
            f'<span class="stepper-dot">{marker}</span>'
            f'<span class="stepper-text"><b>{name}</b></span>'
            f"</div>"
        )

    progress = int(step / (len(STEPS) - 1) * 100)
    st.markdown(
        f'<div class="stepper"><div class="stepper-row">{"".join(parts)}</div>'
        f'<div class="stepper-progress"><div style="width:{progress}%"></div></div></div>',
        unsafe_allow_html=True,
    )


def summary_rows(items: list[tuple[str, Any]]) -> str:
    rows = "".join(
        f'<div class="srow"><span>{escape(label)}</span><b>{escape(value)}</b></div>'
        for label, value in items
    )
    return f'<div class="summary">{rows}</div>'


def render_feature(feature: features.Feature) -> None:
    store = st.session_state["form"]
    version = st.session_state["form_version"]
    key = f"f_{feature.name}_{version}"
    current = store[feature.name]
    sync = {"on_change": sync_feature, "args": (key, feature.name)}

    if feature.kind == "categorical":
        labels = {category.key: category.label for category in feature.categories}
        options = list(labels)
        value = st.selectbox(
            feature.label,
            options,
            index=options.index(current),
            key=key,
            format_func=lambda option: labels[option],
            help=feature.help,
            **sync,
        )
    elif feature.value_labels:
        options = list(feature.value_labels)
        value = st.selectbox(
            feature.label,
            options,
            index=options.index(int(current)),
            key=key,
            format_func=lambda option: feature.value_labels[option],
            help=feature.help,
            **sync,
        )
    elif feature.name == "credit_amount":
        value = st.number_input(
            f"{feature.label} (DM)",
            min_value=feature.minimum,
            max_value=feature.maximum,
            value=int(current),
            step=feature.step,
            key=key,
            help=feature.help,
            **sync,
        )
    else:
        unit = f" ({feature.unit})" if feature.unit else ""
        value = st.slider(
            f"{feature.label}{unit}",
            min_value=feature.minimum,
            max_value=feature.maximum,
            value=int(current),
            step=feature.step,
            key=key,
            help=feature.help,
            **sync,
        )

    store[feature.name] = value


def group_features(group: str) -> list[features.Feature]:
    items = [feature for feature in features.FEATURES if feature.group == group]
    return sorted(items, key=lambda feature: FIELD_ORDER.index(feature.name))


def current_payload() -> dict[str, Any]:
    return dict(st.session_state["form"])


def factor_bars(factors: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for factor in factors:
        direction = "up" if factor["direction"] == "increases" else "down"
        width = max(float(factor["strength"]) * 100, 5)
        sign = "+" if factor["contribution"] >= 0 else "−"
        blocks.append(
            '<div class="factor">'
            f'<div class="factor-top"><span class="factor-label">{escape(factor["label"])} '
            f'<small>· {escape(factor["value"])}</small></span>'
            f'<span class="factor-value {direction}">{sign}{abs(factor["contribution"]):.2f}</span></div>'
            f'<div class="factor-track"><div class="factor-fill {direction}" '
            f'style="width:{width:.1f}%"></div></div></div>'
        )
    return "".join(blocks)


def render_result(result: dict[str, Any], payload: dict[str, Any]) -> None:
    probability = float(result["risk_probability"])
    percent = round(probability * 100)
    is_bad = int(result["prediction"]) == 1
    colour = "#e11d48" if is_bad else "#059669"
    verdict = "High credit risk" if is_bad else "Low credit risk"
    tone = "bad" if is_bad else "good"
    message = (
        "The model sees a high chance of default. Consider a smaller amount, "
        "a shorter term or additional collateral."
        if is_bad
        else "The application looks solid. The model expects this credit to be repaid."
    )
    confidence = round(float(result["confidence"]) * 100)
    marker = min(max(percent, 3), 97)

    with st.container(border=True):
        left, right = st.columns([1.7, 1], gap="large")
        with left:
            st.markdown(
                f"""
                <div class="verdict {tone}">
                    <div class="eyebrow">Model verdict</div>
                    <h2>{verdict}</h2>
                    <p>{message}</p>
                    <div class="pill-row">
                        <span class="pill">Default probability <b>{probability * 100:.1f}%</b></span>
                        <span class="pill">Confidence <b>{confidence}%</b></span>
                        <span class="pill">Threshold <b>50%</b></span>
                    </div>
                    <div class="risk-scale"><div class="marker" style="left:{marker}%"></div></div>
                    <div class="risk-scale-labels"><span>Low risk</span><span>High risk</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with right:
            st.markdown(
                f"""
                <div class="score-wrap">
                    <div class="score-ring" style="--p:{percent}; --ring:{colour};">
                        <div class="score-inner"><b style="color:{colour}">{percent}%</b>
                        <span>default risk</span></div>
                    </div>
                    <div class="score-note">Probability of bad credit</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    factors = result.get("explanations", [])
    top = [factor for factor in factors if abs(factor["contribution"]) >= 0.01][:8]
    rest = [factor for factor in factors if factor not in top]

    st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(
            '<div class="section-title">What drives this result</div>'
            '<div class="section-sub">Each bar shows how a factor moves the default risk '
            "for this applicant.</div>"
            '<div class="legend"><span><i style="background:#e11d48"></i>Increases risk</span>'
            '<span><i style="background:#059669"></i>Reduces risk</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(factor_bars(top), unsafe_allow_html=True)
        if rest:
            with st.expander(f"Show all {len(factors)} factors"):
                st.markdown(factor_bars(rest), unsafe_allow_html=True)
        st.markdown(
            '<div class="footnote">Contributions come from the logistic regression '
            "coefficients: the bar length is the factor's weighted effect on the log-odds "
            "of a bad credit, aggregated across one-hot encoded categories.</div>",
            unsafe_allow_html=True,
        )


def render_metrics(url: str) -> None:
    st.markdown('<div style="height:26px"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Model performance</div>'
        '<div class="section-sub">Logistic Regression evaluated on the held-out 20% '
        "of the dataset.</div>",
        unsafe_allow_html=True,
    )

    data = fetch_metrics(url)
    if data is None:
        st.markdown(
            '<div class="empty-state">Model metrics are unavailable while the API is offline.</div>',
            unsafe_allow_html=True,
        )
        return

    metrics = data["metrics"]
    cards = (
        ("Accuracy", metrics["accuracy"]),
        ("Precision", metrics["precision"]),
        ("Recall", metrics["recall"]),
        ("F1 score", metrics["f1"]),
        ("ROC-AUC", metrics["roc_auc"]),
    )
    cards_html = "".join(
        f'<div class="metric"><span>{label}</span><b>{value * 100:.1f}%</b></div>'
        for label, value in cards
    )
    st.markdown(f'<div class="metric-grid">{cards_html}</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
    matrix = data["confusion_matrix"]
    left, right = st.columns([1, 1.1], gap="large")
    with left:
        st.markdown(
            f"""
            <table class="cm">
                <tr><th class="rowhead">Confusion matrix</th><th>Predicted good</th><th>Predicted bad</th></tr>
                <tr><td class="rowhead">Actual good</td>
                    <td class="hit">{matrix["true_good_pred_good"]}</td>
                    <td class="miss">{matrix["true_good_pred_bad"]}</td></tr>
                <tr><td class="rowhead">Actual bad</td>
                    <td class="miss">{matrix["true_bad_pred_good"]}</td>
                    <td class="hit">{matrix["true_bad_pred_bad"]}</td></tr>
            </table>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f'<div class="footnote">Dataset: {escape(data["dataset"]["name"])} · '
            f'{data["dataset"]["total"]} applications '
            f'({data["dataset"]["good"]} good, {data["dataset"]["bad"]} bad).<br>'
            f'Train {data["split"]["train"]} / test {data["split"]["test"]}, '
            "stratified split with random_state=42.</div>",
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
init_state()
hero()
form = st.session_state["form"]

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-mark">💳</div>
            <div><b>Credit Risk Studio</b><small>German credit scoring</small></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    online = api_available(API_URL)
    st.markdown(
        f'<div class="status {"ok" if online else "down"}"><span class="dot"></span>'
        f'API {"online" if online else "offline"}</div>',
        unsafe_allow_html=True,
    )
    if st.button(
        "Re-check connection",
        use_container_width=True,
        help=(
            "Sends a test request to the FastAPI server and refreshes the "
            "online / offline badge above."
        ),
    ):
        api_available.clear()
        st.rerun()

    st.markdown('<div class="sidebar-heading">Application summary</div>', unsafe_allow_html=True)
    st.markdown(
        summary_rows(
            [
                ("Age", features.compact_value("age", form["age"])),
                ("Job", features.compact_value("job", form["job"])),
                ("Purpose", features.compact_value("purpose", form["purpose"])),
                ("Amount", features.compact_value("credit_amount", form["credit_amount"])),
                ("Duration", features.compact_value("duration_months", form["duration_months"])),
                ("Checking", features.compact_value("checking_account", form["checking_account"])),
                ("Credit history", features.compact_value("credit_history", form["credit_history"])),
            ]
        ),
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-heading">About the model</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="footnote">Logistic Regression with standardised numeric features '
        "and one-hot encoded categories. Outputs the probability of a bad credit.</div>",
        unsafe_allow_html=True,
    )
    st.link_button("Open API docs", f"{API_URL}/docs", use_container_width=True)
    if st.button("Start over", use_container_width=True):
        reset_state()
        st.rerun()

step = int(st.session_state["step"])
stepper(step)

if step < 4:
    group = STEP_GROUP[step]
    with st.container(border=True):
        st.markdown(
            f'<span class="group-label">{escape(group)}</span>'
            f'<div class="section-title">{escape(group)} details</div>'
            '<div class="section-sub">Choose the option that best describes the applicant. '
            "Every field has a tooltip with the dataset definition.</div>",
            unsafe_allow_html=True,
        )
        items = group_features(group)
        left, right = st.columns(2, gap="large")
        for index, feature in enumerate(items):
            with left if index % 2 == 0 else right:
                render_feature(feature)
        if step == 1:
            st.markdown(
                '<div class="footnote">Tip: a shorter term and a smaller amount usually '
                "lower the predicted risk.</div>",
                unsafe_allow_html=True,
            )

    st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)
    back, forward = st.columns([1, 1], gap="medium")
    if back.button("← Back", use_container_width=True, disabled=step == 0):
        goto(step - 1)
        st.rerun()
    if forward.button("Continue →", type="primary", use_container_width=True):
        goto(step + 1)
        st.rerun()

else:
    with st.container(border=True):
        st.markdown(
            '<div class="section-title">Review the application</div>'
            '<div class="section-sub">Check the details below, then score the applicant.</div>',
            unsafe_allow_html=True,
        )
        payload = current_payload()
        groups = list(features.GROUP_ORDER)
        for row_index in range(0, len(groups), 2):
            columns = st.columns(2, gap="medium")
            for column, group_name in zip(columns, groups[row_index : row_index + 2]):
                rows = "".join(
                    f'<div class="srow"><span>{escape(feature.label)}</span>'
                    f'<b>{escape(features.display_value(feature.name, payload[feature.name]))}</b></div>'
                    for feature in group_features(group_name)
                )
                column.markdown(
                    f'<div class="review-card"><h4>{escape(group_name)}'
                    f"<em>{len(group_features(group_name))} fields</em></h4>{rows}</div>",
                    unsafe_allow_html=True,
                )
            st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
        if st.button("Assess credit risk", type="primary", use_container_width=True):
            try:
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
                response.raise_for_status()
                st.session_state["result"] = response.json()
                st.session_state["result_payload"] = payload
            except requests.exceptions.RequestException as error:
                st.session_state.pop("result", None)
                st.error("Could not reach the prediction API.")
                st.code(
                    "uvicorn main:app --reload\n\n"
                    f"API_URL={API_URL}\n\n{error}",
                    language="bash",
                )

    if "result" in st.session_state:
        st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
        render_result(st.session_state["result"], st.session_state.get("result_payload", payload))

    st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)
    back, forward = st.columns([1, 1], gap="medium")
    if back.button("← Back", use_container_width=True):
        goto(step - 1)
        st.rerun()

render_metrics(API_URL)

st.markdown(
    '<div class="footnote" style="text-align:center;margin-top:28px">'
    "Educational demo on the UCI Statlog German Credit dataset. "
    "Predictions are not financial advice.</div>",
    unsafe_allow_html=True,
)
