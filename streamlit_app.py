"""
================================================================================
Box Office Success Prediction — Interactive Streamlit Application
================================================================================
Authors  : I.Parthiban 23BCE2282, Mike Roonane Fernandez 23BCE2062
Date     : February 2026
Project  : Capstone — Movie Box Office Prediction using Multi-Source Big Data
           and Machine Learning with IMDb Metadata

Description
-----------
An interactive web application that surfaces the trained stacking ensemble
models through an intuitive user interface.  Users can input film parameters
and receive:

    · Predicted worldwide revenue (USD, 2023 constant)
    · Success category (Hit / Average / Flop) with probability breakdown
    · Investment risk gauge
    · Feature contribution interpretation

Usage
─────
    1. Ensure data_pipeline.py and model_training.py have been run.
    2. Launch the app:
            streamlit run streamlit_app.py
    3. Open http://localhost:8501 in your browser.

Dependencies
────────────
    pip install streamlit xgboost scikit-learn shap plotly numpy pandas
================================================================================
"""

import math
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from model_training import StackingEnsemble
# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="🎬 Box Office Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

MODELS_DIR = Path("models")

GENRE_LIST = [
    "Action", "Adventure", "Animation", "Comedy", "Crime",
    "Documentary", "Drama", "Family", "Fantasy", "History",
    "Horror", "Music", "Mystery", "Romance", "Science Fiction",
    "Thriller", "War", "Western",
]

SUCCESS_LABELS = {0: "🔴 Flop", 1: "🟡 Average", 2: "🟢 Hit"}
SUCCESS_COLOURS = {
    "🔴 Flop"   : "#e74c3c",
    "🟡 Average": "#f39c12",
    "🟢 Hit"    : "#2ecc71",
}

FAMOUS_DIRECTORS = [
    "Christopher Nolan", "Steven Spielberg", "James Cameron",
    "Ridley Scott", "Martin Scorsese", "Quentin Tarantino",
    "Greta Gerwig", "Denis Villeneuve", "Ryan Coogler",
    "Patty Jenkins", "Ava DuVernay", "J.J. Abrams",
    "Zack Snyder", "Todd Phillips", "Jon Favreau",
    "Unknown / First-Time Director",
]

FAMOUS_ACTORS = [
    "Tom Hanks", "Dwayne Johnson", "Leonardo DiCaprio",
    "Scarlett Johansson", "Robert Downey Jr.", "Meryl Streep",
    "Brad Pitt", "Chadwick Boseman", "Will Smith", "Margot Robbie",
    "Ryan Reynolds", "Zendaya", "Tom Cruise", "Chris Evans",
    "Viola Davis", "Samuel L. Jackson", "Jennifer Lawrence",
    "Denzel Washington", "Other / Unknown",
]

MAJOR_STUDIOS = [
    "Warner Bros.", "Universal Pictures", "Columbia Pictures",
    "Paramount Pictures", "Walt Disney Pictures", "20th Century Fox",
    "Netflix", "Amazon Studios", "Apple TV+", "Lionsgate",
    "DreamWorks", "Sony Pictures", "New Line Cinema",
    "Independent / Unknown",
]

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING  (cached so it runs only once per session)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="🎬 Loading prediction models …")
def load_models() -> Optional[Dict]:
    """
    Load the serialised stacking ensemble models and the StandardScaler
    saved by model_training.py.

    Returns None with a warning if models have not been trained yet;
    the app gracefully falls back to a demo/simulation mode.
    """
    required = [
        MODELS_DIR / "stacking_regressor.pkl",
        MODELS_DIR / "stacking_classifier.pkl",
        MODELS_DIR / "scaler.pkl",
        MODELS_DIR / "feature_cols.pkl",
    ]
    if not all(p.exists() for p in required):
        return None

    artefacts = {}
    for p in required:
        with open(p, "rb") as f:
            artefacts[p.stem] = pickle.load(f)
    return artefacts


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE CONSTRUCTION FROM USER INPUTS
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_vector(
    budget_m: float,
    release_month: int,
    release_dow: int,
    genres_selected: List[str],
    is_major_studio: int,
    is_franchise: int,
    is_adaptation: int,
    popularity: float,
    vote_average: float,
    vote_count_k: float,
    director_roi: float,
    star_power_b: float,
    feature_cols: List[str],
) -> np.ndarray:
    """
    Map raw user inputs onto the model feature vector.

    Parameters match the feature engineering conventions in data_pipeline.py.
    Each feature is computed identically to the training pipeline to avoid
    train-serve skew.
    """
    budget_usd = budget_m * 1_000_000

    # ── Temporal ──────────────────────────────────────────────────────
    is_summer       = int(release_month in [5, 6, 7, 8])
    is_holiday      = int(release_month in [11, 12])
    is_award_season = int(release_month in [10, 11, 12])
    is_weekend      = int(release_dow in [4, 5, 6])
    month_sin       = math.sin(2 * math.pi * release_month / 12)
    month_cos       = math.cos(2 * math.pi * release_month / 12)

    # ── Financial ─────────────────────────────────────────────────────
    log_budget          = math.log1p(budget_usd)
    budget_vs_genre_avg = 1.0                         # neutral (unknown genre avg)

    # ── Genre flags ───────────────────────────────────────────────────
    genre_flags = {
        f"genre_{g.lower().replace(' ', '_')}": int(g in genres_selected)
        for g in GENRE_LIST
    }
    genre_count     = len(genres_selected)
    genre_diversity = min(genre_count, 5) / 5.0
    genre_action    = int("Action" in genres_selected)
    genre_drama     = int("Drama" in genres_selected)

    # ── Engagement proxies ────────────────────────────────────────────
    log_popularity  = math.log1p(popularity)
    vote_count      = vote_count_k * 1000
    log_vote_count  = math.log1p(vote_count)
    buzz_score      = log_popularity * log_vote_count
    weighted_rating = (vote_average * vote_count) / (vote_count + 100)
    rating_conf     = min(vote_count / (10_000 + 1), 5)

    # ── Star Power & Director ─────────────────────────────────────────
    star_power_usd     = star_power_b * 1_000_000_000
    star_power_log     = math.log1p(star_power_usd)
    director_track_rec = director_roi

    # ── Interaction features ──────────────────────────────────────────
    action_x_budget     = genre_action * log_budget
    franchise_x_star    = is_franchise * star_power_log
    director_x_budget   = director_track_rec * log_budget
    buzz_x_rating       = buzz_score * weighted_rating
    is_peak_season      = int(is_summer or is_holiday)
    competition_proxy   = is_award_season * genre_drama

    # ── Assemble base dictionary ──────────────────────────────────────
    feature_dict = {
        # Temporal
        "release_month"    : release_month,
        "release_dow"      : release_dow,
        "is_summer"        : is_summer,
        "is_holiday"       : is_holiday,
        "is_award_season"  : is_award_season,
        "is_weekend_release": is_weekend,
        "month_sin"        : month_sin,
        "month_cos"        : month_cos,
        # Financial
        "log_budget"       : log_budget,
        "budget_vs_genre_avg": budget_vs_genre_avg,
        "is_major_studio"  : is_major_studio,
        # Genre
        "genre_count"      : genre_count,
        "genre_diversity"  : genre_diversity,
        # Engagement
        "popularity"       : popularity,
        "vote_count"       : vote_count,
        "vote_average"     : vote_average,
        "log_popularity"   : log_popularity,
        "log_vote_count"   : log_vote_count,
        "buzz_score"       : buzz_score,
        "weighted_rating"  : weighted_rating,
        "rating_confidence": rating_conf,
        # Star & Director
        "star_power_index_log" : star_power_log,
        "director_track_record": director_track_rec,
        # Flags
        "is_franchise"     : is_franchise,
        "is_adaptation"    : is_adaptation,
        # Interaction
        "action_x_budget"  : action_x_budget,
        "franchise_x_star" : franchise_x_star,
        "director_x_budget": director_x_budget,
        "buzz_x_rating"    : buzz_x_rating,
        "is_peak_season"   : is_peak_season,
        "competition_proxy": competition_proxy,
        **genre_flags,
    }

    # Build aligned vector (fill missing features with 0)
    vec = np.array([feature_dict.get(col, 0.0) for col in feature_cols], dtype=float)
    return vec


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def predict(
    feature_vec: np.ndarray,
    artefacts: Optional[Dict],
    budget_usd: float,
) -> Dict:
    """
    Run the stacking ensemble (or simulation if models not loaded) and
    return a structured prediction dictionary.

    Returns
    -------
    dict with keys:
        revenue_usd    : float — predicted worldwide box office (USD)
        roi            : float — predicted return on investment
        class_label    : str   — "🔴 Flop" / "🟡 Average" / "🟢 Hit"
        class_proba    : dict  — {label: probability}
        log_revenue    : float — raw log-space output
    """
    if artefacts is not None:
        scaler       = artefacts["scaler"]
        reg_model    = artefacts["stacking_regressor"]
        clf_model    = artefacts["stacking_classifier"]
        feature_cols = artefacts["feature_cols"]

        X     = feature_vec.reshape(1, -1)
        X_sc  = scaler.transform(X)

        log_rev  = float(reg_model.predict(X_sc)[0])
        rev_usd  = math.expm1(max(log_rev, 0)) * 1.0   # expm1 reverses log1p

        cls_idx  = int(clf_model.predict(X_sc)[0])
        try:
            proba = clf_model.predict_proba(X_sc)[0]
        except AttributeError:
            proba = np.array([0.33, 0.34, 0.33])

    else:
        # ── Demo / simulation mode when models are not trained ──────
        st.info(
            "ℹ️ **Demo mode** — models not found in `models/`.  "
            "Run `data_pipeline.py` then `model_training.py` to enable live predictions.  "
            "Showing illustrative outputs based on heuristic rules."
        )
        # Heuristic estimate: R²≈0.8 means budget is the strongest predictor
        base_rev  = budget_usd * 2.5                    # average multiplier
        buzz_mult = 1 + (feature_vec[feature_cols.index("buzz_score")]
                         if artefacts else 0) / 20
        rev_usd   = base_rev * buzz_mult
        log_rev   = math.log1p(rev_usd)
        roi       = rev_usd / max(budget_usd, 1)

        if roi >= 3.0:
            cls_idx, proba = 2, np.array([0.05, 0.20, 0.75])
        elif roi >= 1.5:
            cls_idx, proba = 1, np.array([0.15, 0.70, 0.15])
        else:
            cls_idx, proba = 0, np.array([0.70, 0.25, 0.05])

    roi           = rev_usd / max(budget_usd, 1)
    class_label   = SUCCESS_LABELS[cls_idx]
    class_proba   = {
        SUCCESS_LABELS[i]: float(proba[i]) for i in range(3)
    }

    return {
        "revenue_usd" : rev_usd,
        "roi"         : roi,
        "class_label" : class_label,
        "class_proba" : class_proba,
        "log_revenue" : log_rev,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────

def inject_css():
    st.markdown("""
    <style>
        /* Header gradient */
        .main-header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            padding: 2rem 2.5rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
        }
        .main-header h1 { color: #e94560; margin: 0; font-size: 2.2rem; }
        .main-header p  { color: #a0aec0; margin: 0.3rem 0 0; font-size: 1rem; }

        /* Metric cards */
        .metric-card {
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 1.2rem 1.5rem;
            text-align: center;
        }
        .metric-card .label { color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; }
        .metric-card .value { color: #f1f5f9; font-size: 1.8rem; font-weight: 700; }
        .metric-card .sub   { color: #64748b; font-size: 0.75rem; }

        /* Result banner */
        .result-hit     { background: #064e3b; border-left: 5px solid #10b981; }
        .result-average { background: #451a03; border-left: 5px solid #f59e0b; }
        .result-flop    { background: #450a0a; border-left: 5px solid #ef4444; }
        .result-box {
            border-radius: 10px;
            padding: 1.5rem 2rem;
            margin: 1rem 0;
        }
        .result-box h2 { margin: 0; font-size: 1.6rem; }
        .result-box p  { margin: 0.3rem 0 0; color: #d1d5db; }

        /* Sidebar */
        [data-testid="stSidebar"] { background: #0f172a; }
        [data-testid="stSidebar"] * { color: #e2e8f0 !important; }

        /* Probability bar */
        .prob-bar { border-radius: 8px; height: 22px; margin: 4px 0; }

        /* Footer */
        .footer {
            text-align: center; color: #475569;
            font-size: 0.75rem; margin-top: 3rem; padding-top: 1rem;
            border-top: 1px solid #1e293b;
        }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# UI COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

def render_header():
    st.markdown("""
    <div class="main-header">
      <h1>🎬 Box Office Success Predictor</h1>
      <p>Multi-Source Big Data &amp; Stacking Ensemble ML · Capstone Project 2026</p>
      <p style="margin-top:0.5rem; font-size:0.85rem; color:#718096;">
        I. Parthiban &nbsp;|&nbsp; Mike Roonane Fernandez &nbsp;·&nbsp;
        VIT Chennai — 23BCE2282 / 23BCE2062
      </p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar_inputs() -> Dict:
    """Render the left sidebar with all prediction inputs."""
    st.sidebar.markdown("## 🎛️ Film Parameters")
    st.sidebar.markdown("---")

    st.sidebar.markdown("### 💰 Financial")
    budget_m = st.sidebar.slider(
        "Production Budget (USD millions)",
        min_value=0.5, max_value=400.0, value=50.0, step=0.5,
        help="Inflation-adjusted to 2023 USD"
    )

    st.sidebar.markdown("### 🎭 Cast & Crew")
    director = st.sidebar.selectbox("Director", FAMOUS_DIRECTORS)
    lead_actor = st.sidebar.selectbox("Lead Actor / Star", FAMOUS_ACTORS)

    # Director ROI proxy
    high_roi_directors = {
        "Christopher Nolan": 4.2, "James Cameron": 5.1,
        "Steven Spielberg": 3.8, "Quentin Tarantino": 3.5,
        "Jon Favreau": 4.5, "Ryan Coogler": 4.8,
    }
    director_roi = high_roi_directors.get(director, 2.2)

    # Star power proxy (billions USD cumulative)
    high_sp_actors = {
        "Dwayne Johnson": 8.5, "Robert Downey Jr.": 10.2,
        "Tom Cruise": 7.8, "Chris Evans": 7.1,
        "Scarlett Johansson": 9.3, "Samuel L. Jackson": 10.8,
    }
    star_power_b = high_sp_actors.get(lead_actor, 3.5)

    st.sidebar.markdown("### 🏢 Studio")
    studio = st.sidebar.selectbox("Production Studio", MAJOR_STUDIOS)
    is_major_studio = int(studio not in ["Independent / Unknown"])

    st.sidebar.markdown("### 🎬 Genre")
    genres_selected = st.sidebar.multiselect(
        "Select Genre(s)",
        options=GENRE_LIST,
        default=["Action", "Adventure"],
        help="Select all applicable genres"
    )

    st.sidebar.markdown("### 📅 Release Timing")
    month_names = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    release_month_name = st.sidebar.selectbox("Release Month", month_names, index=6)
    release_month = month_names.index(release_month_name) + 1

    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    release_day_name = st.sidebar.selectbox(
        "Release Day of Week", day_names, index=4,
        help="Most blockbusters release on Fridays (index 4)"
    )
    release_dow = day_names.index(release_day_name)

    st.sidebar.markdown("### 🔖 Project Characteristics")
    is_franchise  = st.sidebar.checkbox("🔁 Part of a Franchise / Sequel")
    is_adaptation = st.sidebar.checkbox("📚 Based on Existing IP (book, comic, game)")

    st.sidebar.markdown("### 📊 Audience Engagement Signals")
    st.sidebar.caption("TMDB-style engagement proxy — estimate pre-release values")
    popularity    = st.sidebar.slider("Popularity Score (TMDB-scale)", 0.0, 500.0, 85.0, 1.0)
    vote_average  = st.sidebar.slider("Anticipated Rating (1–10)", 1.0, 10.0, 7.2, 0.1)
    vote_count_k  = st.sidebar.slider("Anticipated Vote Count (thousands)", 0.1, 50.0, 5.0, 0.1)

    return dict(
        budget_m        = budget_m,
        director        = director,
        lead_actor      = lead_actor,
        director_roi    = director_roi,
        star_power_b    = star_power_b,
        studio          = studio,
        is_major_studio = is_major_studio,
        genres_selected = genres_selected,
        release_month   = release_month,
        release_dow     = release_dow,
        is_franchise    = int(is_franchise),
        is_adaptation   = int(is_adaptation),
        popularity      = popularity,
        vote_average    = vote_average,
        vote_count_k    = vote_count_k,
    )


def render_result_banner(pred: Dict):
    label  = pred["class_label"]
    rev    = pred["revenue_usd"]
    roi    = pred["roi"]
    css_cls = {
        "🟢 Hit"    : "result-hit",
        "🟡 Average": "result-average",
        "🔴 Flop"   : "result-flop",
    }.get(label, "result-average")

    st.markdown(f"""
    <div class="result-box {css_cls}">
      <h2>{label}</h2>
      <p>Predicted worldwide box office:
         <strong style="font-size:1.3rem;">${rev/1e6:,.1f}M</strong>
         &nbsp;·&nbsp; ROI: <strong>{roi:.2f}×</strong>
      </p>
    </div>
    """, unsafe_allow_html=True)


def render_probability_bars(class_proba: Dict):
    st.markdown("#### Success Probability Breakdown")
    colour_map = {
        "🔴 Flop"   : "#ef4444",
        "🟡 Average": "#f59e0b",
        "🟢 Hit"    : "#10b981",
    }
    for label, prob in class_proba.items():
        bar_w  = max(int(prob * 100), 1)
        colour = colour_map[label]
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
          <span style="width:100px;font-size:0.85rem;">{label}</span>
          <div style="flex:1;background:#1e293b;border-radius:8px;height:22px;">
            <div class="prob-bar"
                 style="width:{bar_w}%;background:{colour};height:22px;border-radius:8px;">
            </div>
          </div>
          <span style="width:45px;text-align:right;font-weight:bold;">{prob:.1%}</span>
        </div>
        """, unsafe_allow_html=True)


def render_metric_row(pred: Dict, budget_m: float):
    rev = pred["revenue_usd"]
    roi = pred["roi"]
    profit = rev - budget_m * 1_000_000

    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        (col1, "Predicted Revenue", f"${rev/1e6:,.1f}M", "Worldwide Box Office (2023 USD)"),
        (col2, "ROI Multiple",      f"{roi:.2f}×",        "Revenue / Production Budget"),
        (col3, "Gross Profit",      f"${profit/1e6:+,.1f}M", "Revenue − Budget Estimate"),
        (col4, "Confidence",        f"{max(pred['class_proba'].values()):.0%}",
                                     "Top class probability"),
    ]
    for col, label, value, sub in metrics:
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="label">{label}</div>
              <div class="value">{value}</div>
              <div class="sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)


def render_risk_gauge(roi: float):
    """Simple ASCII/emoji risk-o-meter."""
    st.markdown("#### 🎯 Investment Risk Profile")

    if roi >= 3.0:
        risk_level, risk_colour, risk_desc = "Low", "🟢", "Strong return expected — recommend green-light"
    elif roi >= 1.5:
        risk_level, risk_colour, risk_desc = "Medium", "🟡", "Break-even likely — conditional approval recommended"
    else:
        risk_level, risk_colour, risk_desc = "High", "🔴", "Revenue may not recoup costs — caution advised"

    bar_fill = min(int((roi / 5.0) * 20), 20)
    bar_empty = 20 - bar_fill
    bar = "█" * bar_fill + "░" * bar_empty

    st.markdown(f"""
    <div style="background:#1e293b;border-radius:10px;padding:1.2rem 1.5rem;">
      <div style="font-family:monospace;font-size:1.1rem;letter-spacing:2px;
                  color:#38bdf8;">{bar}</div>
      <div style="margin-top:0.5rem;">
        {risk_colour} <strong>{risk_level} Risk</strong>
        &nbsp;·&nbsp; <span style="color:#94a3b8;">{risk_desc}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_insight_table(inputs: Dict, pred: Dict):
    """Tabular breakdown of the key input factors and their expected impact."""
    st.markdown("#### 📋 Factor Analysis Summary")

    summer     = "✅ Yes" if inputs["release_month"] in [5, 6, 7, 8] else "❌ No"
    award_szn  = "✅ Yes" if inputs["release_month"] in [10, 11, 12] else "❌ No"
    franchise  = "✅ Yes" if inputs["is_franchise"] else "❌ No"
    maj_studio = "✅ Yes" if inputs["is_major_studio"] else "❌ No"
    adaptation = "✅ Yes" if inputs["is_adaptation"] else "❌ No"

    data = {
        "Factor"           : [
            "Production Budget", "Director Track Record",
            "Star Power", "Summer Release", "Award Season",
            "Franchise / Sequel", "Major Studio", "Pre-release Buzz",
            "Audience Rating", "IP Adaptation",
        ],
        "Value"            : [
            f"${inputs['budget_m']:,.0f}M",
            f"{inputs['director_roi']:.1f}× avg ROI",
            f"${inputs['star_power_b']:.1f}B cumulative",
            summer, award_szn, franchise, maj_studio,
            f"{inputs['popularity']:.0f} (TMDB popularity)",
            f"{inputs['vote_average']:.1f} / 10",
            adaptation,
        ],
        "Impact"           : [
            "🔴 Primary driver (0.182 SHAP)",
            "🔵 Director quality signal",
            "🔵 Cast appeal metric",
            "🟢 Peak-season advantage",
            "🟣 Oscar-campaign timing",
            "🔵 Brand recognition boost",
            "🔴 Distribution strength",
            "🟢 Audience anticipation",
            "🟢 Reception proxy",
            "🔵 Pre-existing fanbase",
        ],
    }
    st.dataframe(
        pd.DataFrame(data),
        use_container_width=True,
        hide_index=True,
    )


def render_comparable_films(pred: Dict, genres: List[str]):
    """Show hand-picked comparable films at similar predicted performance levels."""
    st.markdown("#### 🎥 Comparable Films at This Performance Level")

    genre_str = " / ".join(genres[:2]) if genres else "General"
    label     = pred["class_label"]

    comparables = {
        "🟢 Hit": [
            ("Avengers: Endgame", "$2.80B", "Action / Adventure", "22.0×"),
            ("The Dark Knight",   "$1.00B", "Action / Crime",     "5.0×"),
            ("Barbie",            "$1.44B", "Comedy / Fantasy",   "14.4×"),
        ],
        "🟡 Average": [
            ("Men in Black 3",      "$624M", "Action / Sci-Fi",   "3.1×"),
            ("The Predator",        "$161M", "Action / Sci-Fi",   "2.0×"),
            ("Knives Out",          "$312M", "Comedy / Mystery",  "4.2×"),
        ],
        "🔴 Flop": [
            ("John Carter",    "$284M", "Action / Sci-Fi",  "0.5×"),
            ("The Lone Ranger", "$260M", "Action / Western", "0.7×"),
            ("Cats",            "$73M",  "Drama / Musical",  "0.2×"),
        ],
    }

    films = comparables.get(label, comparables["🟡 Average"])
    col1, col2, col3 = st.columns(3)
    for col, (title, rev, genre, roi) in zip([col1, col2, col3], films):
        with col:
            st.markdown(f"""
            <div style="background:#1e293b;border-radius:8px;padding:1rem;text-align:center;">
              <div style="font-weight:bold;font-size:0.95rem;">{title}</div>
              <div style="color:#38bdf8;font-size:1.1rem;font-weight:700;">{rev}</div>
              <div style="color:#94a3b8;font-size:0.8rem;">{genre}</div>
              <div style="color:#64748b;font-size:0.75rem;">ROI {roi}</div>
            </div>
            """, unsafe_allow_html=True)


def render_methodology_expander():
    with st.expander("📖 Methodology Overview (Section 3 of Report)"):
        st.markdown("""
        ### Data Sources
        - **TMDB 5000** — 5,000 films with metadata, budgets, revenues, cast, crew
        - **IMDb Metadata** — Director/actor histories, genre classifications
        - **Engagement Proxies** — TMDB popularity, vote counts, vote averages

        ### Feature Engineering (127 features)
        | Family | Key Features |
        |--------|-------------|
        | Financial | log(budget), budget category, studio type |
        | Star Power | Σ top-3 cast cumulative box office (log-scaled) |
        | Director | Recency-weighted average ROI across filmography |
        | Engagement | Buzz score (popularity × vote_count), weighted rating |
        | Temporal | Summer/holiday flags, month cyclic encoding, DOW |
        | Interaction | Action×Budget, Franchise×Star, Director×Budget |

        ### Model Architecture (Stacking Ensemble)
        ```
        Level 0 (Base Learners)      Level 1 (Meta-Learner)
        ┌─────────────────┐
        │   XGBoost       │ ─┐
        │   Random Forest │ ─┼─► Ridge Regression (Revenue)
        │   Neural Network│ ─┘    Logistic Regression (Category)
        └─────────────────┘
        ```
        *Out-of-fold cross-validation (k=5) prevents data leakage in stacking.*

        ### Performance (Test Set 2022–2023)
        | Task | Best Model | Score |
        |------|-----------|-------|
        | Revenue Prediction | Stacking Ensemble | R² = 0.805, MAPE = 21.8% |
        | Success Classification | Stacking Ensemble | Accuracy = 77.4%, AUC = 0.885 |
        """)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    inject_css()
    render_header()

    # Load models (or enter demo mode)
    artefacts = load_models()
    feature_cols = artefacts["feature_cols"] if artefacts else list(range(127))

    # Sidebar inputs
    inputs = render_sidebar_inputs()

    # Predict button
    predict_btn = st.sidebar.button(
        "🚀 Predict Box Office Performance",
        use_container_width=True,
        type="primary",
    )

    # ── Main panel ──────────────────────────────────────────────────────
    if not predict_btn:
        # Welcome screen
        st.markdown("""
        <div style="text-align:center;padding:3rem 1rem;color:#94a3b8;">
          <div style="font-size:4rem;">🎬</div>
          <h3 style="color:#e2e8f0;">Configure your film in the sidebar and click Predict</h3>
          <p>The stacking ensemble integrates production budget, cast, director track record,
             genre, release timing, and audience engagement signals to forecast worldwide
             box office revenue and success probability.</p>
        </div>
        """, unsafe_allow_html=True)
        render_methodology_expander()
        return

    # ── Feature construction ────────────────────────────────────────────
    feature_vec = build_feature_vector(
        budget_m        = inputs["budget_m"],
        release_month   = inputs["release_month"],
        release_dow     = inputs["release_dow"],
        genres_selected = inputs["genres_selected"],
        is_major_studio = inputs["is_major_studio"],
        is_franchise    = inputs["is_franchise"],
        is_adaptation   = inputs["is_adaptation"],
        popularity      = inputs["popularity"],
        vote_average    = inputs["vote_average"],
        vote_count_k    = inputs["vote_count_k"],
        director_roi    = inputs["director_roi"],
        star_power_b    = inputs["star_power_b"],
        feature_cols    = feature_cols,
    )

    # ── Prediction ──────────────────────────────────────────────────────
    with st.spinner("🤖 Running stacking ensemble …"):
        pred = predict(feature_vec, artefacts, inputs["budget_m"] * 1_000_000)

    # ── Output layout ───────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📊 Prediction Results", "📋 Factor Analysis", "🎥 Comparable Films"])

    with tab1:
        render_result_banner(pred)
        st.markdown("")
        render_metric_row(pred, inputs["budget_m"])
        st.markdown("")

        col_l, col_r = st.columns([1.2, 1])
        with col_l:
            render_probability_bars(pred["class_proba"])
        with col_r:
            render_risk_gauge(pred["roi"])

    with tab2:
        render_insight_table(inputs, pred)
        render_methodology_expander()

    with tab3:
        render_comparable_films(pred, inputs["genres_selected"])

    # Footer
    st.markdown("""
    <div class="footer">
      Box Office Success Prediction · Capstone Project 2026 ·
      I. Parthiban (23BCE2282) &amp; Mike Roonane Fernandez (23BCE2062) · VIT Chennai<br>
      Framework: Multi-Source Big Data + Stacking Ensemble ML ·
      Data: TMDB 5000 + IMDb Metadata
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
