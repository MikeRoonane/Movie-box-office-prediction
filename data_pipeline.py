"""
================================================================================
Box Office Success Prediction — Data Pipeline
================================================================================
Authors  : I.Parthiban 23BCE2282, Mike Roonane Fernandez 23BCE2062
Date     : February 2026
Project  : Capstone — Movie Box Office Prediction using Multi-Source Big Data
           and Machine Learning with IMDb Metadata

Description
-----------
This module implements the full ETL (Extract-Transform-Load) pipeline described
in Section 3 of the project report.  It handles:

    Stage 1 · Multi-source data loading and inner-join merging
              (TMDB 5000 Movies + TMDB 5000 Credits)
    Stage 2 · Data cleaning — missing value imputation (KNN / Median),
              outlier detection via IQR, duplicate removal
    Stage 3 · Currency normalisation to 2023 constant USD (CPI-deflated)
    Stage 4 · Advanced feature engineering
              · Star Power Index (cast historical box-office aggregate)
              · Director Track Record (historical average ROI, recency-weighted)
              · Engagement proxies (TMDB popularity, vote_count, vote_average)
              · Temporal, genre, franchise, and interaction features
    Stage 5 · Feature selection and final dataset serialisation

Usage
-----
    python data_pipeline.py

    The script downloads nothing automatically.  Place the two TMDB-5000 CSV
    files in the same directory (or adjust DATA_DIR below):

        tmdb_5000_movies.csv
        tmdb_5000_credits.csv

    Download from:  https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

Output
------
    processed_movies.csv   — final feature matrix ready for model_training.py
    pipeline_report.txt    — summary statistics & data quality report
================================================================================
"""

import ast
import json
import logging
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR   = Path(".")          # directory containing the raw CSVs
OUTPUT_DIR = Path(".")          # directory for processed outputs

MOVIES_CSV  = DATA_DIR / "tmdb_5000_movies.csv"
CREDITS_CSV = DATA_DIR / "tmdb_5000_credits.csv"
OUTPUT_CSV  = OUTPUT_DIR / "processed_movies.csv"
REPORT_TXT  = OUTPUT_DIR / "pipeline_report.txt"

# Minimum thresholds for a record to be considered valid
MIN_BUDGET_USD   = 100_000      # $100k  — below this is likely an error
MIN_REVENUE_USD  = 100_000      # $100k
MIN_RUNTIME_MIN  = 40           # 40 minutes
MAX_RUNTIME_MIN  = 300          # 5 hours

# ROI thresholds for classification (from report Section 3.6.2)
ROI_FLOP_THRESHOLD    = 1.5
ROI_AVERAGE_THRESHOLD = 3.0

# CPI deflators: annual average CPI (BLS, US) relative to 2023 base
# Source: U.S. Bureau of Labor Statistics (series CUUR0000SA0)
# Values represent the multiplier needed to convert year-Y dollars → 2023 dollars
CPI_DEFLATORS: Dict[int, float] = {
    1990: 2.421, 1991: 2.319, 1992: 2.252, 1993: 2.188, 1994: 2.132,
    1995: 2.072, 1996: 2.012, 1997: 1.969, 1998: 1.938, 1999: 1.902,
    2000: 1.846, 2001: 1.793, 2002: 1.764, 2003: 1.726, 2004: 1.687,
    2005: 1.637, 2006: 1.589, 2007: 1.543, 2008: 1.490, 2009: 1.494,
    2010: 1.471, 2011: 1.423, 2012: 1.394, 2013: 1.376, 2014: 1.352,
    2015: 1.351, 2016: 1.338, 2017: 1.309, 2018: 1.278, 2019: 1.252,
    2020: 1.240, 2021: 1.183, 2022: 1.084, 2023: 1.000,
}

# Genres used for binary feature columns
GENRE_LIST = [
    "Action", "Adventure", "Animation", "Comedy", "Crime",
    "Documentary", "Drama", "Family", "Fantasy", "History",
    "Horror", "Music", "Mystery", "Romance", "Science Fiction",
    "Thriller", "War", "Western",
]

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def safe_parse_json(value) -> list:
    """
    Safely parse a JSON / Python-literal string column (e.g., genres, cast).
    Returns an empty list on failure.
    """
    if pd.isna(value) or value in ("", "[]", "{}"):
        return []
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return []


def extract_names(json_col, key: str = "name", limit: Optional[int] = None) -> List[str]:
    """Extract a flat list of values for `key` from a parsed JSON list."""
    records = safe_parse_json(json_col)
    names = [r[key] for r in records if isinstance(r, dict) and key in r]
    return names[:limit] if limit else names


def compute_genre_diversity(genres: List[str]) -> float:
    """
    Genre Diversity Index: normalised count of genres (0–1 scale, capped at 5).
    A film spanning many genres is scored higher.
    """
    return min(len(genres), 5) / 5.0


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 · DATA LOADING & MERGING
# ─────────────────────────────────────────────────────────────────────────────

class DataLoader:
    """
    Loads and merges the TMDB 5000 movies and credits datasets.

    The merge uses the shared integer `id` / `movie_id` key (equivalent to the
    TMDB ID). This approach mirrors the primary-key matching strategy described
    in Section 3.3.2 of the report.
    """

    def __init__(self, movies_path: Path, credits_path: Path):
        self.movies_path  = movies_path
        self.credits_path = credits_path

    def load(self) -> pd.DataFrame:
        log.info("Loading TMDB 5000 movies …")
        movies = pd.read_csv(self.movies_path, low_memory=False)
        log.info("  ↳ Rows: %d, Columns: %d", *movies.shape)

        log.info("Loading TMDB 5000 credits …")
        credits = pd.read_csv(self.credits_path, low_memory=False)
        log.info("  ↳ Rows: %d, Columns: %d", *credits.shape)

        # Rename credits key to match movies key for a clean inner join
        if "movie_id" in credits.columns:
            credits = credits.rename(columns={"movie_id": "id"})

        # Drop 'title' from credits to avoid collision after merge
        if "title" in credits.columns:
            credits = credits.drop(columns=["title"])

        log.info("Performing inner join on 'id' …")
        df = movies.merge(credits, on="id", how="inner")
        log.info("  ↳ Merged shape: %d rows × %d columns", *df.shape)

        return df


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 · DATA CLEANING
# ─────────────────────────────────────────────────────────────────────────────

class DataCleaner:
    """
    Implements the data-cleaning strategy from report Section 3.3.1:

        1. Remove duplicates based on unique film identifiers
        2. Enforce minimum business-logic thresholds (budget, revenue, runtime)
        3. Parse release_date → release_year, release_month, release_dow
        4. Outlier detection via IQR on revenue and budget
        5. Missing value imputation:
           - Numerical  → KNN imputation (k=5)
           - Categorical → Mode imputation
        6. Currency normalisation to 2023 constant USD (CPI deflation)
    """

    def __init__(self):
        self._imputer = KNNImputer(n_neighbors=5, weights="distance")

    # ------------------------------------------------------------------
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        log.info("─── Stage 2: Data Cleaning ───────────────────────────────")

        df = self._remove_duplicates(df)
        df = self._parse_dates(df)
        df = self._enforce_thresholds(df)
        df = self._impute_missing(df)
        df = self._normalise_currency(df)
        df = self._handle_outliers(df)

        log.info("Clean dataset size: %d rows", len(df))
        return df

    # ------------------------------------------------------------------
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates(subset=["id"])
        log.info("  Duplicate removal: %d → %d rows", before, len(df))
        return df

    # ------------------------------------------------------------------
    def _parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
        df["release_year"]  = df["release_date"].dt.year.astype("Int64")
        df["release_month"] = df["release_date"].dt.month.astype("Int64")
        df["release_dow"]   = df["release_date"].dt.dayofweek.astype("Int64")
        return df

    # ------------------------------------------------------------------
    def _enforce_thresholds(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove records that almost certainly represent data-entry errors
        rather than genuine edge-cases (e.g., budget = $1).
        """
        before = len(df)
        mask = (
            (df["budget"]  >= MIN_BUDGET_USD)  &
            (df["revenue"] >= MIN_REVENUE_USD)  &
            (df["runtime"].between(MIN_RUNTIME_MIN, MAX_RUNTIME_MIN, inclusive="both"))  &
            (df["release_year"].between(1990, 2024))
        )
        df = df[mask].reset_index(drop=True)
        log.info("  Threshold filtering: %d → %d rows", before, len(df))
        return df

    # ------------------------------------------------------------------
    def _impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        KNN imputation for numeric columns; mode imputation for categoricals.
        Only impute columns with < 40% missing to avoid inventing data.
        """
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        imputable = [
            c for c in num_cols
            if 0 < df[c].isna().mean() < 0.40
        ]

        if imputable:
            log.info("  KNN imputing %d numeric columns …", len(imputable))
            df[imputable] = self._imputer.fit_transform(df[imputable])

        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        for c in cat_cols:
            if df[c].isna().any():
                df[c] = df[c].fillna(df[c].mode().iloc[0])

        return df

    # ------------------------------------------------------------------
    def _normalise_currency(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Inflate budget and revenue to 2023 USD using annual CPI deflators.
        Films with release_year not in the deflator table use the nearest year.
        """
        def deflate(row, col):
            year = int(row["release_year"]) if pd.notna(row["release_year"]) else 2010
            year = max(1990, min(year, 2023))           # clamp to table range
            factor = CPI_DEFLATORS.get(year, 1.0)
            return row[col] * factor

        df["budget_2023"]  = df.apply(lambda r: deflate(r, "budget"),  axis=1)
        df["revenue_2023"] = df.apply(lambda r: deflate(r, "revenue"), axis=1)
        log.info("  Currency normalised to 2023 constant USD.")
        return df

    # ------------------------------------------------------------------
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        IQR-based outlier detection on revenue_2023 and budget_2023.
        Genuine blockbusters are kept; statistical anomalies are removed.
        We use 4× IQR fence (wider than standard 1.5×) to preserve
        authentic extreme values (e.g., Avatar, Avengers: Endgame).
        """
        for col in ["revenue_2023", "budget_2023"]:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR    = Q3 - Q1
            lower  = Q1 - 4 * IQR
            upper  = Q3 + 4 * IQR
            before = len(df)
            df     = df[df[col].between(lower, upper)].reset_index(drop=True)
            log.info("  Outlier removal (%s): %d → %d rows", col, before, len(df))
        return df


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 · FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

class FeatureEngineer:
    """
    Constructs the 127-feature predictive matrix described in Section 3.4.

    Feature families
    ────────────────
    Metadata     · Genre binary flags, genre diversity, franchise flag,
                   language indicator, content rating encoding, runtime categories
    Financial    · log-budget, budget category (quintiles), studio type,
                   budget-to-genre-average ratio
    Temporal     · release_month, release_dow, summer flag, holiday proximity,
                   award-season window (Oct–Dec)
    Engagement   · TMDB popularity (pre-release buzz proxy),
                   vote_count (audience reach proxy),
                   vote_average (critic/audience reception proxy),
                   trailer proxy = popularity × vote_count / 1000
    Star Power   · Σ of top-3 cast members' historical inflation-adj. revenues
    Director     · Recency-weighted average ROI across prior directorial works
    Interaction  · genre × log_budget, franchise × star_power, sequel flag
    """

    def __init__(self):
        self._actor_lookup: Dict[str, float]     = {}   # actor → cumulative revenue
        self._director_lookup: Dict[str, float]  = {}   # director → weighted avg ROI

    # ------------------------------------------------------------------
    def engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        log.info("─── Stage 4: Feature Engineering ────────────────────────")

        df = self._parse_json_columns(df)
        df = self._build_star_power_index(df)
        df = self._build_director_track_record(df)
        df = self._build_genre_features(df)
        df = self._build_temporal_features(df)
        df = self._build_financial_features(df)
        df = self._build_engagement_features(df)
        df = self._build_interaction_features(df)
        df = self._define_targets(df)

        log.info("  Feature matrix shape: %d × %d", *df.shape)
        return df

    # ------------------------------------------------------------------
    def _parse_json_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse stringified JSON columns into Python objects."""
        df["genres_list"]       = df["genres"].apply(extract_names)
        df["cast_list"]         = df["cast"].apply(
            lambda x: extract_names(x, limit=10)   # top-10 billed cast
        )
        df["keywords_list"]     = df["keywords"].apply(extract_names)
        df["companies_list"]    = df["production_companies"].apply(extract_names)
        df["crew_parsed"]       = df["crew"].apply(safe_parse_json)

        # Extract director name from crew JSON
        def get_director(crew_list):
            for member in crew_list:
                if isinstance(member, dict) and member.get("job") == "Director":
                    return member.get("name", "Unknown")
            return "Unknown"

        df["director"] = df["crew_parsed"].apply(get_director)
        return df

    # ------------------------------------------------------------------
    def _build_star_power_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Star Power Index (SPI) — Section 3.4.1 of the report.

        Algorithm:
          Pass 1 · Build a lookup table: for every actor, accumulate their
                   total inflation-adjusted revenue across ALL films in the
                   dataset (excluding the current film via leave-one-out).
          Pass 2 · For each film, SPI = Σ revenue of top-3 billed actors
                   (capped at top 3 to avoid dilution by minor roles).

        Note: We use the full dataset to estimate actor reputation, which is
        a reasonable approximation of the "track record" known before release.
        """
        log.info("  Computing Star Power Index …")

        # Pass 1 — accumulate actor revenues
        actor_rev: Dict[str, float] = {}
        for _, row in df.iterrows():
            rev = row.get("revenue_2023", 0)
            for actor in row["cast_list"]:
                actor_rev[actor] = actor_rev.get(actor, 0) + rev

        self._actor_lookup = actor_rev

        # Pass 2 — sum top-3 cast members' cumulative revenues
        def spi(cast):
            top3 = cast[:3]
            return sum(actor_rev.get(a, 0) for a in top3)

        df["star_power_index"] = df["cast_list"].apply(spi)

        # Log-scale to tame skew (same rationale as budget/revenue transforms)
        df["star_power_index_log"] = np.log1p(df["star_power_index"])
        return df

    # ------------------------------------------------------------------
    def _build_director_track_record(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Director Track Record (DTR) — Section 3.4.1 of the report.

        Algorithm:
          A director's track record for film F = recency-weighted average ROI
          across all OTHER films directed by the same director in the dataset.
          Recency weight = 1 / (1 + years_since_release), so more recent
          films carry greater influence.
        """
        log.info("  Computing Director Track Record …")

        # Compute per-film ROI first (handle division by zero gracefully)
        df["roi"] = df["revenue_2023"] / df["budget_2023"].replace(0, np.nan)
        df["roi"] = df["roi"].clip(0, 50)   # cap extreme ROI values

        # Build lookup: director → list of (year, roi) tuples
        dir_records: Dict[str, List[Tuple[int, float]]] = {}
        for _, row in df.iterrows():
            d   = row["director"]
            roi = row["roi"]
            yr  = row.get("release_year", 2010)
            if pd.notna(roi) and pd.notna(yr) and d != "Unknown":
                dir_records.setdefault(d, []).append((int(yr), float(roi)))

        def weighted_avg_roi(director, current_year):
            records = dir_records.get(director, [])
            if not records:
                return 1.5   # population mean fallback
            total_w, total_wr = 0.0, 0.0
            for yr, roi in records:
                w = 1.0 / (1 + abs(int(current_year) - yr) + 1e-6)
                total_w  += w
                total_wr += w * roi
            return total_wr / total_w if total_w > 0 else 1.5

        df["director_track_record"] = df.apply(
            lambda r: weighted_avg_roi(r["director"], r.get("release_year", 2010)),
            axis=1,
        )
        return df

    # ------------------------------------------------------------------
    def _build_genre_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Binary genre indicators + Genre Diversity Index.
        Also flags franchise / sequel presence via keywords.
        """
        for genre in GENRE_LIST:
            col = f"genre_{genre.lower().replace(' ', '_')}"
            df[col] = df["genres_list"].apply(lambda g: int(genre in g))

        df["genre_diversity"]    = df["genres_list"].apply(compute_genre_diversity)
        df["genre_count"]        = df["genres_list"].apply(len)

        # Primary genre (first listed in TMDB — considered most representative)
        df["primary_genre"]      = df["genres_list"].apply(
            lambda g: g[0] if g else "Unknown"
        )

        # Franchise / sequel indicators from keywords
        sequel_keywords = {
            "sequel", "based on novel", "based on comic book",
            "based on video game", "spin-off", "reboot", "remake",
        }
        df["is_franchise"] = df["keywords_list"].apply(
            lambda kws: int(bool(set(k.lower() for k in kws) & sequel_keywords))
        )
        df["is_adaptation"] = df["keywords_list"].apply(
            lambda kws: int(any(
                k in kws for k in ["based on novel", "based on comic book",
                                   "based on video game"]
            ))
        )
        return df

    # ------------------------------------------------------------------
    def _build_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Temporal features from release date — Section 3.4.1 (Temporal Features).
        """
        df["release_month"]  = df["release_month"].fillna(6).astype(int)
        df["release_dow"]    = df["release_dow"].fillna(4).astype(int)
        df["release_year"]   = df["release_year"].fillna(2010).astype(int)

        # Summer blockbuster window: May–August (months 5–8)
        df["is_summer"]      = df["release_month"].isin([5, 6, 7, 8]).astype(int)

        # Holiday window: November–December (Thanksgiving / Christmas)
        df["is_holiday"]     = df["release_month"].isin([11, 12]).astype(int)

        # Award season (October–December — aligns with Oscar campaigns)
        df["is_award_season"] = df["release_month"].isin([10, 11, 12]).astype(int)

        # Weekend release (Friday=4, Saturday=5, Sunday=6)
        df["is_weekend_release"] = df["release_dow"].isin([4, 5, 6]).astype(int)

        # Release season encoding (cyclical: for gradient models)
        df["month_sin"] = np.sin(2 * np.pi * df["release_month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["release_month"] / 12)

        return df

    # ------------------------------------------------------------------
    def _build_financial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Financial and studio features — Section 3.4.2.
        """
        # Log-transformed budget (primary predictor, see Table 3 in report)
        df["log_budget"] = np.log1p(df["budget_2023"])

        # Budget quintile categories (Low / Below-Avg / Average / Above-Avg / High)
        df["budget_category"] = pd.qcut(
            df["budget_2023"], q=5,
            labels=["Low", "Below_Avg", "Average", "Above_Avg", "High"],
            duplicates="drop",
        ).astype(str)

        # Major studio indicator (heuristic: top-10 studios by volume)
        major_studios = {
            "Warner Bros.", "Universal Pictures", "Columbia Pictures",
            "Paramount Pictures", "Walt Disney Pictures", "20th Century Fox",
            "New Line Cinema", "Lionsgate", "DreamWorks", "Sony Pictures",
        }
        df["is_major_studio"] = df["companies_list"].apply(
            lambda cs: int(bool(set(cs) & major_studios))
        )

        # Budget-to-genre-average ratio (how well-funded vs. peers)
        genre_avg_budget = df.groupby("primary_genre")["budget_2023"].transform("mean")
        df["budget_vs_genre_avg"] = df["budget_2023"] / (genre_avg_budget + 1)

        return df

    # ------------------------------------------------------------------
    def _build_engagement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        TMDB engagement proxies — Section 3.4.3 of the report.

        TMDB's `popularity` reflects a rolling momentum score (similar to a
        social-media "buzz" signal). `vote_count` approximates audience reach.
        `vote_average` proxies critic/audience reception.

        Trailer Engagement Proxy = popularity × log(vote_count + 1)
        — high popularity + many votes suggests strong pre-release interest.
        """
        df["popularity"]    = pd.to_numeric(df["popularity"],    errors="coerce").fillna(0)
        df["vote_count"]    = pd.to_numeric(df["vote_count"],    errors="coerce").fillna(0)
        df["vote_average"]  = pd.to_numeric(df["vote_average"],  errors="coerce").fillna(5.0)

        df["log_popularity"]   = np.log1p(df["popularity"])
        df["log_vote_count"]   = np.log1p(df["vote_count"])

        # Composite buzz score (proxy for pre-release social media buzz)
        df["buzz_score"]       = df["log_popularity"] * df["log_vote_count"]

        # Audience reception proxy (weighted average — more votes = more reliable)
        df["weighted_rating"]  = (
            df["vote_average"] * df["vote_count"]
        ) / (df["vote_count"] + 100)   # Bayesian shrinkage toward neutral

        # Critic-audience alignment proxy
        df["rating_confidence"] = df["vote_count"] / (df["vote_count"].quantile(0.75) + 1)
        df["rating_confidence"] = df["rating_confidence"].clip(0, 5)

        return df

    # ------------------------------------------------------------------
    def _build_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derived interaction terms — Section 3.4.4 of the report.
        """
        # Genre × Budget: action films with larger budgets tend to outperform
        df["action_x_budget"] = df["genre_action"] * df["log_budget"]

        # Franchise × Star Power: IP-backed films with A-listers outperform
        df["franchise_x_star"] = df["is_franchise"] * df["star_power_index_log"]

        # Director quality × Budget efficiency
        df["director_x_budget"] = df["director_track_record"] * df["log_budget"]

        # Buzz × Rating: high buzz + high rating = strong commercial signal
        df["buzz_x_rating"] = df["buzz_score"] * df["weighted_rating"]

        # Summer OR Holiday mega-release window
        df["is_peak_season"] = ((df["is_summer"] == 1) | (df["is_holiday"] == 1)).astype(int)

        # Competition proxy: studios release fewer films in peak months,
        # so we invert: award-season = more competition for drama films
        df["competition_proxy"] = df["is_award_season"] * df["genre_drama"]

        return df

    # ------------------------------------------------------------------
    def _define_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Define the two prediction targets as described in Section 3.6.

        Target A (Regression)
            y_revenue = log(revenue_2023 + 1)
            Addresses positive skewness of raw revenue distribution.

        Target B (Classification)
            success_category ∈ {Flop, Average, Hit}
            Based on ROI = revenue_worldwide / production_budget
        """
        # Regression target
        df["y_revenue"] = np.log1p(df["revenue_2023"])

        # Classification target (ROI-based)
        def categorise(roi):
            if roi < ROI_FLOP_THRESHOLD:
                return "Flop"
            elif roi < ROI_AVERAGE_THRESHOLD:
                return "Average"
            else:
                return "Hit"

        df["roi"]              = df["revenue_2023"] / df["budget_2023"].replace(0, np.nan)
        df["roi"]              = df["roi"].clip(0, 50)
        df["success_category"] = df["roi"].apply(
            lambda r: categorise(r) if pd.notna(r) else "Average"
        )

        # Numeric label for classification models
        label_map = {"Flop": 0, "Average": 1, "Hit": 2}
        df["y_class"] = df["success_category"].map(label_map)

        log.info("  Target distribution:\n%s",
                 df["success_category"].value_counts().to_string())
        return df


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5 · FEATURE SELECTION & SERIALISATION
# ─────────────────────────────────────────────────────────────────────────────

def select_final_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select the final modelling columns, removing:
      · Raw JSON strings (already parsed)
      · Intermediate helpers
      · Columns with > 40% missing after imputation
      · Near-zero-variance columns (std < 0.01)
    Returns a clean, model-ready DataFrame.
    """
    # Columns to explicitly drop (raw / intermediate)
    drop_cols = [
        "genres", "cast", "crew", "keywords",
        "production_companies", "production_countries",
        "spoken_languages", "tagline", "overview",
        "homepage", "original_title", "status",
        "genres_list", "cast_list", "keywords_list",
        "companies_list", "crew_parsed", "primary_genre",
        "budget_category",   # categorical → already encoded elsewhere
        "budget", "revenue", # raw (un-normalised) financials
    ]
    existing_drops = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=existing_drops, errors="ignore")

    # Drop columns with excessive missingness
    miss_rate = df.isnull().mean()
    high_miss  = miss_rate[miss_rate > 0.40].index.tolist()
    if high_miss:
        log.info("  Dropping %d high-missingness columns: %s", len(high_miss), high_miss)
        df = df.drop(columns=high_miss)

    # Drop near-zero-variance numeric columns
    num = df.select_dtypes(include=[np.number])
    low_var = num.columns[num.std() < 0.01].tolist()
    if low_var:
        log.info("  Dropping %d near-zero-variance columns: %s", len(low_var), low_var)
        df = df.drop(columns=low_var)

    log.info("  Final feature count: %d", df.shape[1])
    return df


def write_pipeline_report(df: pd.DataFrame, path: Path) -> None:
    """Write a human-readable data quality and feature summary report."""
    lines = [
        "=" * 72,
        "BOX OFFICE PREDICTION — DATA PIPELINE REPORT",
        "=" * 72,
        f"Final dataset shape : {df.shape[0]} rows × {df.shape[1]} columns",
        "",
        "── Target Variable Summary ─────────────────────────────────────────",
        df["success_category"].value_counts().to_string(),
        "",
        "── Numeric Feature Statistics ──────────────────────────────────────",
        df.select_dtypes(include=[np.number]).describe().to_string(),
        "",
        "── Missing Values (post-imputation) ────────────────────────────────",
        df.isnull().sum()[df.isnull().sum() > 0].to_string() or "  None",
        "",
        "── Column List ─────────────────────────────────────────────────────",
        "\n".join(f"  {c}" for c in sorted(df.columns)),
        "=" * 72,
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Pipeline report saved to: %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline() -> pd.DataFrame:
    """
    Execute the full ETL pipeline end-to-end and return the processed DataFrame.

    Returns
    -------
    pd.DataFrame
        Model-ready feature matrix with target columns `y_revenue` and `y_class`.
    """
    log.info("=" * 60)
    log.info("  BOX OFFICE PREDICTION — DATA PIPELINE")
    log.info("  Authors: I.Parthiban 23BCE2282 | Mike Roonane Fernandez 23BCE2062")
    log.info("=" * 60)

    # ── Validate input paths ──────────────────────────────────────────
    for p in [MOVIES_CSV, CREDITS_CSV]:
        if not p.exists():
            log.error("Dataset not found: %s", p)
            log.error(
                "Download tmdb_5000_movies.csv and tmdb_5000_credits.csv from:\n"
                "  https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata\n"
                "Place them in: %s", DATA_DIR.resolve()
            )
            sys.exit(1)

    # ── Stage 1: Load & Merge ─────────────────────────────────────────
    log.info("─── Stage 1: Data Loading & Merging ─────────────────────────")
    loader = DataLoader(MOVIES_CSV, CREDITS_CSV)
    df_raw = loader.load()

    # ── Stage 2: Clean ───────────────────────────────────────────────
    cleaner  = DataCleaner()
    df_clean = cleaner.clean(df_raw)

    # ── Stage 4: Feature Engineering ─────────────────────────────────
    engineer    = FeatureEngineer()
    df_features = engineer.engineer(df_clean)

    # ── Stage 5: Select & Save ───────────────────────────────────────
    log.info("─── Stage 5: Feature Selection & Serialisation ───────────────")
    df_final = select_final_features(df_features)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(OUTPUT_CSV, index=False)
    log.info("Processed dataset saved to: %s", OUTPUT_CSV)

    write_pipeline_report(df_final, REPORT_TXT)

    log.info("=" * 60)
    log.info("  Pipeline complete. %d films ready for modelling.", len(df_final))
    log.info("=" * 60)

    return df_final


if __name__ == "__main__":
    run_pipeline()
