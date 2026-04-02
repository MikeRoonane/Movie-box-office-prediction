"""
================================================================================
Box Office Success Prediction — Model Training & Evaluation
================================================================================
Authors  : I.Parthiban 23BCE2282, Mike Roonane Fernandez 23BCE2062
Date     : February 2026
Project  : Capstone — Movie Box Office Prediction using Multi-Source Big Data
           and Machine Learning with IMDb Metadata

Description
-----------
This module implements the full modelling pipeline described in Sections 3.7–3.9
and the evaluation framework of Section 4.  It produces every result table and
figure cited in Section 5 of the report.

Pipeline Stages
───────────────
  A · Chronological train / validation / test split (Section 3.8.1)
  B · Feature standardisation using StandardScaler (Section 3.8.3)
  C · Base learner training with grid-search hyperparameter optimisation
        · XGBoost (Chen & Guestrin, 2016)
        · Random Forest (Breiman, 2001)
        · Neural Network / MLP
  D · Stacking Ensemble (Wolpert, 1992) — Section 3.9.2
        · Level-0: three base learners above
        · Level-1 meta-learner: Ridge (regression) / Logistic Regression (clf)
        · Trained on out-of-fold predictions to prevent data leakage
  E · Evaluation
        Regression  : MAE, RMSE, R², MAPE (Section 4.1.1)
        Classification: Accuracy, Precision, Recall, F1, AUC-ROC (Section 4.1.2)
        Statistical significance: paired t-tests (Section 4.3)
  F · Visualisations (Section 5)
        · Confusion Matrix (Figure 2)
        · SHAP Feature Importance bar chart (Table 3)
        · Revenue vs. Budget scatter with regression line

Usage
─────
    python model_training.py

    Prerequisite: run data_pipeline.py first to generate processed_movies.csv

Output files
────────────
    models/                  — serialised model artefacts (.pkl / .joblib)
    figures/                 — all publication-quality figures (.png)
    results/model_report.txt — full numerical results table
================================================================================
"""

import logging
import os
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe on any system)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import shap
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import (
    LogisticRegression,
    Ridge,
)
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler, label_binarize
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

PROCESSED_CSV = Path("processed_movies.csv")
MODELS_DIR    = Path("models")
FIGURES_DIR   = Path("figures")
RESULTS_DIR   = Path("results")

RANDOM_STATE  = 42
CV_FOLDS      = 5       # k for out-of-fold stacking (Section 3.8.2)

# Colours consistent with the report figures
PALETTE = {
    "flop"   : "#e74c3c",
    "average": "#f39c12",
    "hit"    : "#2ecc71",
    "pred"   : "#2980b9",
    "actual" : "#8e44ad",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# A · DATA LOADING & CHRONOLOGICAL SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def load_and_split(
    csv_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Load the processed feature matrix and perform a chronological train /
    validation / test split as described in Section 3.8.1:

        Training   : release_year ≤ 2018   (~75 % of data)
        Validation : 2019 ≤ year ≤ 2021    (~15 %)
        Test       : year ≥ 2022           (~10 %)

    This temporal split prevents data leakage and mirrors realistic
    deployment conditions where future films must be forecast from past data.

    Returns
    -------
    train, val, test : pd.DataFrame partitions
    feature_cols     : list of predictor column names
    """
    log.info("Loading processed dataset from %s …", csv_path)
    df = pd.read_csv(csv_path, low_memory=False)
    log.info("  Shape: %d × %d", *df.shape)

    # Identify predictor columns (exclude targets, IDs, raw strings)
    exclude = {
        "id", "imdb_id", "title", "original_language", "director",
        "release_date", "release_year",
        "y_revenue", "y_class", "success_category",
        "roi", "budget_2023", "revenue_2023",
        "star_power_index",       # use log-scaled version
    }
    feature_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude
    ]

    df["release_year"] = pd.to_numeric(df.get("release_year", 2010), errors="coerce").fillna(2010)

    train = df[df["release_year"] <= 2018].copy()
    val   = df[(df["release_year"] >= 2019) & (df["release_year"] <= 2021)].copy()
    test  = df[df["release_year"] >= 2022].copy()

    log.info(
        "  Split → Train: %d | Val: %d | Test: %d",
        len(train), len(val), len(test),
    )

    # If test set is too small (TMDB 5000 only goes to ~2017), fall back to
    # a time-aware random split that still preserves the temporal ordering
    if len(test) < 50:
        log.warning(
            "  Test set < 50 rows (dataset coverage may not reach 2022). "
            "Falling back to 70/15/15 chronological split by index."
        )
        df_sorted = df.sort_values("release_year").reset_index(drop=True)
        n = len(df_sorted)
        i70, i85 = int(0.70 * n), int(0.85 * n)
        train, val, test = df_sorted.iloc[:i70], df_sorted.iloc[i70:i85], df_sorted.iloc[i85:]

    return train, val, test, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# B · PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def prepare_arrays(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple:
    """
    Extract numpy arrays and fit StandardScaler on training data only
    (Section 3.8.3).  Impute any residual NaNs with column medians
    before scaling.
    """
    # Ensure only columns present in df are used
    fc = [c for c in feature_cols if c in train.columns]

    def xy(df, target_reg="y_revenue", target_clf="y_class"):
        X = df[fc].copy()
        # Residual NaN fill with median
        X = X.fillna(X.median(numeric_only=True))
        yr = df[target_reg].fillna(df[target_reg].median()).values
        yc = df[target_clf].fillna(1).astype(int).values
        return X.values, yr, yc

    X_tr, yr_tr, yc_tr = xy(train)
    X_va, yr_va, yc_va = xy(val)
    X_te, yr_te, yc_te = xy(test)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    log.info("  Feature matrix: %d predictors", len(fc))
    return (
        X_tr_s, yr_tr, yc_tr,
        X_va_s, yr_va, yc_va,
        X_te_s, yr_te, yc_te,
        scaler, fc,
    )


# ─────────────────────────────────────────────────────────────────────────────
# C · BASE LEARNER DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

def build_base_regressors() -> Dict:
    """
    Base learners for the regression stacking ensemble (Section 3.7.1).
    Hyperparameters match Table 5 (Appendix A.1) of the report.
    """
    return {
        "XGBoost": XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            random_state=RANDOM_STATE,
            verbosity=0,
            n_jobs=-1,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=500,
            max_depth=25,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "NeuralNetwork": MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=0.001,          # L2 regularisation (≈ dropout proxy in sklearn MLP)
            learning_rate_init=0.001,
            max_iter=100,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=RANDOM_STATE,
        ),
    }


def build_base_classifiers() -> Dict:
    """
    Base learners for the classification stacking ensemble (Section 3.7.2).
    """
    return {
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            num_class=3,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            verbosity=0,
            use_label_encoder=False,
            n_jobs=-1,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            max_depth=25,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "NeuralNetwork": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=100,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=RANDOM_STATE,
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# D · STACKING ENSEMBLE (Wolpert, 1992)
# ─────────────────────────────────────────────────────────────────────────────

class StackingEnsemble:
    """
    Two-level stacking ensemble (Section 3.9.2).

        Level 0 · Base models trained via k-fold cross-validation.
                  Out-of-fold (OOF) predictions form the meta-feature matrix.
        Level 1 · Meta-learner trained on the OOF meta-features.
                  Ridge regression (regression) or Logistic Regression (clf).

    This architecture prevents overfitting of the meta-learner because it
    never sees the training labels directly from the base models; only the
    generalised OOF predictions are used.
    """

    def __init__(self, base_models: Dict, meta_learner, task: str = "regression"):
        """
        Parameters
        ----------
        base_models  : dict of name → estimator
        meta_learner : sklearn estimator (Ridge or LogisticRegression)
        task         : "regression" or "classification"
        """
        self.base_models   = base_models
        self.meta_learner  = meta_learner
        self.task          = task
        self._fitted_bases = {}
        self._kf           = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # ------------------------------------------------------------------
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "StackingEnsemble":
        log.info("  Generating out-of-fold predictions (%d folds) …", CV_FOLDS)

        n, m = X_train.shape[0], len(self.base_models)
        oof_preds = np.zeros((n, m))  # meta-feature matrix

        for i, (name, model) in enumerate(self.base_models.items()):
            log.info("    Base learner [%d/%d]: %s", i + 1, m, name)
            if self.task == "regression":
                oof_preds[:, i] = cross_val_predict(
                    model, X_train, y_train, cv=self._kf, n_jobs=-1
                )
            else:
                # Predict class probabilities; keep only the first two columns for
                # meta-learner to remain parsimonious (binary-like encoding)
                proba = cross_val_predict(
                    model, X_train, y_train,
                    cv=self._kf, method="predict_proba", n_jobs=-1
                )
                oof_preds[:, i] = np.argmax(proba, axis=1)

            # Fit the base model on the FULL training set for inference
            model.fit(X_train, y_train)
            self._fitted_bases[name] = model

        # Train meta-learner on OOF meta-features
        log.info("  Training meta-learner …")
        self.meta_learner.fit(oof_preds, y_train)
        return self

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        meta_X = self._make_meta_features(X)
        return self.meta_learner.predict(meta_X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Only valid when task == 'classification'."""
        meta_X = self._make_meta_features(X)
        return self.meta_learner.predict_proba(meta_X)

    # ------------------------------------------------------------------
    def _make_meta_features(self, X: np.ndarray) -> np.ndarray:
        m = len(self._fitted_bases)
        meta = np.zeros((X.shape[0], m))
        for i, (name, model) in enumerate(self._fitted_bases.items()):
            if self.task == "regression":
                meta[:, i] = model.predict(X)
            else:
                meta[:, i] = model.predict(X)
        return meta

    def get_base_models(self) -> Dict:
        return self._fitted_bases


# ─────────────────────────────────────────────────────────────────────────────
# E · EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> Dict:
    """Compute the four regression metrics from Section 4.1.1."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2   = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return {"Model": name, "MAE": mae, "RMSE": rmse, "R²": r2, "MAPE (%)": mape}


def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray,
                             y_proba: np.ndarray, name: str) -> Dict:
    """Compute the five classification metrics from Section 4.1.2."""
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    # Multi-class AUC-ROC (OvR strategy)
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    try:
        auc_roc = roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro")
    except ValueError:
        auc_roc = float("nan")

    return {
        "Model": name, "Accuracy": acc, "Precision": prec,
        "Recall": rec, "F1-Score": f1, "AUC-ROC": auc_roc,
    }


def compare_models_regression(
    models: Dict, X_test: np.ndarray, y_test: np.ndarray
) -> pd.DataFrame:
    """
    Evaluate multiple regression models and return a comparison table
    matching Table 1 in the report.
    """
    rows = []
    for name, model in models.items():
        preds = model.predict(X_test)
        rows.append(evaluate_regression(y_test, preds, name))
    return pd.DataFrame(rows).sort_values("R²", ascending=False)


def compare_models_classification(
    models: Dict, X_test: np.ndarray, y_test: np.ndarray
) -> pd.DataFrame:
    """
    Evaluate multiple classification models — Table 2 in the report.
    """
    rows = []
    for name, model in models.items():
        preds = model.predict(X_test)
        try:
            proba = model.predict_proba(X_test)
        except AttributeError:
            proba = np.zeros((len(y_test), 3))
        rows.append(evaluate_classification(y_test, preds, proba, name))
    return pd.DataFrame(rows).sort_values("Accuracy", ascending=False)


def run_paired_ttest(
    model_a_preds: np.ndarray,
    model_b_preds: np.ndarray,
    y_true: np.ndarray,
    task: str = "regression",
) -> Tuple[float, float]:
    """
    Paired t-test for significance of performance difference (Section 4.3).
    Returns (t-statistic, p-value).
    """
    if task == "regression":
        err_a = np.abs(y_true - model_a_preds)
        err_b = np.abs(y_true - model_b_preds)
    else:
        err_a = (y_true != model_a_preds).astype(float)
        err_b = (y_true != model_b_preds).astype(float)

    t_stat, p_val = stats.ttest_rel(err_a, err_b)
    return float(t_stat), float(p_val)


# ─────────────────────────────────────────────────────────────────────────────
# F · VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

class Visualiser:
    """Generates all figures cited in Section 5 of the report."""

    LABEL_NAMES = {0: "Flop", 1: "Average", 2: "Hit"}

    def __init__(self, output_dir: Path):
        self.out = output_dir
        self.out.mkdir(parents=True, exist_ok=True)
        plt.rcParams.update({
            "font.family"    : "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi"     : 150,
        })

    # ──────────────────────────────────────────────────────────────────
    def confusion_matrix_plot(
        self, y_true: np.ndarray, y_pred: np.ndarray,
        accuracy: float, title: str = "Stacking Ensemble Classification"
    ) -> Path:
        """
        Publication-quality confusion matrix — Figure 2 of the report.
        """
        cm     = confusion_matrix(y_true, y_pred)
        labels = [self.LABEL_NAMES[i] for i in sorted(self.LABEL_NAMES)]
        n_cls  = cm.shape[0]

        fig, ax = plt.subplots(figsize=(7, 5.5))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set(
            xticks=np.arange(n_cls), yticks=np.arange(n_cls),
            xticklabels=labels, yticklabels=labels,
            xlabel="Predicted Category", ylabel="Actual Category",
            title=f"Confusion Matrix: {title}\n(Test Set Performance)",
        )
        ax.xaxis.set_label_position("bottom")
        ax.xaxis.tick_bottom()

        # Add text annotations
        thresh = cm.max() / 2.0
        for i in range(n_cls):
            for j in range(n_cls):
                ax.text(
                    j, i, f"{cm[i, j]}",
                    ha="center", va="center", fontsize=14, fontweight="bold",
                    color="white" if cm[i, j] > thresh else "black",
                )

        # ROI range sub-labels on y-axis
        roi_labels = [
            "Flop\n(ROI < 1.5)", "Average\n(1.5 ≤ ROI < 3.0)", "Hit\n(ROI ≥ 3.0)"
        ]
        ax.set_yticklabels(roi_labels[:n_cls], fontsize=9)
        ax.set_xticklabels(
            ["Flop\n(ROI < 1.5)", "Average\n(1.5 ≤ ROI < 3.0)", "Hit\n(ROI ≥ 3.0)"][:n_cls],
            fontsize=9,
        )

        correct = np.trace(cm)
        total   = cm.sum()
        fig.suptitle(
            f"Overall Accuracy: {accuracy:.1%} ({correct}/{total} correct predictions)",
            fontsize=10, fontweight="bold", color="#2c3e50",
        )
        plt.tight_layout()

        path = self.out / "confusion_matrix.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        log.info("  ✓ Confusion matrix saved to %s", path)
        return path

    # ──────────────────────────────────────────────────────────────────
    def shap_importance_plot(
        self,
        model,
        X_train: np.ndarray,
        feature_names: List[str],
        top_n: int = 15,
        model_name: str = "XGBoost",
    ) -> Path:
        """
        SHAP feature importance bar chart — Table 3 / Figure equivalent
        from Section 5.4 of the report.

        Uses TreeExplainer for tree-based models and KernelExplainer
        as a fallback for neural networks.
        """
        log.info("  Computing SHAP values (this may take 30–90 s) …")

        try:
            explainer  = shap.TreeExplainer(model)
            shap_vals  = explainer.shap_values(X_train[:500])   # subsample for speed
        except Exception:
            log.warning("  TreeExplainer failed; using KernelExplainer (slower) …")
            bg         = shap.kmeans(X_train, 50)
            explainer  = shap.KernelExplainer(model.predict, bg)
            shap_vals  = explainer.shap_values(X_train[:200])

        # For multi-output shap_vals (classification), use class-0 importances
        if isinstance(shap_vals, list):
            sv = np.abs(np.array(shap_vals)).mean(axis=0)
        else:
            sv = np.abs(shap_vals)

        mean_abs_shap = sv.mean(axis=0)
        importance_df = pd.DataFrame({
            "Feature"   : feature_names[:len(mean_abs_shap)],
            "Importance": mean_abs_shap,
        }).sort_values("Importance", ascending=False).head(top_n)

        # Colour-code by feature family
        def feature_colour(name):
            if any(k in name for k in ["budget", "studio", "roi"]):
                return "#e74c3c"   # Financial → red
            if any(k in name for k in ["star", "director", "franchise"]):
                return "#3498db"   # Metadata → blue
            if any(k in name for k in ["buzz", "popularity", "vote", "rating", "trailer"]):
                return "#2ecc71"   # Engagement → green
            if any(k in name for k in ["month", "summer", "holiday", "season", "dow", "year"]):
                return "#9b59b6"   # Temporal → purple
            return "#95a5a6"       # Other → grey

        colours = [feature_colour(f) for f in importance_df["Feature"]]

        fig, ax = plt.subplots(figsize=(9, 6))
        bars = ax.barh(
            importance_df["Feature"][::-1],
            importance_df["Importance"][::-1],
            color=colours[::-1], edgecolor="white", linewidth=0.5,
        )
        ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
        ax.set_title(
            f"Feature Importance — {model_name}\n"
            "(Colours: 🔴 Financial  🔵 Metadata  🟢 Engagement  🟣 Temporal)",
            fontsize=11,
        )

        # Add value labels
        for bar, val in zip(bars, importance_df["Importance"][::-1]):
            ax.text(
                val + max(importance_df["Importance"]) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8,
            )

        plt.tight_layout()
        path = self.out / "shap_importance.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        log.info("  ✓ SHAP importance plot saved to %s", path)
        return path

    # ──────────────────────────────────────────────────────────────────
    def revenue_vs_budget_scatter(
        self,
        df: pd.DataFrame,
        y_pred: np.ndarray,
        title: str = "Revenue vs. Budget — Stacking Ensemble",
    ) -> Path:
        """
        Scatter plot of actual log-revenue vs log-budget, overlaid with
        model predictions and a linear regression trend line.
        """
        df = df.copy()
        df = df[df["budget_2023"].notna() & df["revenue_2023"].notna()]
        df = df.head(len(y_pred))

        x  = np.log1p(df["budget_2023"].values)
        ya = np.log1p(df["revenue_2023"].values)
        yp = y_pred[:len(x)]

        fig, ax = plt.subplots(figsize=(9, 6))

        ax.scatter(x, ya, alpha=0.35, s=18, color=PALETTE["actual"],  label="Actual Revenue")
        ax.scatter(x, yp, alpha=0.35, s=18, color=PALETTE["pred"],   label="Predicted Revenue", marker="^")

        # Regression line on actual data
        m, b, r, p, _ = stats.linregress(x, ya)
        x_line = np.linspace(x.min(), x.max(), 200)
        ax.plot(x_line, m * x_line + b, color="#2c3e50", linewidth=2,
                linestyle="--", label=f"Trend (R²={r**2:.3f})")

        # Perfect-prediction line
        lims = [min(x.min(), yp.min()), max(x.max(), yp.max())]
        ax.plot(lims, lims, color="#e74c3c", linewidth=1.2, alpha=0.5, label="Perfect Prediction")

        ax.set_xlabel("log(Production Budget + 1) — 2023 USD", fontsize=11)
        ax.set_ylabel("log(Box Office Revenue + 1) — 2023 USD", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9)
        plt.tight_layout()

        path = self.out / "revenue_vs_budget.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        log.info("  ✓ Revenue vs. Budget scatter saved to %s", path)
        return path

    # ──────────────────────────────────────────────────────────────────
    def model_comparison_bar(self, reg_df: pd.DataFrame, clf_df: pd.DataFrame) -> Path:
        """Side-by-side bar chart comparing model R² and Accuracy."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        # Regression R²
        colours = ["#2ecc71" if i == 0 else "#3498db" for i in range(len(reg_df))]
        ax1.barh(reg_df["Model"][::-1], reg_df["R²"][::-1], color=colours[::-1])
        ax1.set_xlabel("R²", fontsize=11)
        ax1.set_title("Regression — R² Score", fontsize=12)
        ax1.axvline(0.8, color="#e74c3c", linestyle="--", linewidth=1, label="R²=0.80 target")
        ax1.legend(fontsize=8)

        # Classification Accuracy
        colours2 = ["#2ecc71" if i == 0 else "#9b59b6" for i in range(len(clf_df))]
        ax2.barh(clf_df["Model"][::-1], clf_df["Accuracy"][::-1], color=colours2[::-1])
        ax2.set_xlabel("Accuracy", fontsize=11)
        ax2.set_title("Classification — Accuracy", fontsize=12)
        ax2.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax2.axvline(0.77, color="#e74c3c", linestyle="--", linewidth=1, label="77% target")
        ax2.legend(fontsize=8)

        plt.suptitle(
            "Model Comparison — Regression & Classification Performance",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout()
        path = self.out / "model_comparison.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        log.info("  ✓ Model comparison bar chart saved to %s", path)
        return path

    # ──────────────────────────────────────────────────────────────────
    def roc_curve_plot(
        self, y_true: np.ndarray, y_proba: np.ndarray
    ) -> Path:
        """Multi-class AUC-ROC curves (OvR) for the stacking ensemble."""
        class_names = ["Flop", "Average", "Hit"]
        colours_roc  = [PALETTE["flop"], PALETTE["average"], PALETTE["hit"]]
        y_bin = label_binarize(y_true, classes=[0, 1, 2])

        fig, ax = plt.subplots(figsize=(7, 5))
        for i, (cname, col) in enumerate(zip(class_names, colours_roc)):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            auc_val = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=col, lw=2, label=f"{cname} (AUC = {auc_val:.3f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves — Stacking Ensemble (One-vs-Rest)")
        ax.legend()
        plt.tight_layout()
        path = self.out / "roc_curves.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        log.info("  ✓ ROC curves saved to %s", path)
        return path


# ─────────────────────────────────────────────────────────────────────────────
# RESULTS REPORT WRITER
# ─────────────────────────────────────────────────────────────────────────────

def write_results_report(
    reg_table: pd.DataFrame,
    clf_table: pd.DataFrame,
    t_stats: Dict,
    output_path: Path,
) -> None:
    lines = [
        "=" * 76,
        "BOX OFFICE SUCCESS PREDICTION — MODEL RESULTS REPORT",
        "Authors: I.Parthiban 23BCE2282 | Mike Roonane Fernandez 23BCE2062",
        "=" * 76,
        "",
        "── Table 1: Regression Model Comparison ───────────────────────────────",
        reg_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "── Table 2: Classification Model Comparison ───────────────────────────",
        clf_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "── Statistical Significance Testing (Section 4.3) ─────────────────────",
    ]
    for key, (t, p) in t_stats.items():
        sig = "✓ SIGNIFICANT" if p < 0.05 else "✗ NOT significant"
        lines.append(f"  {key:<50}  t={t:+.3f}  p={p:.4f}  → {sig}")

    lines += [
        "",
        "Note: All results computed on the held-out test set (chronological split).",
        "=" * 76,
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Results report saved to: %s", output_path)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────

def save_artefacts(models: Dict, scaler: StandardScaler, feature_cols: List[str]) -> None:
    """Serialise trained models and the scaler for use by the Streamlit app."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        path = MODELS_DIR / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f)
        log.info("  Saved: %s", path)

    with open(MODELS_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(MODELS_DIR / "feature_cols.pkl", "wb") as f:
        pickle.dump(feature_cols, f)
    log.info("  Saved: scaler + feature_cols")


def load_artefacts() -> Tuple:
    """Load serialised models and scaler (used by Streamlit app)."""
    models = {}
    for p in MODELS_DIR.glob("*.pkl"):
        if p.stem in ("scaler", "feature_cols"):
            continue
        with open(p, "rb") as f:
            models[p.stem] = pickle.load(f)
    with open(MODELS_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(MODELS_DIR / "feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    return models, scaler, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def run_training() -> None:
    log.info("=" * 60)
    log.info("  BOX OFFICE PREDICTION — MODEL TRAINING")
    log.info("  Authors: I.Parthiban 23BCE2282 | Mike Roonane Fernandez 23BCE2062")
    log.info("=" * 60)

    if not PROCESSED_CSV.exists():
        log.error("processed_movies.csv not found — run data_pipeline.py first.")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── A: Load & Split ──────────────────────────────────────────────
    train, val, test, feature_cols = load_and_split(PROCESSED_CSV)

    # ── B: Preprocess ────────────────────────────────────────────────
    (
        X_tr, yr_tr, yc_tr,
        X_va, yr_va, yc_va,
        X_te, yr_te, yc_te,
        scaler, feature_cols,
    ) = prepare_arrays(train, val, test, feature_cols)

    # Combine train+val for final model fitting (standard practice)
    X_tv  = np.vstack([X_tr, X_va])
    yr_tv = np.concatenate([yr_tr, yr_va])
    yc_tv = np.concatenate([yc_tr, yc_va])

    # ── C: Train Base Learners individually (for comparison tables) ──
    log.info("─── Stage C: Training Individual Base Learners ───────────────")
    reg_base_models = build_base_regressors()
    clf_base_models = build_base_classifiers()

    individual_reg, individual_clf = {}, {}

    for name, model in reg_base_models.items():
        log.info("  Training regressor: %s …", name)
        model.fit(X_tv, yr_tv)
        individual_reg[name] = model

    for name, model in clf_base_models.items():
        log.info("  Training classifier: %s …", name)
        model.fit(X_tv, yc_tv)
        individual_clf[name] = model

    # ── D: Stacking Ensembles ────────────────────────────────────────
    log.info("─── Stage D: Training Stacking Ensembles ─────────────────────")

    # Regression stacking
    reg_stack = StackingEnsemble(
        base_models  = build_base_regressors(),
        meta_learner = Ridge(alpha=1.0),
        task         = "regression",
    )
    reg_stack.fit(X_tv, yr_tv)

    # Classification stacking
    clf_stack = StackingEnsemble(
        base_models  = build_base_classifiers(),
        meta_learner = LogisticRegression(
            C=1.0, max_iter=500, random_state=RANDOM_STATE
        ),
        task         = "classification",
    )
    clf_stack.fit(X_tv, yc_tv)

    # ── E: Evaluation ────────────────────────────────────────────────
    log.info("─── Stage E: Model Evaluation ────────────────────────────────")

    # Add stacking ensembles to comparison pool
    all_reg = {**individual_reg, "Stacking Ensemble": reg_stack}
    all_clf = {**individual_clf, "Stacking Ensemble": clf_stack}

    reg_table = compare_models_regression(all_reg, X_te, yr_te)
    clf_table = compare_models_classification(all_clf, X_te, yc_te)

    log.info("\n%s", "─" * 70)
    log.info("REGRESSION RESULTS:\n%s", reg_table.to_string(index=False))
    log.info("\nCLASSIFICATION RESULTS:\n%s", clf_table.to_string(index=False))

    # Statistical significance: stacking vs. best individual (XGBoost)
    reg_stack_preds   = reg_stack.predict(X_te)
    reg_xgb_preds     = individual_reg["XGBoost"].predict(X_te)
    clf_stack_preds   = clf_stack.predict(X_te)
    clf_xgb_preds     = individual_clf["XGBoost"].predict(X_te)

    t_stats = {
        "Stacking vs XGBoost (Regression)":      run_paired_ttest(reg_stack_preds, reg_xgb_preds, yr_te, "regression"),
        "Stacking vs XGBoost (Classification)":  run_paired_ttest(clf_stack_preds, clf_xgb_preds, yc_te, "classification"),
        "Stacking vs RF (Regression)":            run_paired_ttest(reg_stack_preds, individual_reg["RandomForest"].predict(X_te), yr_te, "regression"),
        "Stacking vs RF (Classification)":        run_paired_ttest(clf_stack_preds, individual_clf["RandomForest"].predict(X_te), yc_te, "classification"),
    }

    write_results_report(reg_table, clf_table, t_stats, RESULTS_DIR / "model_report.txt")

    # ── F: Visualisations ────────────────────────────────────────────
    log.info("─── Stage F: Generating Visualisations ───────────────────────")
    vis = Visualiser(FIGURES_DIR)

    # 1. Confusion Matrix
    stk_acc = accuracy_score(yc_te, clf_stack_preds)
    vis.confusion_matrix_plot(yc_te, clf_stack_preds, stk_acc)

    # 2. SHAP Feature Importance (use XGBoost — fastest with TreeExplainer)
    vis.shap_importance_plot(
        model        = individual_reg["XGBoost"],
        X_train      = X_tv,
        feature_names = feature_cols,
        model_name   = "XGBoost Regressor",
    )

    # 3. Revenue vs. Budget scatter
    vis.revenue_vs_budget_scatter(test, reg_stack_preds)

    # 4. Model comparison bar
    vis.model_comparison_bar(reg_table, clf_table)

    # 5. AUC-ROC curves
    clf_stack_proba = clf_stack.predict_proba(X_te)
    vis.roc_curve_plot(yc_te, clf_stack_proba)

    # ── Save Artefacts ───────────────────────────────────────────────
    log.info("─── Saving model artefacts ────────────────────────────────────")
    all_models_to_save = {
        "stacking_regressor"   : reg_stack,
        "stacking_classifier"  : clf_stack,
        "xgboost_regressor"    : individual_reg["XGBoost"],
        "xgboost_classifier"   : individual_clf["XGBoost"],
        "randomforest_regressor": individual_reg["RandomForest"],
    }
    save_artefacts(all_models_to_save, scaler, feature_cols)

    log.info("=" * 60)
    log.info("  Training complete. All artefacts saved.")
    log.info("  · Models   → %s/", MODELS_DIR)
    log.info("  · Figures  → %s/", FIGURES_DIR)
    log.info("  · Results  → %s/", RESULTS_DIR)
    log.info("=" * 60)


if __name__ == "__main__":
    run_training()
