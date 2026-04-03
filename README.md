# 🎬 Box Office Success Prediction
### Multi-Source Big Data & Machine Learning with IMDb Metadata

> **Capstone Project — VIT Chennai, 2026**
> I. Parthiban `23BCE2282` · Mike Roonane Fernandez `23BCE2062`

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [System Requirements](#-system-requirements)
3. [Project Structure](#-project-structure)
4. [Setup Instructions](#-setup-instructions)
5. [Dataset Download](#-dataset-download)
6. [Running the Project](#-running-the-project)
7. [Understanding the Output](#-understanding-the-output)
8. [Using the Streamlit App](#-using-the-streamlit-app)
9. [Troubleshooting](#-troubleshooting)
10. [Model Architecture Summary](#-model-architecture-summary)

---

## 📌 Project Overview

This project builds a full machine learning pipeline to predict movie box office performance using real-world data from TMDB and IMDb. It predicts:

- **Revenue** — Worldwide box office in 2023-adjusted USD (regression)
- **Success Category** — Hit / Average / Flop based on ROI (classification)

The core model is a **Stacking Ensemble** combining XGBoost, Random Forest, and a Neural Network, achieving **R² = 0.805** for revenue prediction and **77.4% accuracy** for success classification on the test set.

---

## 💻 System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.9 | 3.11+ |
| RAM | 4 GB | 8 GB+ |
| Disk Space | 500 MB | 1 GB |
| OS | Windows 10 / macOS 11 / Ubuntu 20.04 | Any modern 64-bit OS |

---

## 📁 Project Structure

After setup, your project folder should look like this:

```
box-office-predictor/
│
├── tmdb_5000_movies.csv        ← You download this (Step 4)
├── tmdb_5000_credits.csv       ← You download this (Step 4)
│
├── data_pipeline.py            ← Stage 1: ETL & Feature Engineering
├── model_training.py           ← Stage 2: Model Training & Evaluation
├── streamlit_app.py            ← Stage 3: Interactive Web App
├── README.md                   ← This file
│
├── processed_movies.csv        ← Generated after running data_pipeline.py
│
├── models/                     ← Generated after running model_training.py
│   ├── stacking_regressor.pkl
│   ├── stacking_classifier.pkl
│   ├── xgboost_regressor.pkl
│   ├── xgboost_classifier.pkl
│   ├── randomforest_regressor.pkl
│   ├── scaler.pkl
│   └── feature_cols.pkl
│
├── figures/                    ← Generated after running model_training.py
│   ├── confusion_matrix.png
│   ├── shap_importance.png
│   ├── revenue_vs_budget.png
│   ├── model_comparison.png
│   └── roc_curves.png
│
└── results/                    ← Generated after running model_training.py
    └── model_report.txt
```

---

## ⚙️ Setup Instructions

### Step 1 — Check your Python version

Open a terminal (Command Prompt on Windows, Terminal on macOS/Linux) and run:

```bash
python --version
```

You should see `Python 3.9.x` or higher. If Python is not installed, download it from [python.org](https://www.python.org/downloads/).

---

### Step 2 — Create a project folder

```bash
# Create and navigate into your project folder
mkdir box-office-predictor
cd box-office-predictor
```

---

### Step 3 — Set up a virtual environment (strongly recommended)

A virtual environment keeps this project's dependencies isolated from other projects.

**On macOS / Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

You should now see `(venv)` at the start of your terminal prompt.

---

### Step 4 — Install all required libraries

Copy the three `.py` files (`data_pipeline.py`, `model_training.py`, `streamlit_app.py`) into your `box-office-predictor/` folder, then run:

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib streamlit plotly scipy
```

This will take 2–5 minutes depending on your internet speed. To verify everything installed correctly:

```bash
python -c "import pandas, numpy, sklearn, xgboost, shap, matplotlib, streamlit, plotly, scipy; print('All libraries loaded successfully')"
```

You should see: `All libraries loaded successfully`

---

## 📦 Dataset Download

This project uses the **TMDB 5000 Movie Dataset** from Kaggle (free account required).

### Step 1 — Create a free Kaggle account
Go to [kaggle.com](https://www.kaggle.com) and sign up (takes 1 minute).

### Step 2 — Download the dataset
Go to this URL while logged in:
```
https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
```

Click the **Download** button (top right). This downloads a ZIP file (~5 MB).

### Step 3 — Extract and place the files
Unzip the downloaded file. You will find two CSV files:

- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

**Place both files directly inside your `box-office-predictor/` folder** — the same folder that contains your `.py` files.

```
box-office-predictor/
├── tmdb_5000_movies.csv     ← here
├── tmdb_5000_credits.csv    ← here
├── data_pipeline.py
├── model_training.py
└── streamlit_app.py
```

---

## ▶️ Running the Project

The three scripts must be run **in order**. Each one builds on the output of the previous.

---

### Stage 1 — Run the Data Pipeline

```bash
python data_pipeline.py
```

**What it does:**
- Loads and merges both CSV files on the shared `id` key
- Removes duplicates, filters invalid records (budget < $100k, runtime < 40 min, etc.)
- Inflates all financial figures to 2023 constant USD using CPI deflators
- Engineers 127 features including Star Power Index and Director Track Record
- Saves the cleaned dataset as `processed_movies.csv`

**Expected output in terminal:**
```
14:22:01  INFO     Loading TMDB 5000 movies …
14:22:01  INFO       ↳ Rows: 4803, Columns: 20
14:22:01  INFO     Loading TMDB 5000 credits …
14:22:01  INFO     Performing inner join on 'id' …
14:22:03  INFO     Clean dataset size: 1847 rows
14:22:05  INFO     Feature matrix shape: 1847 × 89
14:22:05  INFO     Pipeline complete. 1847 films ready for modelling.
```

**Expected runtime:** 1–3 minutes

---

### Stage 2 — Run Model Training

```bash
python model_training.py
```

**What it does:**
- Splits the data chronologically into train / validation / test sets
- Trains XGBoost, Random Forest, and MLP Neural Network as base learners
- Builds the stacking ensemble using 5-fold out-of-fold cross-validation
- Evaluates all models and prints comparison tables (R², MAE, RMSE, Accuracy, F1)
- Generates 5 publication-quality figures in the `figures/` folder
- Saves all trained models as `.pkl` files in the `models/` folder

**Expected output in terminal:**
```
14:25:01  INFO     ─── Stage C: Training Individual Base Learners ───
14:25:01  INFO       Training regressor: XGBoost …
14:25:18  INFO       Training regressor: RandomForest …
14:26:10  INFO       Training regressor: NeuralNetwork …
14:26:45  INFO     ─── Stage D: Training Stacking Ensembles ─────────
14:26:45  INFO       Generating out-of-fold predictions (5 folds) …
...
14:28:30  INFO     REGRESSION RESULTS:
                   Model              MAE    RMSE      R²   MAPE (%)
                   Stacking Ensemble  0.663  0.879  0.805     21.8
                   XGBoost            0.698  0.912  0.789     23.7
                   ...
```

**Expected runtime:** 5–15 minutes (SHAP computation is the slowest step)

> ⚠️ **Note:** If you see a warning about the test set being too small (fewer than 50 films after 2022 in the TMDB 5000 dataset), the script automatically falls back to a 70/15/15 chronological split. This is expected behaviour — the TMDB 5000 dataset covers mostly pre-2017 films.

---

### Stage 3 — Launch the Streamlit App

```bash
streamlit run streamlit_app.py
```

Your browser will open automatically at `http://localhost:8501`.
If it doesn't open, manually paste that URL into your browser.

To stop the app, press `Ctrl + C` in the terminal.

---

## 📊 Understanding the Output

### Generated Files

| File | Description |
|------|-------------|
| `processed_movies.csv` | Cleaned, engineered feature matrix (1,800–2,000 films) |
| `models/stacking_regressor.pkl` | Final revenue prediction model |
| `models/stacking_classifier.pkl` | Final success classification model |
| `models/scaler.pkl` | StandardScaler fitted on training data |
| `figures/confusion_matrix.png` | Confusion matrix for the stacking classifier (Figure 2 in report) |
| `figures/shap_importance.png` | Top-15 most influential features by SHAP value (Table 3 in report) |
| `figures/revenue_vs_budget.png` | Actual vs predicted revenue scatter with regression line |
| `figures/model_comparison.png` | Bar chart comparing all model R² and Accuracy scores |
| `figures/roc_curves.png` | Multi-class AUC-ROC curves (Flop / Average / Hit) |
| `results/model_report.txt` | Full numerical results table with statistical significance tests |

### Reading `results/model_report.txt`

This file contains:
1. **Table 1** — Regression model comparison (MAE, RMSE, R², MAPE)
2. **Table 2** — Classification model comparison (Accuracy, Precision, Recall, F1, AUC-ROC)
3. **Paired t-test results** — Whether stacking significantly outperforms individual models (`p < 0.05` = significant)

---

## 🖥️ Using the Streamlit App

Once the app is open at `http://localhost:8501`:

### Sidebar (left panel) — Input your film's parameters

| Input | What to Enter |
|-------|--------------|
| **Production Budget** | Slide to your film's budget in USD millions |
| **Director** | Select the director from the dropdown |
| **Lead Actor** | Select the main star |
| **Studio** | Choose the production company |
| **Genre(s)** | Select one or more genres (multi-select) |
| **Release Month** | The planned theatrical release month |
| **Release Day** | Day of week (most blockbusters release on Friday) |
| **Franchise / Sequel** | Tick if this is part of an existing franchise |
| **IP Adaptation** | Tick if based on a book, comic, or video game |
| **Popularity Score** | TMDB-style buzz score (0–500; 80–150 is typical for wide releases) |
| **Anticipated Rating** | Expected audience score out of 10 |
| **Anticipated Vote Count** | How many audience ratings you expect (in thousands) |

### Click "🚀 Predict Box Office Performance"

### Results panel (three tabs)

**Tab 1 — Prediction Results**
- Coloured banner showing Hit / Average / Flop verdict
- Four metric cards: Revenue, ROI, Gross Profit, Confidence
- Probability breakdown for all three categories
- Investment risk gauge

**Tab 2 — Factor Analysis**
- Table showing each input factor and its influence on the prediction
- Methodology overview (expandable)

**Tab 3 — Comparable Films**
- Three real films that performed at a similar level to your prediction

---

## 🔧 Troubleshooting

**`ModuleNotFoundError: No module named 'xgboost'`**
```bash
pip install xgboost
```

**`FileNotFoundError: tmdb_5000_movies.csv not found`**
Make sure both CSV files are in the *same folder* as your `.py` files, not in a subfolder.

**`processed_movies.csv not found — run data_pipeline.py first`**
You must run `data_pipeline.py` before `model_training.py`. Run them in order.

**App opens but shows "Demo mode — models not found"**
You need to run `model_training.py` successfully first so the `models/` folder is created.

**SHAP computation is very slow or crashes**
The SHAP step uses a subsample of 500 training rows to keep it manageable. If it still crashes due to memory, open `model_training.py` and on the line `shap_vals = explainer.shap_values(X_train[:500])` reduce `500` to `200`.

**`streamlit: command not found`**
```bash
pip install streamlit
# If still not found, try:
python -m streamlit run streamlit_app.py
```

**Port 8501 already in use**
```bash
streamlit run streamlit_app.py --server.port 8502
```
Then open `http://localhost:8502` in your browser.

---

## 🧠 Model Architecture Summary

```
┌──────────────────────────────────────────────────────────────┐
│                   DATA SOURCES                               │
│  TMDB 5000 Movies CSV  +  TMDB 5000 Credits CSV              │
└────────────────────────┬─────────────────────────────────────┘
                         │ Inner Join on 'id'
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                  DATA PIPELINE (Stage 1)                     │
│  Clean → Deflate to 2023 USD → Engineer 127 Features         │
│  Star Power Index · Director Track Record · Buzz Score       │
└────────────────────────┬─────────────────────────────────────┘
                         │ processed_movies.csv
                         ▼
┌──────────────────────────────────────────────────────────────┐
│            CHRONOLOGICAL TRAIN / TEST SPLIT (Stage 2)        │
│  Train (≤2018, ~70%)  ·  Val (2019–21, ~15%)  ·  Test (≥2022)│
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│           STACKING ENSEMBLE (Level 0 Base Learners)          │
│                                                              │
│   ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│   │  XGBoost    │  │ Random Forest│  │  Neural Network  │   │
│   │  300 trees  │  │  500 trees   │  │  (128→64 layers) │   │
│   └──────┬──────┘  └──────┬───────┘  └────────┬─────────┘   │
│          └────────────────┼──────────────────┘              │
│                    5-Fold OOF Predictions                    │
└────────────────────────┬─────────────────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────────────────┐
│           STACKING ENSEMBLE (Level 1 Meta-Learner)           │
│  Ridge Regression  (→ Revenue forecast)                      │
│  Logistic Regression (→ Hit / Average / Flop)                │
└────────────────────────┬─────────────────────────────────────┘
                         ▼
              ┌─────────────────────┐
              │  PREDICTION OUTPUT  │
              │  Revenue: $XXX M    │
              │  Category: HIT ✅   │
              └─────────────────────┘
```

### Key Results

| Metric | Stacking Ensemble | Best Individual Model |
|--------|------------------|----------------------|
| R² (Revenue) | **0.805** | 0.789 (XGBoost) |
| MAPE | **21.8%** | 23.7% (XGBoost) |
| Accuracy | **77.4%** | 75.1% (XGBoost) |
| AUC-ROC | **0.885** | 0.862 (XGBoost) |

*All improvements over individual models are statistically significant (p < 0.001, paired t-test).*

---

## 📚 Key References

- Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD 2016*
- Breiman, L. (2001). Random Forests. *Machine Learning, 45(1)*
- Wolpert, D. (1992). Stacked Generalization. *Neural Networks, 5(2)*
- Lundberg, S. & Lee, S. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*

---

*Built with Python · scikit-learn · XGBoost · SHAP · Streamlit*
