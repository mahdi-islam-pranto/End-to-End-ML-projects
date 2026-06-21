# EPL Match Outcome Predictor

An end-to-end machine learning project that predicts the result of an English Premier League football match — **Home Win, Draw, or Away Win** — using only information available before the match starts.

Built with: Python · Scikit-learn · XGBoost · LightGBM · MLflow · FastAPI · Streamlit

---

## Table of Contents

1. [Project in One Sentence](#1-project-in-one-sentence)
2. [The Core Problem](#2-the-core-problem)
3. [Dataset](#3-dataset)
4. [The Leakage Problem — The Most Important Concept](#4-the-leakage-problem--the-most-important-concept)
5. [Data Ingestion — How the Data is Split](#5-data-ingestion--how-the-data-is-split)
6. [Feature Engineering — Building Pre-Match Features](#6-feature-engineering--building-pre-match-features)
7. [The 18 Features Explained](#7-the-18-features-explained)
8. [Model Training](#8-model-training)
9. [Why These Evaluation Metrics?](#9-why-these-evaluation-metrics)
10. [MLflow — Experiment Tracking](#10-mlflow--experiment-tracking)
11. [Prediction Pipeline — How Inference Works](#11-prediction-pipeline--how-inference-works)
12. [FastAPI — Serving the Model](#12-fastapi--serving-the-model)
13. [Streamlit — The Frontend](#13-streamlit--the-frontend)
14. [Project File Structure](#14-project-file-structure)
15. [How to Run the Project](#15-how-to-run-the-project)
16. [Interview Q&A Cheat Sheet](#16-interview-qa-cheat-sheet)

---

## 1. Project in One Sentence

> Given two Premier League teams, a home/away assignment, and a match date, predict whether the match will end in a Home Win, Draw, or Away Win — using only statistics from matches that have already been played.

This is a **3-class classification problem**. The three output classes are:

| Code | Meaning   | ~Frequency |
|------|-----------|-----------|
| `H`  | Home Win  | 43%       |
| `D`  | Draw      | 27%       |
| `A`  | Away Win  | 30%       |

---

## 2. The Core Problem

### Why not just train on the raw match stats?

The raw dataset contains columns like shots, shots on target, fouls, corners, and yellow cards. These are extremely predictive — if you know that Team A had 18 shots on target and Team B had 3, you can guess the winner almost perfectly.

**The problem: you don't know these numbers before the match starts.** They are recorded *during* the match.

This is called **data leakage** — using information in your model that would not be available at the time you actually need to make a prediction. A model trained with these features would look great on your validation set but would be completely useless in production, because you cannot feed it pre-match.

**The solution: throw away all in-match statistics and engineer new features from historical match data instead.** Everything the model sees must describe what is known before kickoff.

---

## 3. Dataset

- **Source:** Football-Data.co.uk (10 EPL seasons, 2016–17 to 2025–26)
- **Size:** ~3,800 matches (380 matches × 10 seasons)
- **Teams:** 34 unique clubs (teams are promoted and relegated each season)
- **Raw columns:** Date, HomeTeam, AwayTeam, Goals (FT + HT), Result (FT + HT), Referee, Shots, Shots on Target, Fouls, Corners, Yellow Cards, Red Cards

### Columns that are DROPPED (leaky — only known during/after the match)

| Column | Why dropped |
|--------|-------------|
| `half_time_home_goals` | Only known at half-time |
| `half_time_away_goals` | Only known at half-time |
| `half_time_result` | Only known at half-time |
| `home_team_shots` | Only known after the match |
| `away_team_shots` | Only known after the match |
| `home_team_shots_on_target` | Only known after the match |
| `away_team_shots_on_target` | Only known after the match |
| `home_team_fouls_committed` | Only known after the match |
| `away_team_fouls_committed` | Only known after the match |
| `home_team_corners` | Only known after the match |
| `away_team_corners` | Only known after the match |
| `home_team_yellow_cards` | Only known after the match |
| `away_team_yellow_cards` | Only known after the match |
| `home_team_red_cards` | Only known after the match |
| `away_team_red_cards` | Only known after the match |
| `full_time_home_goals` | This IS the target — cannot use as a feature |
| `full_time_away_goals` | Directly encodes the target |
| `home_points` / `away_points` | Directly encodes the result |

**Target column:** `full_time_result` → encoded as `H=0, D=1, A=2`

---

## 4. The Leakage Problem — The Most Important Concept

This is the single most important concept in this project. You will almost certainly be asked about it.

### What is data leakage?

Data leakage happens when your model is trained using information it would not have access to at prediction time. The model learns a shortcut that works on historical data but fails in production.

### Example of leakage in this project

Imagine you kept `home_team_shots_on_target` as a feature. During training, the model learns: "when the home team has 10+ shots on target, they almost always win." This is true — but it is circular reasoning. The shots on target happen *because* a team is winning, and you only know the number *after* the match. At prediction time (before kickoff), you have no idea how many shots either team will have.

A model with this feature would achieve maybe 85% accuracy in testing but 33% accuracy in production (barely better than random guessing for 3 classes).

### How this project avoids leakage

Every feature in this model is computed from matches that were played **before** the match being predicted. The feature engineering loop processes matches in strict chronological order, and for each match, it reads the team's history **before** updating it with the current match's result.

```
Match 1 → read history (empty) → predict → update history with Match 1 result
Match 2 → read history (Match 1 only) → predict → update history with Match 2 result
Match 3 → read history (Match 1, 2 only) → predict → update history with Match 3 result
...
```

This is called **point-in-time correctness**.

### The half-time leakage trap

Even half-time statistics are leaky for a pre-match model. In the data, when a team leads at half-time, they go on to win ~74% of the time. If you use `half_time_result` as a feature, your model will perform very well — but it is useless for pre-match prediction. This is why all half-time columns are dropped.

---

## 5. Data Ingestion — How the Data is Split

### Why not use `train_test_split(shuffle=True)`?

In a typical ML project (like predicting house prices or student scores), the data has no time dimension. Rows are independent of each other, so shuffling and random splitting is fine.

Football matches are **not independent**. A team's performance in March 2024 depends on their form from February 2024. If you randomly split the data:

- A March 2024 match might end up in your training set
- A January 2024 match might end up in your test set
- The model sees "future" data during training

This artificially inflates validation accuracy and means the model will underperform when deployed to predict genuinely future matches.

### The correct approach: Chronological splitting by season

```
Seasons 1–8  →  Training set    (~3,040 matches)
Season 9     →  Validation set  (~380 matches)   ← used for hyperparameter tuning
Season 10    →  Test set        (~380 matches)   ← touched only once, at the very end
```

The season boundary is the correct split unit because:
- It keeps each season intact (no mid-season splits)
- It mirrors how the model will actually be used (train on past seasons, predict the current season)
- It is human-interpretable ("I trained on 2016–2024, validated on 2024–25, and tested on 2025–26")

### Why a 3-way split (train/val/test) instead of 2-way?

The validation set is used to compare models and tune hyperparameters during development. If you use the test set for this, you are effectively training on the test set (you keep adjusting the model until it performs well on it). The test set must be **untouched** until you have a final model — it represents the true, honest measure of performance.

---

## 6. Feature Engineering — Building Pre-Match Features

This is the `DataTransformation` component. It is the most technically complex part of the project.

### The key constraint

Every feature must be computed using only matches that occurred **before** the match being predicted, for the specific team in question.

### How the engineering loop works

```python
for each match in chronological order:
    1. SNAPSHOT: read each team's current state (Elo, form, H2H, last date)
    2. BUILD: compute the 18 features from the snapshot
    3. APPEND: save the feature row
    4. UPDATE: update both teams' history with this match's result
```

Steps 1–3 happen **before** step 4. This is the key to zero leakage.

### Why features must be computed on the full dataset before splitting

The rolling features for the validation season's first match (e.g., Matchweek 1 of 2024–25) depend on every match played in the 2016–2024 period. If you computed features separately per split, the validation set's early rows would have empty history windows (their context is in the training set file).

**The correct order:**
1. Concatenate all three splits back into one sorted DataFrame
2. Compute features across the full sequence (each row only looks backward)
3. Re-split using the same season boundaries

---

## 7. The 18 Features Explained

All 18 features are computed per-match, per-team, using only prior matches. None require any in-match data.

### Group 1: Elo Ratings (3 features)

**What is Elo?** Elo is a rating system originally invented for chess. Every team starts at 1500. After each match:
- The winner gains points, the loser loses points
- The amount transferred depends on how surprising the result was
- Beating a much stronger team gains more points than beating a weaker one

**Formula:**
```
Expected score for home team:  E_home = 1 / (1 + 10^((Elo_away - Elo_home) / 400))
New Elo for home team:         Elo_home_new = Elo_home + K × (actual_score - E_home)
  where actual_score = 1.0 (win), 0.5 (draw), 0.0 (loss)
  and K = 20 (how fast ratings change)
```

**The 3 Elo features:**

| Feature | What it captures |
|---------|-----------------|
| `home_elo` | Home team's current strength rating |
| `away_elo` | Away team's current strength rating |
| `elo_diff` | `home_elo - away_elo` — the single most predictive feature. Positive = home team is stronger |

After 10 seasons, top clubs like Arsenal/Man City sit around 1600–1650. Relegated clubs hover around 1350–1400. A difference of +100 Elo points means a significant advantage.

### Group 2: Overall Rolling Form — Last 5 Matches (6 features)

For each team, look at their last 5 matches (regardless of home/away venue) and compute:

| Feature | What it captures |
|---------|-----------------|
| `home_form_pts` | Points (3=win, 1=draw, 0=loss) in last 5 matches. Max = 15. Captures recent momentum |
| `away_form_pts` | Same for the away team |
| `home_form_gf` | Goals scored in last 5 matches. Captures attacking form |
| `home_form_ga` | Goals conceded in last 5 matches. Captures defensive form |
| `away_form_gf` | Same for away team |
| `away_form_ga` | Same for away team |

**Why last 5 and not last 10?** 5 is the standard "form window" in football analytics — short enough to capture current form, long enough to smooth out a single lucky/unlucky result.

### Group 3: Venue-Specific Form — Last 5 Home/Away Games (4 features)

Some teams are very strong at home but poor travelers. This group captures that pattern:

| Feature | What it captures |
|---------|-----------------|
| `home_home_pts` | Home team's points in their last 5 **home** games only |
| `home_home_gf` | Home team's goals scored in their last 5 home games |
| `away_away_pts` | Away team's points in their last 5 **away** games only |
| `away_away_gf` | Away team's goals scored in their last 5 away games |

**Example:** A team might have `home_form_pts = 12` (great overall) but `away_away_pts = 3` (poor away form). The overall form hides the venue split.

### Group 4: Head-to-Head Record (3 features)

The last 5 meetings between these two specific teams:

| Feature | What it captures |
|---------|-----------------|
| `h2h_home_wins` | Times the current home team won in last 5 H2H meetings |
| `h2h_away_wins` | Times the current away team won in last 5 H2H meetings |
| `h2h_draws` | Draws in last 5 H2H meetings |

**Why H2H?** Some teams have psychological edges over specific opponents that persist even when their Elo ratings are similar. Classic example: certain teams consistently outperform their Elo predictions against specific rivals.

### Group 5: Rest Days (2 features)

| Feature | What it captures |
|---------|-----------------|
| `home_days_rest` | Days since the home team last played |
| `away_days_rest` | Days since the away team last played |

**Why rest days?** A team playing their third game in 7 days (common in cup competitions and congested schedules) is at a meaningful disadvantage — tired legs, higher injury risk, possible rotation. A team with 14 days rest is fresher.

Default value for the very first match of a team (no history): 30 days.

---

## 8. Model Training

### Phase 1: Model Comparison

Five classifiers are trained with default hyperparameters and compared on the validation set:

| Model | Why included |
|-------|-------------|
| **Logistic Regression** | Linear baseline. Fast, interpretable, good for establishing a floor |
| **Random Forest** | Ensemble of decision trees. Robust to outliers, handles non-linear relationships |
| **XGBoost** | Gradient boosting. Usually the best performer on tabular sports data |
| **LightGBM** | Faster gradient boosting alternative. Good on small datasets (3,800 matches is small) |
| **KNN** | Distance-based. Uses the scaled Elo/form features directly |

**Selection criterion:** Validation `f1_weighted` (explained below). The model with the highest `f1_weighted` on the validation season advances to Phase 2.

### Phase 2: Hyperparameter Tuning

`RandomizedSearchCV` is used on the Phase 1 winner, not `GridSearchCV`.

**Why Randomized and not Grid Search?**

Grid Search exhaustively tries every combination of hyperparameters. For XGBoost with 8 parameters and 5 options each, that's 5^8 = 390,625 combinations — it would run for days. RandomizedSearch samples 40 random combinations from the search space and finds roughly 90% of the benefit in 1% of the compute time.

**Why `StratifiedKFold` for cross-validation?**

Regular KFold could produce a fold where the Draw class (only 27% of data) barely appears. StratifiedKFold guarantees each fold has the same class proportions as the full training set. This is critical for minority-class performance.

### Class imbalance handling

All models that support it use `class_weight='balanced'`. This tells the loss function to weight each training sample inversely to its class frequency — the model is penalised more for getting a Draw wrong than for getting a Home Win wrong, because Draws are rarer. Without this, the model would learn to mostly predict Home Win and ignore Draws.

---

## 9. Why These Evaluation Metrics?

### Why not just accuracy?

A naive model that always predicts "Home Win" achieves **43% accuracy for free** — without learning anything. Accuracy alone is a misleading metric for imbalanced multi-class problems.

### F1 Score per class

F1 is the harmonic mean of Precision and Recall for each class:

```
Precision = True Positives / (True Positives + False Positives)
Recall    = True Positives / (True Positives + False Negatives)
F1        = 2 × (Precision × Recall) / (Precision + Recall)
```

Reporting F1 per class (Home Win, Draw, Away Win) exposes exactly where the model struggles. In practice, Draw always has the lowest F1 — it is the hardest outcome to predict because it depends on both teams being closely matched AND no decisive moment occurring.

### F1 Weighted (primary selection metric)

Takes a weighted average of per-class F1 scores, weighted by each class's support (number of actual samples). This is the metric used for model selection and hyperparameter tuning because it balances all three classes by their actual occurrence rate.

### Log Loss

Measures how confident and correct the model's probability predictions are. A model that predicts "Home Win: 51%, Draw: 25%, Away Win: 24%" and gets it right scores better than one that predicts "Home Win: 99%" and gets it right — overconfident predictions are penalised. This is important because the API returns probabilities, not just a class label.

---

## 10. MLflow — Experiment Tracking

MLflow tracks every training run so results are reproducible and comparable.

### Why MLflow?

Without experiment tracking, you train a model, get some numbers, change a hyperparameter, retrain, and have no record of what changed. After 5 experiments, you have no idea which setting produced which result. MLflow solves this by logging everything automatically.

### Structure of the tracking

Every time training runs, MLflow creates:

```
Parent run: "EPL_Full_Experiment_<timestamp>"
├── Child run: "Phase1_Logistic_Regression"   → params, metrics, fitted model
├── Child run: "Phase1_Random_Forest"         → params, metrics, fitted model
├── Child run: "Phase1_XGBoost"               → params, metrics, fitted model
├── Child run: "Phase1_LightGBM"              → params, metrics, fitted model
├── Child run: "Phase1_KNN"                   → params, metrics, fitted model
└── Child run: "Phase2_XGBoost_Tuned"         → best params, CV score, final metrics, model
```

### What gets logged per run

- **Parameters:** All model hyperparameters (e.g., `n_estimators=200, learning_rate=0.05`)
- **Metrics:** accuracy, f1_weighted, f1_macro, log_loss, f1_Home_Win, f1_Draw, f1_Away_Win — for both train and val splits
- **Model artifact:** The fitted model saved in its native format (XGBoost uses `mlflow.xgboost`, not sklearn, so feature importance is preserved)
- **Text artifacts:** Confusion matrix, full classification report

### Viewing results

```bash
mlflow ui --port 5001 --backend-store-uri sqlite:///mlflow.db
# Open: http://127.0.0.1:5001
```

The UI lets you compare all models side-by-side, filter by metric, and inspect every artifact from every run.

---

## 11. Prediction Pipeline — How Inference Works

This is what happens when a user submits "Arsenal vs Chelsea on 2026-08-16".

### The challenge

To predict a future match, you need the 18 features — Elo ratings, form, H2H, rest days — for both teams as of the match date. You cannot re-run the full 10-season feature engineering loop at prediction time (it takes minutes and the result would be identical every time).

### The solution: team_states.json

At the end of `DataTransformation`, a snapshot of every team's current state is saved to `artifacts/team_states.json`. It contains:

```json
{
  "teams": {
    "Arsenal": {
      "elo": 1642.3,
      "last_date": "2026-05-18",
      "recent": [last 5 matches with pts/gf/ga/venue],
      "recent_home": [last 5 home matches],
      "recent_away": [last 5 away matches]
    }
  },
  "h2h": {
    "Arsenal|||Chelsea": ["Arsenal_win", "draw", "Arsenal_win", "Chelsea_win", "Arsenal_win"]
  }
}
```

### The inference flow

```
User request: home_team="Arsenal", away_team="Chelsea", match_date="2026-08-16"
       ↓
Load team_states.json → read Arsenal's state, Chelsea's state
       ↓
Compute 18 features (same logic as training, same formula, same order)
       ↓
Build numpy array of shape (1, 18)
       ↓
Apply preprocessor.pkl (StandardScaler) → scale the features
       ↓
Apply model.pkl → predict class + predict_proba
       ↓
Return: { predicted_result: "H", probabilities: {home_win: 0.52, draw: 0.24, away_win: 0.24} }
```

### What if a team is not in team_states.json?

Newly promoted teams won't be in the file. The pipeline falls back to safe defaults:
- Elo → 1500 (league average)
- Form → all zeros (no history)
- Rest days → 14 days (typical mid-season gap)

This is disclosed in the API response as a warning.

### Artifact dependency

Three files must exist for prediction to work. All are created by the training pipeline:

| File | Created by | Contains |
|------|-----------|---------|
| `artifacts/model.pkl` | ModelTrainer | Fitted best classifier |
| `artifacts/preprocessor.pkl` | DataTransformation | Fitted StandardScaler |
| `artifacts/team_states.json` | DataTransformation | Team Elo + form snapshots |

---

## 12. FastAPI — Serving the Model

FastAPI exposes the prediction pipeline as an HTTP API.

### Why FastAPI over Flask?

- **Automatic documentation:** FastAPI generates Swagger UI (`/docs`) and ReDoc (`/redoc`) automatically from the Pydantic schema definitions — no extra work
- **Pydantic validation:** Request bodies are validated before they reach your code. An invalid date format or same team submitted twice returns a clear 422 error automatically
- **Performance:** FastAPI is built on Starlette (async) and is significantly faster than Flask for I/O-bound workloads
- **Type hints:** The entire codebase uses Python type hints, making the code self-documenting

### Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/` | Welcome message |
| `GET` | `/health` | Confirms model is loaded, returns team count and data freshness date |
| `GET` | `/teams?search=man` | Lists all known teams. Supports partial search |
| `POST` | `/predict` | Single match prediction |
| `POST` | `/predict/batch` | Up to 10 matches in one request |

### Input validation

```python
@field_validator('match_date')
def validate_date(cls, v):
    datetime.strptime(v, '%Y-%m-%d')   # raises 422 if format is wrong
    return v

@model_validator(mode='after')
def teams_must_differ(self):
    if self.home_team == self.away_team:
        raise ValueError("home_team and away_team must be different.")
    return self
```

### API response format

```json
{
  "home_team": "Arsenal",
  "away_team": "Chelsea",
  "match_date": "2026-08-16",
  "predicted_result": "H",
  "predicted_label": "Home Win",
  "probabilities": {
    "home_win": 0.52,
    "draw": 0.24,
    "away_win": 0.24
  },
  "confidence": 0.52,
  "elo_ratings": { "home": 1642.3, "away": 1558.7, "diff": 83.6 },
  "features_used": { "home_elo": 1642.3, "elo_diff": 83.6, ... }
}
```

---

## 13. Streamlit — The Frontend

The Streamlit app calls the FastAPI endpoints and renders the results visually. It does **not** import the ML model directly — all logic stays in the API layer.

### Three pages

**⚽ Predict Match**
- Two team dropdowns (populated from `/teams` endpoint)
- Date picker for the match date
- On submit: calls `POST /predict`, renders a colour-coded result card, probability bars, Elo comparison, and a collapsible feature breakdown

**📅 Match Week**
- Dynamic fixture builder — add up to 10 matches
- Fixtures stored in `st.session_state` (persists across Streamlit reruns)
- On submit: calls `POST /predict/batch`, renders all results as compact rows with mini probability bars

**📊 Power Rankings**
- Fetches Elo for all teams by calling `/predict` for each team vs a reference team
- Renders a ranked leaderboard with zone colouring: green (Top 4 Champions League), blue (Europa League), red (Bottom 3 relegation)

### Why Streamlit for the frontend?

- Zero HTML/CSS/JavaScript required for an interactive data app
- Fast prototyping — the entire frontend is ~780 lines of Python
- `@st.cache_data(ttl=300)` caches the team list for 5 minutes so the `/teams` endpoint isn't called on every UI interaction
- `st.session_state` persists fixture data across button clicks

---

## 14. Project File Structure

```
epl-predictor/
├── artifacts/                    # All trained model artifacts (git-ignored)
│   ├── model.pkl                 # Best trained classifier
│   ├── preprocessor.pkl          # Fitted StandardScaler
│   ├── team_states.json          # Team Elo + form snapshots for inference
│   ├── train.csv / val.csv / test.csv
│   ├── raw.csv
│   ├── featured_data.csv         # Full dataset with 18 engineered features
│   └── model_report.json         # All model metrics from training run
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py     # Chronological train/val/test split
│   │   ├── data_transformation.py # Feature engineering + preprocessing
│   │   └── model_trainer.py      # Model comparison + tuning + MLflow
│   ├── pipeline/
│   │   └── predict_pipeline.py   # Loads artifacts, builds features, predicts
│   ├── exception.py              # Custom exception with file + line info
│   ├── logger.py                 # Timestamped file + console logging
│   └── utils.py                  # save_object / load_object helpers
│
├── notebook/
│   ├── data/                     # Raw season CSV files
│   └── EDA.ipynb                 # Exploratory data analysis
│
├── app.py                        # FastAPI application (all endpoints)
├── streamlit_app.py              # Streamlit frontend (3 pages)
├── mlflow.db                     # MLflow experiment tracking database
├── requirements.txt
└── README.md
```

---

## 15. How to Run the Project

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the full training pipeline
```bash
python src/components/data_ingestion.py      # creates train/val/test CSVs
python src/components/data_transformation.py # creates featured_data, preprocessor, team_states
python src/components/model_trainer.py       # trains models, logs to MLflow, saves model.pkl
```

### Step 3: View experiment results
```bash
mlflow ui --port 5001 --backend-store-uri sqlite:///mlflow.db
# Open: http://127.0.0.1:5001
```

### Step 4: Start the API
```bash
uvicorn app:app --port 8000 --reload
# Swagger UI: http://127.0.0.1:8000/docs
```

### Step 5: Start the Streamlit frontend
```bash
streamlit run streamlit_app.py
# Opens: http://localhost:8501
```

---

## 16. Interview Q&A Cheat Sheet

### "Tell me about this project."

> I built a machine learning system that predicts English Premier League match outcomes — whether a match ends in a home win, draw, or away win. The core challenge was avoiding data leakage, since all the obvious features like shots and corners are only known after the match starts. So I engineered 18 pre-match features from historical data: Elo ratings that update after every result, rolling form over the last 5 matches, venue-specific performance splits, head-to-head records, and rest days. I trained and compared 5 classifiers, tracked everything with MLflow, and served the model through a FastAPI backend with a Streamlit frontend.

---

### "What is data leakage and how did you handle it?"

> Data leakage is when your model is trained on information it wouldn't have at prediction time. In this dataset, columns like shots on target are extremely predictive — but you only know them after the match is played. If you used them, the model would look great on historical data but fail completely in production. I handled it by dropping all in-match statistics and engineering features exclusively from prior matches, using a chronological loop that reads each team's history before updating it with the current match's result.

---

### "Why did you use F1 weighted instead of accuracy?"

> The class distribution is 43% home wins, 27% draws, 30% away wins. A naive model that always predicts "home win" gets 43% accuracy for free without learning anything. F1 weighted takes a weighted average of precision and recall per class, weighted by support, so the model is penalised for ignoring draws. I also report per-class F1 separately, which shows exactly how the model handles each outcome — in practice, draws always have the lowest F1 because they're the hardest to predict before the match.

---

### "Why did you split by season instead of randomly?"

> Football is a time series — a team's future performance is influenced by their recent history. A random split could put a March 2024 match in the training set and a January 2024 match in the test set, which means the model sees future data during training and the validation numbers are artificially inflated. Splitting by season mirrors how the model is actually used: train on past seasons, predict the current season.

---

### "What is Elo and why did you use it?"

> Elo is a rating system from chess. Every team starts at 1500, and after each match, points are transferred from the loser to the winner. The transfer amount depends on how surprising the result was — beating a much stronger team gains more points. After 10 seasons, the ratings reflect long-term team quality. The Elo difference between two teams turned out to be the single most predictive feature in the model, because it encodes historical matchup strength without leaking any in-match information.

---

### "Why RandomizedSearchCV over GridSearchCV?"

> Grid search tries every combination of hyperparameters. For XGBoost with 8 parameters and 5 options each, that's 390,625 combinations — computationally infeasible. Randomized search samples 40 random combinations from the space and finds roughly 90% of the performance gain in 1% of the compute time. For a project with a small dataset (3,800 matches) and limited compute, this is the right tradeoff.

---

### "How does the prediction work at inference time?"

> During training, a snapshot of every team's state — their Elo rating, last 5 match results, home/away form, head-to-head history — is saved to a JSON file. At inference time, the pipeline loads this snapshot, looks up both teams, computes the same 18 features using the same formulas as training, scales them with the saved preprocessor, and passes them to the model. The whole process takes milliseconds. If a team isn't in the snapshot (a newly promoted club), it falls back to league-average defaults and flags a warning in the API response.

---

### "Why FastAPI over Flask?"

> FastAPI auto-generates Swagger documentation from Pydantic schema definitions, has built-in request validation (bad date formats or same team submitted twice return clear errors automatically), and is significantly faster due to its async foundation. For a model serving API, type safety and automatic docs are high-value features that Flask requires significant extra work to replicate.

---

### "What would you improve if you had more time?"

> Several things. First, I'd add rolling xG (expected goals) as a feature — it's a much better measure of attacking/defensive quality than raw shot counts. Second, I'd replace the form window (last 5 matches) with a learned decay function that weights recent matches more. Third, I'd add transfer window effects — squad quality changes significantly in August and January. Fourth, I'd build a proper model registry in MLflow so I could promote specific runs to production rather than just using the latest training run. And finally, I'd add proper test coverage for the feature engineering logic, since that's where leakage bugs are most likely to hide.

---

*Built by Mahdi Islam Pranto · [GitHub](https://github.com/mahdi-islam-pranto)*
