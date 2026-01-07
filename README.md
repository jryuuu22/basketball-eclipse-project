# Basketball Eclipse Analytics  
## Free Throw Pressure Modeling

**Authors:** Jay Wu, Alay Nawab  
**Environment:** Google Colab  
**Primary Language:** Python  

This project builds an end-to-end machine learning system to model **free throw success under pressure** using NBA play-by-play data. The goal is to quantify how contextual pressure and player skill affect free-throw outcomes, compare multiple models, and deliver a reusable, deployment-ready prediction module.

---

## Project Goals

- Clean and preprocess raw NBA play-by-play data  
- Perform clear, interpretable exploratory data analysis (EDA)  
- Engineer player-skill and game-pressure features  
- Compare **Logistic Regression**, **Random Forest**, and **XGBoost** using **Optuna**  
- Optimize models for **PR-AUC** (precision–recall focus due to class imbalance)  
- Evaluate performance on a **chronological test split** to prevent leakage  
- Produce clear visualizations and save all artifacts  
- Deliver a production-style module (`ft_pressure_final.py`) with a reusable predictor class  

---

## Dataset

**Source:** Kaggle – Basketball Play-by-Play Dataset  
- Event-level NBA play-by-play data  
- Includes timestamps, score state, player actions, and game context  
- Free-throw events are extracted and modeled as binary outcomes (make vs miss)

---

## Feature Engineering

From raw play-by-play data, the pipeline constructs a modeling dataset with:

### Player Skill Features
- `season_FT_pct`
- `overall_ft_pct`
- `clutch_ft_pct`
- `clutch_factor` (clutch FT% − overall FT%)
- `career_attempts_so_far`

### Game Context / Pressure Features
- `period`
- `seconds_remaining`
- `point_differential`
- `is_clutch` (last 5 minutes of 4th+ period)
- `close_game` (≤ 3 point margin)
- `late_game` (4th quarter or later)
- `pressure_score` (composite pressure signal)

The final modeling frame contains **12 engineered features**, a binary target (`FT_made`), and `game_id` for chronological splitting.

---

## Exploratory Data Analysis (EDA)

Key analyses include:
- Free throw percentage in clutch vs non-clutch situations  
- FT% by game period  
- FT% across player skill tiers (quartiles of season FT%)  
- FT% as a function of increasing pressure score  

EDA confirms strong separation by **player skill**, with subtler but measurable effects from **pressure context**.

---

## Modeling Pipeline

### Train / Test Split
- **Chronological split by game_id** (no shuffling)
- Prevents future-game leakage into training

### Preprocessing
- Median imputation for missing values  
- Feature standardization via `StandardScaler`  
- Preprocessing artifacts saved for reuse

### Models Evaluated
- Logistic Regression  
- Random Forest Classifier  
- XGBoost Classifier  

### Hyperparameter Tuning
- **Optuna** used for each model (~30 trials each)
- Optimization metric: **PR-AUC**

### Winning Model
- **XGBoost** achieved the best validation PR-AUC  
- Retrained on full training data before final test evaluation

---

## Test Set Performance (XGBoost)

- **Accuracy:** ~0.75  
- **PR-AUC:** ~0.84  
- **ROC-AUC:** ~0.64  
- **Brier Score:** ~0.18  

Performance reflects the dominance of player skill while retaining meaningful signal from pressure-related features.

---

## Feature Importance

Tree-based feature importance highlights:
- Player skill metrics (season and overall FT%)  
- Clutch indicators and late-game context  
- Pressure-related composite features  

These align with basketball intuition while remaining data-driven.

---

## Deployment Module (`ft_pressure_final.py`)

The final module provides a clean, reusable API:

### Core Components
- `prepare_ft_dataset_from_df`  
  Builds the modeling dataset from raw play-by-play data  

- `train_and_save_model`  
  Trains a model with a chronological split and saves all artifacts  

- `FTPressurePredictor`  
  - Load trained model, imputer, and scaler  
  - Predict probabilities and binary outcomes  
  - Provide simple per-shot explanations  
  - Expose global feature importance  

- `evaluate_model`  
  Computes accuracy, PR-AUC, ROC-AUC, Brier score, and confusion matrix  

---

## Example Usage

```python
from ft_pressure_final import FTPressurePredictor

predictor = FTPressurePredictor()

shot = {
    "season_FT_pct": 0.80,
    "overall_ft_pct": 0.78,
    "clutch_ft_pct": 0.85,
    "clutch_factor": 0.07,
    "career_attempts_so_far": 100,
    "period": 4,
    "seconds_remaining": 45,
    "is_clutch": 1,
    "close_game": 1,
    "late_game": 1,
    "pressure_score": 3,
    "point_differential": 2,
}

prediction = predictor.predict_single_with_explanation(shot)
