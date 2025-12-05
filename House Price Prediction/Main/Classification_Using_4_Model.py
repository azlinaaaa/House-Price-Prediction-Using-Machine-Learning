"""
Optuna-tuned Classifiers for House Price Range Prediction
Models: XGBoost, Random Forest, LightGBM, CatBoost
- Classifies properties into Low, Medium, High price categories
- Compares confusion matrices across four models
- Saves best overall classifier as .pkl
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
import optuna
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

# =====================================
# Configuration
# =====================================
DATA_PATH = "cleaned_no_luxury.csv"
OUT_DIR = Path("classification_comparison_output")
OUT_DIR.mkdir(exist_ok=True)
PLOTS_DIR = OUT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

FEATURES = [
    'Location', 'Price per SqFt', 'Bedroom', 'Bathroom', 'Carpark',
    'Built-up', 'Furnishing', 'Property Type', 'State'
]
CATEGORICAL_COLS = ['Location', 'Furnishing', 'Property Type', 'State']
TARGET = 'Price'

BINS = [0, 300000, 800000, np.inf]
RANGE_LABELS = ['Low', 'Medium', 'High']
RANDOM_STATE = 42
TRIALS = 30

# =====================================
# Helper
# =====================================
def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

# =====================================
# 1) Load and Preprocess Data
# =====================================
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Encode categorical
label_encoders = {}
for col in CATEGORICAL_COLS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Target binning
df['PriceRange'] = pd.cut(df[TARGET], bins=BINS, labels=RANGE_LABELS).astype(str)
price_range_le = LabelEncoder()
df['PriceRange_encoded'] = price_range_le.fit_transform(df['PriceRange'])
print("Price range mapping:", dict(zip(price_range_le.classes_, price_range_le.transform(price_range_le.classes_))))

# Split data
X = df[FEATURES]
y = df['PriceRange_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=RANDOM_STATE)

# Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(label_encoders, OUT_DIR / "label_encoders.joblib")
joblib.dump(price_range_le, OUT_DIR / "price_range_encoder.joblib")
joblib.dump(scaler, OUT_DIR / "scaler.joblib")

# =====================================
# 2) Define Optuna Objectives
# =====================================
def objective_xgb(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        "random_state": RANDOM_STATE,
        "use_label_encoder": False,
        "objective": "multi:softprob",
        "eval_metric": "mlogloss"
    }
    model = XGBClassifier(**params)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring="accuracy")
    return np.mean(scores)

def objective_rf(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 5, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "random_state": RANDOM_STATE
    }
    model = RandomForestClassifier(**params, n_jobs=-1)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring="accuracy")
    return np.mean(scores)

def objective_lgbm(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "random_state": RANDOM_STATE
    }
    model = LGBMClassifier(**params)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring="accuracy")
    return np.mean(scores)

def objective_cat(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 100, 800),
        "depth": trial.suggest_int("depth", 3, 10),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "random_state": RANDOM_STATE,
        "verbose": 0
    }
    model = CatBoostClassifier(**params)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring="accuracy")
    return np.mean(scores)

# =====================================
# 3) Tune, Train and Select Best Model
# =====================================
models = {}
studies = {}
accuracy_scores = {}

for name, objective in {
    "XGBoost": objective_xgb,
    "RandomForest": objective_rf,
    "LightGBM": objective_lgbm,
    "CatBoost": objective_cat
}.items():
    print(f"\n=== Tuning {name} ===")
    study = optuna.create_study(direction="maximize", study_name=f"{name}_study")
    study.optimize(objective, n_trials=TRIALS, show_progress_bar=True)
    studies[name] = study
    print(f"Best params for {name}: {study.best_params}")

    # Train model with best params
    if name == "XGBoost":
        model = XGBClassifier(**study.best_params, objective="multi:softprob", use_label_encoder=False, random_state=RANDOM_STATE)
    elif name == "RandomForest":
        model = RandomForestClassifier(**study.best_params, n_jobs=-1, random_state=RANDOM_STATE)
    elif name == "LightGBM":
        model = LGBMClassifier(**study.best_params, random_state=RANDOM_STATE)
    else:
        model = CatBoostClassifier(**study.best_params, random_state=RANDOM_STATE, verbose=0)

    model.fit(X_train_scaled, y_train)
    models[name] = model
    joblib.dump(model, OUT_DIR / f"{name.lower()}_classifier.joblib")

    # Save accuracy for selecting best
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    accuracy_scores[name] = acc

# Select best overall classifier
best_model_name = max(accuracy_scores, key=accuracy_scores.get)
best_model = models[best_model_name]
save_pickle(best_model, OUT_DIR / "best_classifier.pkl")
print(f"\n🏆 Best classifier: {best_model_name} with accuracy {accuracy_scores[best_model_name]:.4f}")

# =====================================
# 4) Evaluation and Confusion Matrix
# =====================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, (name, model) in enumerate(models.items()):
    y_pred = model.predict(X_test_scaled)
    print(f"\n=== {name} Classification Report ===")
    print(classification_report(
        price_range_le.inverse_transform(y_test),
        price_range_le.inverse_transform(y_pred),
        digits=4
    ))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=price_range_le.classes_,
                yticklabels=price_range_le.classes_,
                ax=axes[i])
    axes[i].set_title(f"{name} Confusion Matrix")
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("Actual")

plt.tight_layout()
plt.savefig(PLOTS_DIR / "comparison_confusion_matrices.png")
plt.show()

# =====================================
# 5) Save Optuna Studies
# =====================================
for name, study in studies.items():
    save_pickle(study, OUT_DIR / f"{name.lower()}_study.pkl")

print(f"\n✅ All models, best classifier, and plots saved in: {OUT_DIR.resolve()}")
