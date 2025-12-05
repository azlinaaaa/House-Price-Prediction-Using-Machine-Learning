"""
Two-stage House Price Prediction:
1) RandomForest Classifier predicts Price Range (Low, Medium, High)
2) Regression per Price Range using RandomForest, XGBoost, LightGBM, CatBoost
   - Compare metrics (MAE, RMSE, R2)
   - Choose best model per range
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report, confusion_matrix, accuracy_score

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier

# =====================================
# Configurations
# =====================================
DATA_PATH = "cleaned_no_luxury.csv"
OUT_DIR = Path("two_stage_output")
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
TRIALS = 20  # Reduce for faster test

# =====================================
# Helper
# =====================================
def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# =====================================
# Load Data
# =====================================
df = pd.read_csv(DATA_PATH)

# Encode categorical features
label_encoders = {}
for col in CATEGORICAL_COLS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
joblib.dump(label_encoders, OUT_DIR / "label_encoders.joblib")

# =====================================
# 1) Classification Stage (Price Range)
# =====================================
df['PriceRange'] = pd.cut(df[TARGET], bins=BINS, labels=RANGE_LABELS).astype(str)
price_range_le = LabelEncoder()
df['PriceRange_encoded'] = price_range_le.fit_transform(df['PriceRange'])
joblib.dump(price_range_le, OUT_DIR / "price_range_encoder.joblib")
print("Price range mapping:", dict(zip(price_range_le.classes_, price_range_le.transform(price_range_le.classes_))))

# Train/test split
X_cls = df[FEATURES]
y_cls = df['PriceRange_encoded']
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, stratify=y_cls, test_size=0.2, random_state=RANDOM_STATE)

# Scale numeric
scaler = StandardScaler()
X_train_cls_scaled = scaler.fit_transform(X_train_cls)
X_test_cls_scaled = scaler.transform(X_test_cls)
joblib.dump(scaler, OUT_DIR / "scaler_cls.joblib")

# Optuna objective for RandomForest classifier
def objective_rf_cls(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 5, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "random_state": RANDOM_STATE
    }
    model = RandomForestClassifier(**params, n_jobs=-1)
    scores = cross_val_score(model, X_train_cls_scaled, y_train_cls, cv=3, scoring="accuracy")
    return np.mean(scores)

print("\n=== Tuning RandomForest Classifier ===")
study_cls = optuna.create_study(direction="maximize")
study_cls.optimize(objective_rf_cls, n_trials=TRIALS, show_progress_bar=True)
print("Best classifier params:", study_cls.best_params)

# Train classifier
rf_cls = RandomForestClassifier(**study_cls.best_params, n_jobs=-1, random_state=RANDOM_STATE)
rf_cls.fit(X_train_cls_scaled, y_train_cls)
joblib.dump(rf_cls, OUT_DIR / "rf_price_range_classifier.pkl")

# Evaluate classifier
y_pred_cls = rf_cls.predict(X_test_cls_scaled)
acc_cls = accuracy_score(y_test_cls, y_pred_cls)
print("\nClassification Accuracy:", acc_cls)
print(classification_report(price_range_le.inverse_transform(y_test_cls),
                            price_range_le.inverse_transform(y_pred_cls),
                            digits=4))

cm_cls = confusion_matrix(y_test_cls, y_pred_cls)
plt.figure(figsize=(6,5))
sns.heatmap(cm_cls, annot=True, fmt="d", cmap="Blues",
            xticklabels=price_range_le.classes_,
            yticklabels=price_range_le.classes_)
plt.title("Random Forest Price Range Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "rf_cls_confusion_matrix.png")
plt.show()

# =====================================
# 2) Regression Stage per Predicted Range
# =====================================
algorithms = {
    "RandomForest": RandomForestRegressor,
    "XGBoost": XGBRegressor,
    "LightGBM": LGBMRegressor,
    "CatBoost": CatBoostRegressor
}

reg_models = {}
reg_metrics = {}
reg_scalers = {}

for label in RANGE_LABELS:
    print(f"\n=== Regression for Price Range: {label} ===")
    subset = df.copy()
    # Use predicted price range from classifier
    subset = subset[price_range_le.inverse_transform(rf_cls.predict(scaler.transform(subset[FEATURES]))) == label]

    if len(subset) < 20:
        print("⚠️ Not enough samples; skipping.")
        continue

    q97 = subset[TARGET].quantile(0.97)
    subset = subset[subset[TARGET] < q97]

    X_sub = subset[FEATURES]
    y_sub = subset[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X_sub, y_sub, test_size=0.2, random_state=RANDOM_STATE)

    scaler_r = StandardScaler()
    X_train_scaled = scaler_r.fit_transform(X_train)
    X_test_scaled = scaler_r.transform(X_test)
    reg_scalers[label] = scaler_r

    reg_models[label] = {}
    reg_metrics[label] = {}

    for algo_name, AlgoClass in algorithms.items():
        print(f"--- Training {algo_name} ---")

        # Optuna objective for regression
        def objective_reg(trial):
            if algo_name == "RandomForest":
                n_estimators = trial.suggest_int("n_estimators", 100, 600)
                max_depth = trial.suggest_int("max_depth", 3, 25)
                min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
                min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 4)
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                )
            elif algo_name == "XGBoost":
                model = XGBRegressor(
                    n_estimators=trial.suggest_int("n_estimators",100,600),
                    max_depth=trial.suggest_int("max_depth",3,12),
                    learning_rate=trial.suggest_float("learning_rate",0.01,0.3),
                    subsample=trial.suggest_float("subsample",0.6,1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree",0.5,1.0),
                    random_state=RANDOM_STATE,
                    n_jobs=1
                )
            elif algo_name == "LightGBM":
                model = LGBMRegressor(
                    n_estimators=trial.suggest_int("n_estimators",100,600),
                    max_depth=trial.suggest_int("max_depth",3,15),
                    learning_rate=trial.suggest_float("learning_rate",0.01,0.3),
                    num_leaves=trial.suggest_int("num_leaves",20,100),
                    subsample=trial.suggest_float("subsample",0.6,1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree",0.5,1.0),
                    random_state=RANDOM_STATE
                )
            else:  # CatBoost
                model = CatBoostRegressor(
                    iterations=trial.suggest_int("iterations",100,600),
                    depth=trial.suggest_int("depth",3,10),
                    learning_rate=trial.suggest_float("learning_rate",0.01,0.3),
                    l2_leaf_reg=trial.suggest_float("l2_leaf_reg",1,10),
                    random_state=RANDOM_STATE,
                    verbose=0
                )
            scores = cross_val_score(model, X_train_scaled, y_train, cv=3,
                                     scoring="neg_root_mean_squared_error", n_jobs=1)
            return float(np.mean(scores))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective_reg, n_trials=TRIALS, show_progress_bar=True)
        best_params = study.best_params

        if algo_name == "CatBoost":
            model_final = AlgoClass(**best_params, random_state=RANDOM_STATE, verbose=0)
        else:
            model_final = AlgoClass(**best_params, random_state=RANDOM_STATE)
        model_final.fit(X_train_scaled, y_train)
        y_pred = model_final.predict(X_test_scaled)

        # Metrics
        mae_t = mean_absolute_error(y_test, y_pred)
        rmse_t = rmse(y_test, y_pred)
        r2_t = r2_score(y_test, y_pred)
        train_r2 = r2_score(y_train, model_final.predict(X_train_scaled))

        reg_models[label][algo_name] = model_final
        reg_metrics[label][algo_name] = {
            "mae_test": mae_t,
            "rmse_test": rmse_t,
            "r2_test": r2_t,
            "r2_train": train_r2
        }

        # Save model
        joblib.dump(model_final, OUT_DIR / f"{algo_name}_{label}_best.joblib")
        print(f"{algo_name} | RMSE={rmse_t:.2f}, R2_test={r2_t:.4f}, R2_train={train_r2:.4f}, MAE={mae_t:.2f}")

# =====================================
# Compare Regression Metrics
# =====================================
metrics_list = []
for label in reg_metrics:
    for algo in reg_metrics[label]:
        m = reg_metrics[label][algo]
        metrics_list.append({
            "PriceRange": label,
            "Model": algo,
            "MAE": m["mae_test"],
            "RMSE": m["rmse_test"],
            "R2_train": m["r2_train"],
            "R2_test": m["r2_test"]
        })
df_comparison = pd.DataFrame(metrics_list).sort_values(["PriceRange","RMSE"])
df_comparison.to_csv(OUT_DIR / "regression_comparison_metrics.csv", index=False)
print("\n✅ Regression comparison metrics saved.")
print(df_comparison)
