from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import joblib
from catboost import CatBoostClassifier


DATA_FILE  = Path("../processed/horse_race_features.parquet")
MODEL_DIR  = Path("../models")
MODEL_DIR.mkdir(exist_ok=True)


df = pd.read_parquet(DATA_FILE)

# target: did the horse win?
y = (df["position"] == 1).astype(int)
X = df.drop(columns=["position", "horseName", "jockeyName",
                     "trainerName", "rid", "date"])  # drop IDs we won't use

# identify column types
categorical = ["ncond", "class"]                  # low‑cardinality cats
binary       = ["jockey_unknown", "trainer_unknown"]
numeric      = [col for col in X.columns
                if col not in categorical + binary]

# preprocessing
pre = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("bin", "passthrough", binary),
        ("num", StandardScaler(), numeric)
    ]
)

df_sorted = df.sort_values("date").reset_index(drop=True)
n = len(df_sorted)
train_end = int(n * 0.8)
val_end   = int(n * 0.9)

X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]

X_val   = X.iloc[train_end:val_end]
y_val   = y.iloc[train_end:val_end]

X_test  = X.iloc[val_end:]
y_test  = y.iloc[val_end:]


models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=3, n_jobs=-1
    ),
    "CatBoost": CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.05,
        loss_function="Logloss", verbose=False, random_seed=42
    )
}

results = {}

for name, clf in models.items():
    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)

    prob_val = pipe.predict_proba(X_val)[:, 1]
    prob_tst = pipe.predict_proba(X_test)[:, 1]

    results[name] = {
        "val_logloss": log_loss(y_val, prob_val),
        "val_auc":     roc_auc_score(y_val, prob_val),
        "test_logloss": log_loss(y_test, prob_tst),
        "test_auc":     roc_auc_score(y_test, prob_tst),
        "pipe": pipe
    }


best_name = min(results, key=lambda k: results[k]["val_logloss"])
best_pipe = results[best_name]["pipe"]

print("Validation results")
for n, r in results.items():
    print(f"{n:<13}  logloss={r['val_logloss']:.4f}  auc={r['val_auc']:.4f}")

print(f"\nBest model: {best_name}")
print(f"Test  logloss={results[best_name]['test_logloss']:.4f}  "
      f"auc={results[best_name]['test_auc']:.4f}")


out_path = MODEL_DIR / "best_model.joblib"
joblib.dump(best_pipe, out_path)
print(f"\nSaved best pipeline to {out_path}")
