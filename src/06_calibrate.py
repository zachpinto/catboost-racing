from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.calibration import CalibratedClassifierCV

# ------------------------------------------------------------------
# CONSTANTS
TAKEOUT     = 0.18
FLAT_STAKE  = 1.0
KELLY_BANK  = 1000.0
THRESHOLDS  = np.arange(0.00, 0.0601, 0.0025)
DATA_FILE  = Path("../processed/horse_race_features.parquet")
MODEL_FILE = Path("../models/best_model.joblib")
CAL_FILE   = Path("../models/best_model_calibrated.joblib")

# 1. Load data & split
df = pd.read_parquet(DATA_FILE).sort_values("date").reset_index(drop=True)
y  = (df["position"] == 1).astype(int)
X  = df.drop(columns=["position", "horseName", "jockeyName",
                      "trainerName", "rid", "date"])

n        = len(df)
train_end= int(n*0.8)
val_end  = int(n*0.9)

X_trainval = X.iloc[:val_end]
y_trainval = y.iloc[:val_end]
X_test     = X.iloc[val_end:]
y_test     = y.iloc[val_end:]
odds_test  = df["decimalPrice"].astype(float).iloc[val_end:]

# 2. Calibrate
base_pipe = joblib.load(MODEL_FILE)
cal_pipe  = CalibratedClassifierCV(base_pipe, cv="prefit", method="isotonic")
cal_pipe.fit(X_trainval, y_trainval)
joblib.dump(cal_pipe, CAL_FILE)
print("✅ Calibrated model saved ->", CAL_FILE)

# 3. Predictions & edge
probs = cal_pipe.predict_proba(X_test)[:, 1]
fair  = (1 - TAKEOUT) / odds_test
edge  = probs - fair
wins  = y_test.values

def flat_roi(mask):
    profit = np.sum(
        FLAT_STAKE * (wins[mask] * (odds_test[mask] - 1) - (1 - wins[mask]))
    )
    return profit / (mask.sum() * FLAT_STAKE) if mask.sum() else 0.0

# 4. Threshold sweep
print("\n=== After isotonic calibration ===")
best_roi, best_thr, best_bets = -1, None, 0
for thr in THRESHOLDS:
    m   = edge >= thr
    roi = flat_roi(m)
    if roi > best_roi:
        best_roi, best_thr, best_bets = roi, thr, m.sum()
    print(f"thr ≥ {thr:5.3f}  |  bets={m.sum():4d}  |  ROI={roi:6.2%}")

# 5. Report best threshold
if best_roi > 0:
    print(f"\n🎉 Positive ROI at threshold {best_thr:.3%}: "
          f"{best_roi:.2%} on {best_bets} bets")
else:
    print("\nStill negative ROI; need richer features or exotic‑pool odds.")
