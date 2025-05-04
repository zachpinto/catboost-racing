from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

DATA_FILE  = Path("../processed/horse_race_features.parquet")
MODEL_FILE = Path("../models/best_model.joblib")
OUT_DIR    = Path("../reports")
OUT_DIR.mkdir(exist_ok=True)

# 1. Load data & model
df   = pd.read_parquet(DATA_FILE).sort_values("date").reset_index(drop=True)
pipe = joblib.load(MODEL_FILE)

# features / target
y = (df["position"] == 1).astype(int)
X = df.drop(columns=["position", "horseName", "jockeyName",
                     "trainerName", "rid", "date"])

# split data (80/10/10)
n         = len(df)
train_end = int(n * 0.8)
val_end   = int(n * 0.9)

X_test = X.iloc[val_end:]
y_test = y.iloc[val_end:]

# 2. Predict probabilities
probs = pipe.predict_proba(X_test)[:, 1]
preds = (probs >= 0.5).astype(int)     # threshold for CM only

# 3. ROC curve
fpr, tpr, _ = roc_curve(y_test, probs)
roc_auc     = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Test Set")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(OUT_DIR / "roc_curve.png")
plt.close()

# 4. Precision‑Recall curve
prec, rec, _ = precision_recall_curve(y_test, probs)
pr_auc       = auc(rec, prec)

plt.figure()
plt.plot(rec, prec, label=f"PR AUC = {pr_auc:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision‑Recall Curve – Test Set")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig(OUT_DIR / "pr_curve.png")
plt.close()

# 5. Calibration curve
prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10)

plt.figure()
plt.plot(prob_pred, prob_true, marker="o")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("Predicted probability")
plt.ylabel("Observed win rate")
plt.title("Calibration – Test Set")
plt.tight_layout()
plt.savefig(OUT_DIR / "calibration_curve.png")
plt.close()

# 6. Confusion matrix
cm = confusion_matrix(y_test, preds, labels=[1, 0])
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["Win", "Not"])
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix – Threshold 0.5")
plt.tight_layout()
plt.savefig(OUT_DIR / "confusion_matrix.png")
plt.close()

print("Diagnostics saved in ./reports/")
