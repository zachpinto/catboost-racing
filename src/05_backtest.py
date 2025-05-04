from pathlib import Path
import pandas as pd
import numpy as np
import joblib

DATA_FILE  = Path("../processed/horse_race_features.parquet")
MODEL_FILE = Path("../models/best_model.joblib")

TAKEOUT    = 0.18          # assumed track deduction
FLAT_STAKE = 1.0           # $1 flat bets
KELLY_BANK = 1000.0        # starting bank

THRESHOLDS = np.arange(0.00, 0.0601, 0.0025)   # 0% … 6% in 0.25 pp

df   = pd.read_parquet(DATA_FILE).sort_values("date").reset_index(drop=True)
pipe = joblib.load(MODEL_FILE)

y    = (df["position"] == 1).astype(int)
odds = df["decimalPrice"].astype(float)
X    = df.drop(columns=["position", "horseName", "jockeyName",
                        "trainerName", "rid", "date"])

# test slice (last 10 %)
n         = len(df)
val_end   = int(n * 0.9)
X_test    = X.iloc[val_end:]
y_test    = y.iloc[val_end:]
odds_test = odds.iloc[val_end:]

probs = pipe.predict_proba(X_test)[:, 1]
fair  = (1 - TAKEOUT) / odds_test
edge  = probs - fair

results = []
for thr in THRESHOLDS:
    mask   = edge >= thr
    bets   = mask.sum()
    profit = np.sum(
        FLAT_STAKE * (y_test[mask]*(odds_test[mask]-1) - (1 - y_test[mask]))
    )
    roi = profit / (bets*FLAT_STAKE) if bets else 0.0
    results.append((thr, bets, roi))

res_df = pd.DataFrame(results, columns=["threshold","bets","roi"])
best = res_df.loc[res_df["roi"].idxmax()]

print("=== Threshold sweep (flat‑stake ROI) ===")
print(res_df.to_string(index=False, formatters={"roi":"{:6.2%}".format}))

if best["roi"] <= 0:
    print("\nNo positive‑ROI threshold found in test slice.")
else:
    thr = best["threshold"]
    mask = edge >= thr
    print(f"\n>>> Optimal threshold: {thr:.3%}  "
          f"| bets={int(best['bets'])}  | ROI={best['roi']:.2%}")

    # ---- flat stake ----
    flat_profit = np.sum(
        FLAT_STAKE * (y_test[mask]*(odds_test[mask]-1) - (1 - y_test[mask]))
    )
    flat_roi = flat_profit / (mask.sum()*FLAT_STAKE)

    # ---- Kelly ----
    bank = KELLY_BANK
    peak = bank
    for p, o, w in zip(probs[mask], odds_test[mask], y_test[mask]):
        frac  = max(p - fair.loc[w.index[0]], 0) / (o - 1)
        stake = bank * frac
        bank += stake * (o - 1) if w else -stake
        peak  = max(peak, bank)
    kelly_roi = (bank - KELLY_BANK) / KELLY_BANK
    kelly_dd  = (peak - bank) / peak if peak else 0

    print(f"\nFlat‑stake strategy at optimal threshold:")
    print(f"  Net profit:  ${flat_profit:,.2f}")
    print(f"  ROI:         {flat_roi:.2%}")

    print(f"\nKelly strategy at optimal threshold:")
    print(f"  Final bank:  ${bank:,.2f}")
    print(f"  ROI:         {kelly_roi:.2%}")
    print(f"  Max drawdown:{kelly_dd:.2%}")
