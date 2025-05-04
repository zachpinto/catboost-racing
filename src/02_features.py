from pathlib import Path
import pandas as pd
import numpy as np


IN_FILE  = Path("../processed/horse_race_merged.parquet")
OUT_FILE = Path("../processed/horse_race_features.parquet")

ROLL_WIN_BACK = 6
LAG_FINISHES  = 3
SMOOTH_ALPHA  = 10
GLOBAL_WIN    = 0.10


df = pd.read_parquet(IN_FILE).sort_values(["horseName", "date"]).reset_index(drop=True)


df["prev_date"] = df.groupby("horseName")["date"].shift(1)
df["days_since_last"] = (df["date"] - df["prev_date"]).dt.days.fillna(9999)

for k in range(1, LAG_FINISHES + 1):
    df[f"finish_lag{k}"] = df.groupby("horseName")["position"].shift(k).fillna(40)

def wins_last6(series):
    return series.shift(1).eq(1).rolling(ROLL_WIN_BACK, min_periods=1).sum()

df["wins_last6"] = df.groupby("horseName")["position"].transform(wins_last6)

df["course_runs"] = df.groupby(["horseName", "ncond"]).cumcount()

metric_clean = pd.to_numeric(df["metric"], errors="coerce").fillna(-1)
df["dist_band"] = (metric_clean // 200).astype(int)
df["dist_runs"] = df.groupby(["horseName", "dist_band"]).cumcount()


df["jockey_win_rate"]  = GLOBAL_WIN
df["jockey_unknown"]   = 1
df["trainer_win_rate"] = GLOBAL_WIN
df["trainer_unknown"]  = 1

def add_role_stats(group, role):
    idx = group.index
    starts = pd.Series(range(len(group)), index=idx)
    wins   = group["position"].shift(1).eq(1).cumsum()
    rate   = (wins + SMOOTH_ALPHA * GLOBAL_WIN) / (starts + SMOOTH_ALPHA)
    unknown = (starts == 0).astype(int)
    df.loc[idx, f"{role}_win_rate"] = rate
    df.loc[idx, f"{role}_unknown"]  = unknown

df.groupby("jockeyName", group_keys=False).apply(add_role_stats, role="jockey")
df.groupby("trainerName", group_keys=False).apply(add_role_stats, role="trainer")


df = df.drop(columns=["prev_date", "dist_band"])
df.to_parquet(OUT_FILE, index=False)
print(f"Feature dataset saved to {OUT_FILE}  ({len(df):,} rows)")
