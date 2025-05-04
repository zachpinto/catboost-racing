from pathlib import Path
import pandas as pd
import numpy as np


DATA_DIR = Path("../data")          # raw CSVs folder
OUT_DIR  = Path("../processed")     # output folder
OUT_DIR.mkdir(exist_ok=True)

race_files  = sorted(DATA_DIR.glob("races_*.csv"))
horse_files = sorted(DATA_DIR.glob("horses_*.csv"))

races_raw  = pd.concat([pd.read_csv(f, low_memory=False) for f in race_files],
                       ignore_index=True)
horses_raw = pd.concat([pd.read_csv(f, low_memory=False) for f in horse_files],
                       ignore_index=True)


races = races_raw[["rid", "date", "ncond", "metric", "class"]].copy()

races["date"]  = pd.to_datetime(races["date"], errors="coerce")
races          = races.dropna(subset=["date"])                # drop bad dates

races["ncond"] = races["ncond"].astype(str)
races["class"] = races["class"].astype(str)


horses = horses_raw[[
    "rid", "horseName", "jockeyName", "trainerName",
    "age", "weight", "saddle",
    "decimalPrice", "position"
]].copy()

horses["age"]          = pd.to_numeric(horses["age"], errors="coerce")
horses["weight"]       = pd.to_numeric(horses["weight"], errors="coerce")
horses["decimalPrice"] = pd.to_numeric(horses["decimalPrice"], errors="coerce")
horses["log_odds"]     = np.log(horses["decimalPrice"].replace(0, np.nan))


df = horses.merge(races, on="rid", how="left", validate="many_to_one")
df["field_size"] = df.groupby("rid")["rid"].transform("count")
df = df.sort_values(["horseName", "date"]).reset_index(drop=True)


OUT_FILE = OUT_DIR / "horse_race_merged.parquet"
df.to_parquet(OUT_FILE, index=False)
print(f"Saved {len(df):,} rows  ->  {OUT_FILE}")
