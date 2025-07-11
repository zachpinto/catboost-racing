import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import math

# Helper functions
def build_feature_rows(track: dict, horses: list[dict]) -> pd.DataFrame:
    rows = []
    for h in horses:
        dec_odds = max(h["odds"], 1.01)            # protect against ≤1
        rows.append({
            # race‑level
            "ncond": track["surface"],
            "class": track["race_class"],
            "metric": track["distance"],
            "field_size": len(horses),

            # raw PP fields
            "age": h["age"],
            "weight": h["weight"],
            "saddle": h["post"],
            "decimalPrice": dec_odds,
            "log_odds": math.log(dec_odds),

            # engineered inputs
            "days_since_last": h["days_since_last"],
            "finish_lag1": h["finish_lag1"],
            "finish_lag2": h["finish_lag2"],
            "finish_lag3": h["finish_lag3"],
            "wins_last6":   h["wins_last6"],
            "course_runs":  h["course_runs"],
            "dist_runs":    h["dist_runs"],
            "jockey_win_rate":  h["jockey_win_rate"],
            "jockey_unknown":   h["jockey_unknown"],
            "trainer_win_rate": h["trainer_win_rate"],
            "trainer_unknown":  h["trainer_unknown"],
        })
    return pd.DataFrame(rows)

# Load model
MODEL_PATH = Path("models/best_model_calibrated.joblib")
if not MODEL_PATH.exists():
    MODEL_PATH = Path("models/best_model.joblib")

from sklearn.compose import _column_transformer

if not hasattr(_column_transformer, "_RemainderColsList"):
    class _RemainderColsList(list):
        """Dummy placeholder needed only to unpickle old ColumnTransformer objects."""
        pass

    # register the dummy on the module so pickle can find it
    _column_transformer._RemainderColsList = _RemainderColsList
    
pipe = joblib.load(MODEL_PATH)

# Collect inputs
st.title("Horse‑race Win Probability Model")

with st.form("race_form"):
    st.header("Race information")
    surface     = st.selectbox("Surface", ["DIRT", "TURF", "SYNTHETIC"])
    distance    = st.number_input("Distance (meters)", 800, 4000, 1600, 100)
    race_class  = st.text_input("Class (e.g. G1/HCP)", value="UNK")
    n_runners   = st.number_input("Field size (Press Enter to confirm)", 2, 20, 8)

    st.header("Horse entries")
    horses = []
    for i in range(1, n_runners + 1):
        with st.expander(f"Horse {i}", expanded=(i == 1)):
            age    = st.number_input("Age", 2, 15, 4, key=f"age{i}")
            weight = st.number_input("Weight kg", 40.0, 70.0, 55.0, key=f"wt{i}")
            odds   = st.number_input("Morning‑line odds (decimal)", 1.01, 99.9, 5.0, key=f"odds{i}")

            st.markdown("*Recent form — use 40 for DNF*")
            f1   = st.number_input("Last finish", 1, 40, 3, key=f"f1{i}")
            f2   = st.number_input("Two back",    1, 40, 5, key=f"f2{i}")
            f3   = st.number_input("Three back",  1, 40, 2, key=f"f3{i}")
            wins6= st.number_input("Wins last 6", 0, 6, 1, key=f"w6{i}")
            dsl  = st.number_input("Days since last", 1, 400, 30, key=f"dsl{i}")

            course_runs = st.number_input("Runs on this surface", 0, 50, 3, key=f"cr{i}")
            dist_runs   = st.number_input("Runs at this dist",    0, 50, 4, key=f"dr{i}")

            j_wr = st.number_input("Jockey win‑rate (0‑1)", 0.0, 1.0, 0.10, 0.01, key=f"jwr{i}")
            t_wr = st.number_input("Trainer win‑rate (0‑1)", 0.0, 1.0, 0.12, 0.01, key=f"twr{i}")
            j_new = st.checkbox("Unknown jockey", key=f"jun{i}")
            t_new = st.checkbox("Unknown trainer", key=f"tun{i}")

        horses.append(dict(age=age, weight=weight, post=i, odds=odds,
                           finish_lag1=f1, finish_lag2=f2, finish_lag3=f3,
                           wins_last6=wins6, days_since_last=dsl,
                           course_runs=course_runs, dist_runs=dist_runs,
                           jockey_win_rate=j_wr, trainer_win_rate=t_wr,
                           jockey_unknown=int(j_new), trainer_unknown=int(t_new)))

    submitted = st.form_submit_button("Predict")

# Prediction & display
if submitted:
    track = dict(surface=surface, distance=distance, race_class=race_class)
    X_new = build_feature_rows(track, horses)
    probs = pipe.predict_proba(X_new)[:, 1]

    X_new["Horse"] = [f"H{i+1}" for i in range(len(horses))]
    X_new["Prob"]  = probs

    st.subheader("Win probabilities")
    st.dataframe(
        X_new[["Horse", "decimalPrice", "Prob"]]
          .sort_values("Prob", ascending=False)
          .style.format({"decimalPrice": "{:.2f}", "Prob": "{:.1%}"})
    )

    st.subheader("Top 3 horses")
    top3 = X_new.nlargest(3, "Prob")
    for row in top3.itertuples(index=False):
        st.write(f"**{row.Horse}** — {row.Prob:.1%}")
