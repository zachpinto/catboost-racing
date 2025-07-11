# app.py  ──────────────────────────────────────────────────────────────────────
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# 0.  Make the pickled model readable by the older scikit-learn runtime
#     (Streamlit Cloud ships sklearn-1.4.x for Python-3.12).
#     • First let joblib unpickle: provide the legacy helper class
#     • Then walk through the entire object graph and add any missing
#       attributes that newer versions create during fit.
# ──────────────────────────────────────────────────────────────────────────────
from sklearn.compose import _column_transformer as _ct_mod
from sklearn.compose import ColumnTransformer


# ---- a) legacy helper that 1.6 created during fit ---------------------------
if not hasattr(_ct_mod, "_RemainderColsList"):
    class _RemainderColsList(list):
        """Dummy placeholder so objects pickled with >=1.6 load on 1.4."""
        pass

    _ct_mod._RemainderColsList = _RemainderColsList


# ---- b) load the model -------------------------------------------------------
MODEL_PATH = Path("models/best_model_calibrated.joblib")
if not MODEL_PATH.exists():
    MODEL_PATH = Path("models/best_model.joblib")

pipe = joblib.load(MODEL_PATH)


# ---- c) patch every ColumnTransformer inside the loaded object --------------
def _patch_every_ct(obj):
    """
    Recursively traverse *obj* and patch each ColumnTransformer so that it
    contains the private attrs that exist only in newer sklearn versions.
    """
    if isinstance(obj, ColumnTransformer):
        # new in sklearn-1.6; harmless default is an empty dict
        if not hasattr(obj, "_name_to_fitted_passthrough"):
            obj._name_to_fitted_passthrough = {}

    # recurse into common container/wrapper patterns
    if isinstance(obj, dict):
        for v in obj.values():
            _patch_every_ct(v)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            _patch_every_ct(v)
    else:
        # sklearn wrappers keep inner estimators in these attributes
        for attr in (
            "steps", "estimators", "base_estimator", "estimator",
            "calibrated_classifiers_", "classifier"
        ):
            if hasattr(obj, attr):
                _patch_every_ct(getattr(obj, attr))


_patch_every_ct(pipe)  # run once at import time


# ──────────────────────────────────────────────────────────────────────────────
# 1. Helper – convert UI dictionaries to one row per horse
# ──────────────────────────────────────────────────────────────────────────────
def build_feature_rows(track: dict, horses: list[dict]) -> pd.DataFrame:
    rows = []
    for h in horses:
        dec_odds = max(h["odds"], 1.01)               # protect against ≤1
        rows.append(
            {
                # race-level
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
                # engineered
                "days_since_last": h["days_since_last"],
                "finish_lag1": h["finish_lag1"],
                "finish_lag2": h["finish_lag2"],
                "finish_lag3": h["finish_lag3"],
                "wins_last6": h["wins_last6"],
                "course_runs": h["course_runs"],
                "dist_runs": h["dist_runs"],
                "jockey_win_rate": h["jockey_win_rate"],
                "jockey_unknown": h["jockey_unknown"],
                "trainer_win_rate": h["trainer_win_rate"],
                "trainer_unknown": h["trainer_unknown"],
            }
        )
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Streamlit UI
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Horse-race Win Model", layout="centered")
st.title("Horse-race Win Probability Model")

with st.form("race_form"):
    st.header("Race information")
    surface = st.selectbox("Surface", ["DIRT", "TURF", "SYNTHETIC"])
    distance = st.number_input("Distance (metres)", 800, 4000, 1600, 100)
    race_class = st.text_input("Class (e.g. G1 / HCP)", value="UNK")
    n_runners = st.number_input(
        "Field size (press Enter to confirm)",
        min_value=2,
        max_value=20,
        value=8,
    )

    st.header("Horse entries")
    horses = []
    for i in range(1, n_runners + 1):
        with st.expander(f"Horse {i}", expanded=(i == 1)):
            age = st.number_input("Age", 2, 15, 4, key=f"age{i}")
            weight = st.number_input("Weight kg", 40.0, 70.0, 55.0, key=f"wt{i}")
            odds = st.number_input(
                "Morning-line odds (decimal)", 1.01, 99.9, 5.0, key=f"odds{i}"
            )

            st.markdown("*Recent form — use **40** for DNF*")
            f1 = st.number_input("Last finish", 1, 40, 3, key=f"f1{i}")
            f2 = st.number_input("Two back", 1, 40, 5, key=f"f2{i}")
            f3 = st.number_input("Three back", 1, 40, 2, key=f"f3{i}")
            wins6 = st.number_input("Wins last 6", 0, 6, 1, key=f"w6{i}")
            dsl = st.number_input("Days since last", 1, 400, 30, key=f"dsl{i}")

            course_runs = st.number_input(
                "Runs on this surface", 0, 50, 3, key=f"cr{i}"
            )
            dist_runs = st.number_input("Runs at this dist", 0, 50, 4, key=f"dr{i}")

            j_wr = st.number_input(
                "Jockey win-rate (0-1)", 0.0, 1.0, 0.10, 0.01, key=f"jwr{i}"
            )
            t_wr = st.number_input(
                "Trainer win-rate (0-1)", 0.0, 1.0, 0.12, 0.01, key=f"twr{i}"
            )
            j_new = st.checkbox("Unknown jockey", key=f"jun{i}")
            t_new = st.checkbox("Unknown trainer", key=f"tun{i}")

        horses.append(
            dict(
                age=age,
                weight=weight,
                post=i,
                odds=odds,
                finish_lag1=f1,
                finish_lag2=f2,
                finish_lag3=f3,
                wins_last6=wins6,
                days_since_last=dsl,
                course_runs=course_runs,
                dist_runs=dist_runs,
                jockey_win_rate=j_wr,
                trainer_win_rate=t_wr,
                jockey_unknown=int(j_new),
                trainer_unknown=int(t_new),
            )
        )

    submitted = st.form_submit_button("Predict")

# ──────────────────────────────────────────────────────────────────────────────
# 3. Prediction
# ──────────────────────────────────────────────────────────────────────────────
if submitted:
    track = dict(surface=surface, distance=distance, race_class=race_class)
    X_new = build_feature_rows(track, horses)
    probs = pipe.predict_proba(X_new)[:, 1]

    X_new["Horse"] = [f"H{i+1}" for i in range(len(horses))]
    X_new["Prob"] = probs

    st.subheader("Win probabilities")
    st.dataframe(
        X_new[["Horse", "decimalPrice", "Prob"]]
        .sort_values("Prob", ascending=False)
        .style.format({"decimalPrice": "{:.2f}", "Prob": "{:.1%}"})
    )

    st.subheader("Top 3")
    top3 = X_new.nlargest(3, "Prob")
    for row in top3.itertuples(index=False):
        st.write(f"**{row.Horse}** — {row.Prob:.1%}")
