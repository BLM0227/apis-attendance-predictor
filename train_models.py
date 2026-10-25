"""Train the three candidate models and generate all evaluation evidence from the raw dataset.

Usage:  python train_models.py [--data data/APIS_PUF_2022_Integrated_Cleaned.csv]

Outputs in models/:
  logistic_regression.joblib, decision_tree.joblib, gaussian_nb.joblib  (fitted Pipelines, TRAIN split only)
  metadata.json   (form options/ranges, library versions)
  evidence.json   (CV, out-of-fold threshold selection, held-out test, subgroup results; read by the app)

Protocol (see src/evaluation.py): split -> CV + threshold selection on TRAIN only -> test scored once.
"""
import argparse
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

from src.config import (CATEGORICAL_COLS, DATA_DIR, GROUP_COL, MODEL_FILES, MODELS_DIR, NUMERICAL_COLS,
                        PREDICTOR_COLS, RANDOM_STATE, RECOMMENDED_MODEL, TARGET)
from src.evaluation import compute_evidence
from src.pipeline import build_models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA_DIR / "APIS_PUF_2022_Integrated_Cleaned.csv"))
    args = ap.parse_args()

    df_raw = pd.read_csv(args.data)
    df_ml = df_raw.drop_duplicates().copy()
    print(f"Rows: {len(df_raw)} -> {len(df_ml)} after removing exact duplicates")
    X, y, groups = df_ml[PREDICTOR_COLS], df_ml[TARGET], df_ml[GROUP_COL]

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    tr, te = next(splitter.split(X, y, groups=groups))
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
    gtr, gte = groups.iloc[tr], groups.iloc[te]
    overlap = set(gtr) & set(gte)
    assert not overlap, "Household leakage between train and test!"
    print(f"Train {Xtr.shape}, test {Xte.shape}, households {gtr.nunique()}/{gte.nunique()}, overlap 0")

    MODELS_DIR.mkdir(exist_ok=True)
    final = build_models()
    for name, model in final.items():
        model.fit(Xtr, ytr)                         # preprocessing fitted on TRAIN only
        joblib.dump(model, MODELS_DIR / MODEL_FILES[name])

    print("Running 5-fold grouped CV on the training split, then scoring the test split once...")
    ev = compute_evidence(df_raw, df_ml, Xtr, ytr, gtr, Xte, yte, final,
                          {"test_households": gte.nunique(), "overlap": len(overlap)})
    ev["meta"] = {"sklearn_version": sklearn.__version__,
                  "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (MODELS_DIR / "evidence.json").write_text(json.dumps(ev, indent=1, ensure_ascii=False), encoding="utf-8")

    meta = {
        "trained_at_utc": ev["meta"]["generated_utc"], "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__, "pandas_version": pd.__version__,
        "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
        "categories": {c: sorted(df_ml[c].dropna().astype(str).unique().tolist()) for c in CATEGORICAL_COLS},
        "defaults": {c: str(df_ml[c].mode().iloc[0]) for c in CATEGORICAL_COLS},
        "numeric": {c: {"min": float(df_ml[c].min()), "max": float(df_ml[c].max()),
                        "median": float(df_ml[c].median())} for c in NUMERICAL_COLS},
    }
    (MODELS_DIR / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- console summary ----
    t = pd.DataFrame(ev["test_not_attending"]).merge(pd.DataFrame(ev["test_overall"])[["Model", "Accuracy"]])
    print("\nHeld-out TEST (threshold 0.50):\n", t[["Model", "Accuracy", "Precision (not attending)",
                                                   "Recall (not attending)", "AP (not attending)"]].to_string(index=False))
    print(f"\nSelected threshold (OOF, train only): {ev['selected_threshold']:.2f}")
    best_ap = pd.DataFrame(ev["cv_ap"]).sort_values("AP not attending (mean)").iloc[-1]["Model"]
    if best_ap != RECOMMENDED_MODEL:
        print(f"WARNING: {best_ap} has the highest CV not-attending AP, not {RECOMMENDED_MODEL}. "
              "Revisit RECOMMENDED_MODEL in src/config.py and the rationale text.")
    print(f"Saved models, metadata.json and evidence.json to {MODELS_DIR}")


if __name__ == "__main__":
    main()
