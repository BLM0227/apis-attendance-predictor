"""Leak-safe evaluation protocol used to generate models/evidence.json.

1. Split once (household-grouped): TRAIN / TEST.  TEST is not touched until step 4.
2. 5-fold StratifiedGroupKFold CV on TRAIN only -> fold metrics and out-of-fold (OOF) P(not attending).
3. Operating threshold for the recommended model is chosen on the OOF predictions (TRAIN only).
4. Final models are fitted on all of TRAIN and scored ONCE on TEST (default and selected threshold);
   subgroup results are reported from that single test run.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedGroupKFold

from .config import (RANDOM_STATE, RECOMMENDED_MODEL, THRESHOLD_GRID, age_group)
from .pipeline import build_models

DEFAULT_T = 0.50


def _pct(x):
    return round(float(x) * 100, 2)


def _p_not(model, X):
    return model.predict_proba(X)[:, list(model.classes_).index(0)]


def _threshold_metrics(y, p_not, t):
    actual_not = (np.asarray(y) == 0)
    pred_not = p_not >= t
    return {
        "Threshold": t,
        "Accuracy": _pct((pred_not == actual_not).mean()),
        "Precision (not attending)": _pct(precision_score(actual_not, pred_not, zero_division=0)),
        "Recall (not attending)": _pct(recall_score(actual_not, pred_not, zero_division=0)),
        "F1 (not attending)": _pct(f1_score(actual_not, pred_not, zero_division=0)),
    }


def cross_validate(Xtr, ytr, gtr):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    folds = {n: [] for n in build_models()}
    oof = {n: np.zeros(len(Xtr)) for n in build_models()}
    for tr, va in cv.split(Xtr, ytr, groups=gtr):
        for name, model in build_models().items():
            model.fit(Xtr.iloc[tr], ytr.iloc[tr])
            yv, pred = ytr.iloc[va], model.predict(Xtr.iloc[va])
            p_not = _p_not(model, Xtr.iloc[va])
            oof[name][va] = p_not
            folds[name].append({
                "acc": accuracy_score(yv, pred), "prec_att": precision_score(yv, pred, zero_division=0),
                "rec_att": recall_score(yv, pred), "rec_not": recall_score(yv, pred, pos_label=0),
                "f1_att": f1_score(yv, pred), "ap_att": average_precision_score(yv == 1, 1 - p_not),
                "ap_not": average_precision_score(yv == 0, p_not)})
    return folds, oof


def compute_evidence(df_raw, df_ml, Xtr, ytr, gtr, Xte, yte, final_models, extra_test_cols):
    names = list(final_models)
    folds, oof = cross_validate(Xtr, ytr, gtr)

    def stat(name, key, sd=False):
        v = np.array([f[key] for f in folds[name]])
        return _pct(v.std()) if sd else _pct(v.mean())

    cv_main = pd.DataFrame({
        "Model": names,
        "Accuracy (mean)": [stat(n, "acc") for n in names],
        "Accuracy (SD, pp)": [stat(n, "acc", True) for n in names],
        "Precision (mean)": [stat(n, "prec_att") for n in names],
        "Recall attending (mean)": [stat(n, "rec_att") for n in names],
        "Recall not attending (mean)": [stat(n, "rec_not") for n in names],
        "F1 attending (mean)": [stat(n, "f1_att") for n in names]})
    cv_ap = pd.DataFrame({
        "Model": names,
        "AP attending (mean)": [stat(n, "ap_att") for n in names],
        "AP attending (SD, pp)": [stat(n, "ap_att", True) for n in names],
        "AP not attending (mean)": [stat(n, "ap_not") for n in names],
        "AP not attending (SD, pp)": [stat(n, "ap_not", True) for n in names]})

    # ---- threshold chosen on OOF predictions (train only) ----
    oof_thr = pd.DataFrame([_threshold_metrics(ytr, oof[RECOMMENDED_MODEL], t) for t in THRESHOLD_GRID])
    selected = float(oof_thr.loc[oof_thr["F1 (not attending)"].idxmax(), "Threshold"])

    # ---- final held-out TEST (scored once) ----
    rows_o, rows_n, conf, train_acc = [], [], {}, {}
    p_test = {}
    for n in names:
        m = final_models[n]
        pred = m.predict(Xte)
        p_test[n] = _p_not(m, Xte)
        train_acc[n] = accuracy_score(ytr, m.predict(Xtr))
        rows_o.append({"Model": n, "Accuracy": _pct(accuracy_score(yte, pred)),
                       "Precision (attending)": _pct(precision_score(yte, pred, zero_division=0)),
                       "Recall (attending)": _pct(recall_score(yte, pred)),
                       "F1 (attending)": _pct(f1_score(yte, pred))})
        rows_n.append({"Model": n,
                       "Precision (not attending)": _pct(precision_score(yte, pred, pos_label=0, zero_division=0)),
                       "Recall (not attending)": _pct(recall_score(yte, pred, pos_label=0)),
                       "F1 (not attending)": _pct(f1_score(yte, pred, pos_label=0)),
                       "AP (not attending)": _pct(average_precision_score(yte == 0, p_test[n])),
                       "AP (attending)": _pct(average_precision_score(yte == 1, 1 - p_test[n]))})
        conf[n] = confusion_matrix(yte, pred, labels=[0, 1]).tolist()   # rows actual [not, att]
    test_overall, test_not = pd.DataFrame(rows_o), pd.DataFrame(rows_n)

    gen = pd.DataFrame({
        "Model": names,
        "Training accuracy": [_pct(train_acc[n]) for n in names],
        "Test accuracy": test_overall["Accuracy"].tolist(),
        "CV accuracy": cv_main["Accuracy (mean)"].tolist()})
    gen["Train–test gap (pp)"] = (gen["Training accuracy"] - gen["Test accuracy"]).round(2)

    lr_p = p_test[RECOMMENDED_MODEL]
    test_thr = pd.DataFrame([_threshold_metrics(yte, lr_p, t) for t in THRESHOLD_GRID])

    # ---- subgroups: recommended model on TEST ----
    actual_not = (yte.values == 0)

    def grp_table(label, values):
        rows = []
        for g in pd.unique(values):
            mask = np.asarray(values == g)
            n_not = int(actual_not[mask].sum())
            if n_not == 0:
                continue
            rows.append({label: g, "n": int(mask.sum()), "Not attending (n)": n_not,
                         "Recall @0.50": _pct((lr_p[mask & actual_not] >= DEFAULT_T).mean()),
                         "Recall @selected": _pct((lr_p[mask & actual_not] >= selected).mean())})
        return pd.DataFrame(rows)

    ages = Xte["Age as of Last Birthday"].astype(int).map(age_group).values
    sub_age = grp_table("Age group", pd.Series(ages, index=Xte.index))
    order = ["5–9", "10–12", "13–15", "16–18", "19–24"]
    sub_age["o"] = sub_age["Age group"].map(order.index)
    sub_age = sub_age.sort_values("o").drop(columns="o")
    sub_region = grp_table("Region", Xte["Region"]).sort_values("Recall @selected", ascending=False)
    sub_sex = grp_table("Sex", Xte["Sex"])
    sub_res = grp_table("Residence", Xte["Urban / Rural"])
    age_by_thr = {f"{t:.2f}": {g: _pct((lr_p[(ages == g) & actual_not] >= t).mean()) for g in order}
                  for t in THRESHOLD_GRID}

    facts = {
        "rows_raw": int(len(df_raw)), "duplicates_removed": int(len(df_raw) - len(df_ml)),
        "rows_final": int(len(df_ml)),
        "attending_pct": _pct((df_ml["school_attendance"] == 1).mean()),
        "not_attending_pct": _pct((df_ml["school_attendance"] == 0).mean()),
        "train_rows": int(len(Xtr)), "test_rows": int(len(Xte)),
        "train_households": int(gtr.nunique()), "test_households": int(extra_test_cols["test_households"]),
        "household_overlap": int(extra_test_cols["overlap"])}

    ev = {
        "selected_threshold": selected,
        "selection_rule": (f"Threshold with the highest not-attending F1 on out-of-fold predictions of the "
                           f"training split ({RECOMMENDED_MODEL}); the test split was not used."),
        "dataset": facts,
        "cv_main": cv_main.to_dict("records"), "cv_ap": cv_ap.to_dict("records"),
        "threshold_oof": oof_thr.to_dict("records"), "threshold_test": test_thr.to_dict("records"),
        "test_overall": test_overall.to_dict("records"), "test_not_attending": test_not.to_dict("records"),
        "confusion": conf, "generalization": gen.to_dict("records"),
        "subgroup_age": sub_age.to_dict("records"), "subgroup_region": sub_region.to_dict("records"),
        "subgroup_sex": sub_sex.to_dict("records"), "subgroup_residence": sub_res.to_dict("records"),
        "age_recall_by_threshold": age_by_thr,
    }
    return ev
