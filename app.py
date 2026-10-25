"""Streamlit app: Predicting school attendance among Filipino children and youth aged 5-24 (APIS 2022).
Run locally:  streamlit run app.py

Required areas (Final Project Requirements §5):
  A Project Overview · B Models & Evaluation · C Prediction (input) · D Prediction Result (+ interpretation)
  · E About / Limitations
"""
import pandas as pd
import streamlit as st

from src import evidence as ev
from src import plots
from src import ui
from src.config import (MODEL_FILES, MODEL_INFO,
                        PREPROCESSING_NOTE, RECOMMENDED_MODEL, THRESHOLD_GRID)
from src.interpretation import build_interpretation
from src.prediction import (ArtifactError, load_metadata, load_models, predict, validate_inputs,
                            version_warning)

st.set_page_config(page_title="School Attendance Predictor · APIS 2022", page_icon="🎓", layout="wide")
ui.inject_css()

if not ev.AVAILABLE:
    ui.header("School Attendance Predictor", "Setup required")
    st.error("`models/evidence.json` was not found. Run `python train_models.py` first. It trains the models "
             "and generates all evaluation evidence the app displays (see README).")
    st.stop()

F = ev.F
SEL = ev.SELECTED_THRESHOLD

P_OVERVIEW = "Overview"
P_EVAL = "Models & Evaluation"
P_PREDICT = "Prediction"
P_ABOUT = "Limitations & responsible use"
NAV_ICONS = {P_OVERVIEW: ":material/home:", P_EVAL: ":material/analytics:",
             P_PREDICT: ":material/online_prediction:", P_ABOUT: ":material/policy:"}


@st.cache_resource(show_spinner="Loading trained models…")
def get_artifacts():
    return load_models(), load_metadata()


def show_table(df: pd.DataFrame):
    d = df.copy()
    if "Model" in d.columns:
        d["Model"] = d["Model"].apply(lambda m: f"⭐ {m}" if m == RECOMMENDED_MODEL else m)
    st.dataframe(d, hide_index=True)


# ======================================================================================
# A. Project overview
# ======================================================================================
def page_overview():
    f = ev.DATASET_FACTS
    ui.header("Predicting School Attendance of Children and Youth in the Philippines",
              "Machine-learning extension of the APIS 2022 data story · ages 5–24")

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        ui.callout_card(
            "Stakeholder question",
            "<b>DepEd and education policymakers:</b> Can the characteristics we already know about a person "
            "and their household help flag who is likely not attending school, so that attention can be directed "
            "to the right places for verification and further investigation?", tint=True)
    with c2:
        ui.callout_card(
            "Core message",
            f"Logistic Regression is the recommended model: <b>{F['lr_acc']:.1f}%</b> test accuracy. At the "
            f"default 0.50 threshold it finds about <b>{F['lr_rec']:.0f}%</b> of non-attending individuals "
            f"({F['lr_prec']:.0f}% of its flags are correct); at the validation-selected {SEL:.2f} threshold, "
            f"about <b>{F['te_rec_s']:.0f}%</b> ({F['te_prec_s']:.0f}% correct). It is weakest for "
            f"<b>ages 5–9</b> (only {F['age_min']:.0f}% found at {SEL:.2f}), so it supports human review "
            "and cannot replace it.")

    ui.notice(
        "Figures on these pages come from the unweighted survey sample used for model training "
        f"({ev.DATASET_FACTS['not_attending_pct']:.1f}% not attending), so they differ from the weighted national estimates in the data-story "
        "dashboard. “Not attending” does not automatically mean dropped out, and many people aged 19–24 may "
        "have finished their education. Predictions describe patterns in 2022 data and do not prove cause "
        "and effect.", lead="Read before interpreting.")

    ui.section(1, "Context: what data was modelled?")
    k = st.columns(4, gap="medium")
    with k[0]:
        ui.kpi(f"{f['not_attending_pct']}%", "Not attending school", "Ages 5–24, unweighted sample", "Sample")
    with k[1]:
        ui.kpi(f"{f['attending_pct']}%", "Attending school", "Ages 5–24, unweighted sample")
    with k[2]:
        ui.kpi(f"{f['rows_final']:,}", "Individuals modelled",
               f"{f['rows_raw']:,} raw rows minus {f['duplicates_removed']} exact duplicates")
    with k[3]:
        ui.kpi(f"{f['household_overlap']}", "Households in both train and test",
               f"{f['train_rows']:,} train / {f['test_rows']:,} test individuals")

    ui.section(2, "Problem: what is predicted, and from what?")
    a, b = st.columns(2, gap="medium")
    with a, ui.card("problem"):
        ui.card_title("Binary classification of school-attendance status")
        st.markdown(
            "- **Target:** `school_attendance`, where 1 = attending and 0 = not attending\n"
            "- **Unit:** one individual aged 5–24\n"
            "- **Positive class for stakeholder purposes:** not attending (the group DepEd wants to find)\n"
            "- **Source:** PSA Annual Poverty Indicators Survey 2022 (Member, Household and Other Relevant "
            "Information records merged by Region and Household ID)\n"
            "- **SDG link:** SDG 4, Target 4.1 (complete free, equitable, quality primary and secondary education)")
    with b, ui.card("predictors"):
        ui.card_title("13 predictors, all known without asking about attendance")
        st.markdown(
            "- **Person:** age, sex, marital status, relationship to household head, highest grade level completed\n"
            "- **Household:** family size, average monthly family income, head's sex / age / marital status / "
            "highest grade\n"
            "- **Place:** region, urban or rural\n"
            "- **Excluded to prevent leakage:** *Currently Attending School*, *Why not attending school* and "
            "other current-schooling variables")

    ui.section(3, "Models: what was compared?")
    m = st.columns(3, gap="medium")
    for col, (name, info) in zip(m, MODEL_INFO.items()):
        with col:
            tag = " ⭐ Recommended" if name == RECOMMENDED_MODEL else ""
            ui.callout_card(f"{info['type']}{tag}", f"<b>{name}</b><br>{info['desc']}",
                            tint=(name == RECOMMENDED_MODEL))

    ui.section(4, "Using this app")
    s = st.columns(3, gap="medium")
    steps = [("Review the evidence", "Open <b>Models & Evaluation</b> to see validation and held-out test results, "
                                    "error trade-offs and subgroup gaps."),
             ("Make a prediction", "Open <b>Prediction</b>, pick a model in the sidebar, enter a profile and press "
                                   "<b>Predict</b>."),
             ("Read the meaning", "Each result is explained for DepEd: what it suggests, a possible next step, "
                                  "and the cautions that apply.")]
    for col, (t, body) in zip(s, steps):
        with col:
            ui.callout_card(t, body)


# ======================================================================================
# B. Models & evaluation
# ======================================================================================
def subgroup_note(df):
    small = df[df["Not attending (n)"] < 100]
    if small.empty:
        return ""
    names = ", ".join(f"{g} (n={n})" for g, n in zip(small.iloc[:, 0], small["Not attending (n)"]))
    return f" Small samples, interpret with caution: {names}."


def page_evaluation():
    D = ev.DATASET_FACTS
    ui.header("Models & Evaluation", "Which model should be trusted, and what do its errors cost?")
    ui.notice(f"<b>Recommended model: ⭐ {RECOMMENDED_MODEL}.</b> It is the primary candidate for further "
              "validation, not a model approved for deployment. Models were compared and the threshold was "
              "chosen using <b>validation evidence from the training split only</b> (sections 2 and 5). The "
              "held-out test split was scored once afterwards (sections 3, 4 and 6).", blue=True)

    # 1 ---------------------------------------------------------------------------------
    ui.section(1, "Design: how were the models compared?")
    a, b = st.columns(2, gap="medium")
    with a, ui.card("design1"):
        ui.card_title("Same data, same preprocessing, same split for all three models")
        st.write(PREPROCESSING_NOTE)
        st.write(f"Split: household-grouped and stratified, so members of one household are never in both train "
                 f"and test ({D['train_rows']:,} train / {D['test_rows']:,} test individuals; "
                 f"{D['train_households']:,} / {D['test_households']:,} households). Cross-validation and "
                 "threshold selection use the training split only (`random_state=42`).")
    with b, ui.card("design2"):
        ui.card_title("Why accuracy alone is not enough")
        st.write(f"About {D['attending_pct']:.0f}% of individuals attend, so predicting “attending” for everyone "
                 f"would already score about {D['attending_pct']:.0f}% accuracy. The primary metric is "
                 "**recall for the not-attending class**, read together with precision, F1, Average Precision "
                 "(AP) and the confusion matrices.")
    rows = [{"Model": n, "Type": i["type"], "Settings": i["settings"]} for n, i in MODEL_INFO.items()]
    with ui.card("design3"):
        ui.card_title("Candidate models")
        show_table(pd.DataFrame(rows))

    # 2 ---------------------------------------------------------------------------------
    ui.section(2, "Validation evidence: is performance consistent?")
    dt_vs = "slightly higher" if F["dt_cv_rec"] > F["lr_cv_rec"] else "not higher"
    with ui.card("cv"):
        ui.card_title("Five-fold household-grouped cross-validation on the training split "
                      "(means; SD in percentage points)",
                      "Used to compare models. The test split is not involved.")
        show_table(ev.cv_main)
        show_table(ev.cv_ap)
        st.write(f"Accuracy varied by at most {F['max_cv_acc_sd']:.2f} pp (SD) across folds for every model, but "
                 "consistency is not usefulness: Gaussian Naive Bayes is consistently poor. Logistic Regression "
                 f"has the highest cross-validated AP for not attending ({F['lr_cv_ap']:.2f}%). The Decision "
                 f"Tree's not-attending recall is {dt_vs} ({F['dt_cv_rec']:.2f}% vs {F['lr_cv_rec']:.2f}%), which "
                 "is why several metrics are weighed.")

    # 3 ---------------------------------------------------------------------------------
    ui.section(3, "Final held-out test: how does it do on unseen households?")
    st.caption(f"{D['test_rows']:,} test individuals from {D['test_households']:,} households never used for "
               "fitting or selection · default threshold 0.50")
    c1, c2 = st.columns([3, 2], gap="medium")
    with c1, ui.card("testchart"):
        ui.card_title("Logistic Regression has the best balance of precision and recall for non-attenders")
        st.pyplot(plots.not_attending_metrics_figure())
    with c2, ui.card("testtable"):
        ui.card_title("Test metrics, not-attending class")
        show_table(ev.test_not_attending[["Model", "Precision (not attending)", "Recall (not attending)",
                                          "F1 (not attending)"]])
        ui.card_title("Test metrics, attending class and overall")
        show_table(ev.test_overall[["Model", "Accuracy", "Precision (attending)", "Recall (attending)"]])
    lr_cm, nb_cm = F["lr_cm"], F["nb_cm"]
    with ui.card("confusion"):
        ui.card_title("Gaussian Naive Bayes labels almost everyone as not attending",
                      "Confusion matrices on the test set (rows = actual, columns = predicted)")
        cols = st.columns(3)
        for col, name in zip(cols, ev.MODELS):
            col.pyplot(plots.confusion_figure(name))
        st.write(f"Logistic Regression correctly flags {lr_cm[0][0]:,} of {sum(lr_cm[0]):,} non-attending "
                 f"individuals and wrongly flags {lr_cm[1][0]:,} attending ones. Gaussian Naive Bayes catches "
                 f"{nb_cm[0][0]:,} of the {sum(nb_cm[0]):,} non-attenders only by also labelling "
                 f"{nb_cm[1][0]:,} of {sum(nb_cm[1]):,} attending individuals as not attending.")

    # 4 ---------------------------------------------------------------------------------
    ui.section(4, "Generalisation: does it memorise?")
    g1, g2 = st.columns([3, 2], gap="medium")
    with g1, ui.card("gen"):
        ui.card_title(f"Only the Decision Tree overfits: {F['dt_train_acc']:.2f}% training vs "
                      f"{F['dt_test_acc']:.2f}% test accuracy")
        st.pyplot(plots.generalization_figure())
    with g2, ui.card("gentable"):
        ui.card_title("Train, validation and test accuracy (%)")
        show_table(ev.generalization)

    # 5 ---------------------------------------------------------------------------------
    ui.section(5, "Trade-off: false negatives versus false positives")
    t1, t2 = st.columns([3, 2], gap="medium")
    with t1, ui.card("thr"):
        ui.card_title("Lowering the threshold finds more non-attenders but adds false flags",
                      "Logistic Regression, out-of-fold predictions on the training split (validation)")
        st.pyplot(plots.threshold_figure())
    with t2, ui.card("thrtable"):
        ui.card_title("Validation (out-of-fold) – used to choose the threshold")
        show_table(ev.threshold_oof)
        ui.card_title("Held-out test at each threshold – informational only")
        show_table(ev.threshold_test)
    ui.notice(
        "A <b>false negative</b> is a non-attending person predicted to attend and possibly overlooked. A "
        "<b>false positive</b> is an attending person flagged, which may waste follow-up effort. "
        f"{ev.SELECTION_RULE} That gave <b>{SEL:.2f}</b>: on validation it raises not-attending recall from "
        f"{F['oof_rec_d']:.1f}% to {F['oof_rec_s']:.1f}% (precision {F['oof_prec_d']:.1f}% to "
        f"{F['oof_prec_s']:.1f}%); on the held-out test, recall {F['te_rec_d']:.1f}% to {F['te_rec_s']:.1f}% "
        f"and precision {F['te_prec_d']:.1f}% to {F['te_prec_s']:.1f}%. Whether to adopt it depends on "
        "follow-up capacity and the cost of false flags, which stakeholders have not yet defined (no "
        "monetary cost ratio is used).", lead="Operating threshold chosen without the test set.")

    # 6 ---------------------------------------------------------------------------------
    ui.section(6, "Subgroups: who does the model miss?")
    st.caption(f"Recommended model, held-out test set, threshold {SEL:.2f}. n = actual not-attending "
               "individuals in the group; the 0.50 values are in the tables.")
    s1, s2 = st.columns(2, gap="medium")
    with s1, ui.card("age"):
        ui.card_title(f"Recall rises from {F['age_min']:.0f}% at ages {F['age_min_grp']} to "
                      f"{F['age_max']:.0f}% at ages {F['age_max_grp']}")
        st.pyplot(plots.dotplot(ev.subgroup_age, "Age group"))
        with st.expander("Table (with sample sizes)"):
            st.dataframe(ev.subgroup_age, hide_index=True)
    with s2, ui.card("region"):
        ui.card_title(f"Regional recall ranges from {F['reg_min_v']:.0f}% ({F['reg_min']}) to "
                      f"{F['reg_max_v']:.0f}% ({F['reg_max']})")
        st.pyplot(plots.region_figure(ev.subgroup_region))
        with st.expander("Table (with sample sizes)"):
            st.dataframe(ev.subgroup_region, hide_index=True)
    s3, s4 = st.columns(2, gap="medium")
    with s3, ui.card("sex"):
        ui.card_title(f"Sex: {F['sex_gap']:.1f} pp difference")
        st.pyplot(plots.dotplot(ev.subgroup_sex, "Sex"))
    with s4, ui.card("res"):
        ui.card_title(f"Urban / rural: {F['res_gap']:.1f} pp difference")
        st.pyplot(plots.dotplot(ev.subgroup_residence, "Residence"))
    ui.notice("These gaps are signals for further investigation, not proof of unfairness."
              + subgroup_note(ev.subgroup_age) +
              " The model should <b>not</b> be the sole method for identifying younger children who may "
              "need support.", lead="Interpret with care.")

    # 7 ---------------------------------------------------------------------------------
    ui.section(7, f"Recommendation: {RECOMMENDED_MODEL}")
    rec = pd.DataFrame({
        "Decision question": [
            "Strongest supporting evidence", "Main trade-off", "Why not simply the highest accuracy?",
            "Why not the Decision Tree?", "Why not Gaussian Naive Bayes?", "Stakeholder consequence"],
        "Response": [
            f"Validation: highest not-attending AP ({F['lr_cv_ap']:.2f}%) and stable folds. Test: accuracy "
            f"{F['lr_acc']:.2f}%, not-attending AP {F['lr_ap']:.2f}%; train/CV/test accuracy within "
            f"{max(F['lr_gap'], 0.2):.1f} pp; interpretable.",
            f"At 0.50 it finds about {F['lr_rec']:.0f}% of non-attenders with about {F['lr_prec']:.0f}% "
            "precision; lowering the threshold finds more but adds false flags.",
            "Most individuals attend, so accuracy hides how many non-attenders are missed. The choice weighs "
            "not-attending recall, precision, AP and generalisation against follow-up cost.",
            f"Similar not-attending recall (CV {F['dt_cv_rec']:.1f}% vs {F['lr_cv_rec']:.1f}%) but a "
            f"{F['dt_gap']:.1f} pp overfitting gap and much lower not-attending AP ({F['dt_ap']:.1f}% test).",
            f"{F['nb_rec']:.1f}% not-attending recall but only {F['nb_prec']:.1f}% precision and "
            f"{F['nb_acc']:.1f}% accuracy: it flags nearly everyone.",
            "Missed non-attenders may not get support and false flags can waste effort. Both need human "
            f"verification, and ages 5–9 need other screening (recall only {F['age_min']:.0f}% at {SEL:.2f})."],
    })
    with ui.card("rec"):
        st.dataframe(rec, hide_index=True)


# ======================================================================================
# C/D. Prediction + result
# ======================================================================================
def page_prediction(model_name: str, threshold: float):
    ui.header("Predict School Attendance", "Describe a person, choose a model, and read what the result means "
              "for DepEd", pills=("APIS 2022", "SDG 4 · Target 4.1", f"Model: {model_name}"))
    try:
        models, meta = get_artifacts()
    except ArtifactError as e:
        st.error(f"{e}")
        st.stop()
    except Exception as e:  # corrupted or incompatible pickle
        st.error(f"Could not load model artifacts: {e}")
        st.stop()
    warn = version_warning(meta)
    if warn:
        st.warning(warn)

    ui.notice(f"Model used: <b>{model_name}</b>"
              + (" (recommended)" if model_name == RECOMMENDED_MODEL else " (comparison model, not recommended)")
              + f" · decision threshold <b>{threshold:.2f}</b>. Change both in the sidebar. "
              + (f"{SEL:.2f} is the threshold selected on validation data (training split only); it still \
needs stakeholder agreement." if threshold == SEL else ""), blue=True)

    cats, nums, defaults = meta["categories"], meta["numeric"], meta["defaults"]

    def select(label, col, key):
        opts = cats[col]
        return st.selectbox(label, opts, index=opts.index(defaults[col]), key=key)

    age_col, fam_col = "Age as of Last Birthday", "Family Size"
    head_age_col = "Head: Age as of last birthday"
    inc_col = "Average monthly income of the family January - June 2022"

    ui.section(1, "Inputs: describe the person")
    with st.form("prediction_form"):
        st.markdown("**The person**")
        a, b, c = st.columns(3)
        with a:
            age = st.number_input("Age as of last birthday (years)", min_value=5, max_value=24, value=12, step=1)
            sex = select("Sex", "Sex", "sex")
        with b:
            region = select("Region", "Region", "region")
            urban = select("Urban / Rural", "Urban / Rural", "urban")
        with c:
            rel = select("Relationship to household head", "Relationship to Household Head", "rel")
            marital = select("Marital status", "Marital Status", "marital")
        grade = select("Highest grade completed (level)", "Highest Grade Completed - Level", "grade")

        st.markdown("**The household**")
        d, e, f = st.columns(3)
        with d:
            fam_min, fam_max = int(nums[fam_col]["min"]), int(nums[fam_col]["max"])
            fam = st.number_input("Family size (persons)", min_value=fam_min, max_value=fam_max,
                                  value=min(max(int(nums[fam_col]["median"]), fam_min), fam_max), step=1)
            inc_max = float(nums[inc_col]["max"])
            income = st.number_input("Average monthly family income, Jan–Jun 2022 (PHP)",
                                     min_value=float(nums[inc_col]["min"]), max_value=inc_max,
                                     value=min(float(nums[inc_col]["median"]), inc_max),
                                     step=500.0)
        with e:
            h_min, h_max = int(nums[head_age_col]["min"]), int(nums[head_age_col]["max"])
            head_age = st.number_input("Household head's age (years)", min_value=h_min, max_value=h_max,
                                       value=min(max(int(nums[head_age_col]["median"]), h_min), h_max), step=1)
            head_sex = select("Household head's sex", "Head: Sex", "hsex")
        with f:
            head_marital = select("Household head's marital status", "Head: Marital Status", "hmar")
            head_grade = select("Household head's highest grade completed", "Head: Highest grade completed", "hgr")

        compare = st.checkbox("Also show what the other models predict for the same input")
        submitted = st.form_submit_button("Predict", type="primary")

    if not submitted:
        ui.notice("No prediction yet. Results appear here after you press <b>Predict</b>.", blue=True)
        return

    values = {
        "Region": region, "Relationship to Household Head": rel, "Sex": sex, age_col: int(age),
        "Marital Status": marital, "Highest Grade Completed - Level": grade, fam_col: int(fam),
        "Head: Sex": head_sex, head_age_col: int(head_age), "Head: Marital Status": head_marital,
        "Head: Highest grade completed": head_grade, "Urban / Rural": urban, inc_col: float(income),
    }
    errors, warnings = validate_inputs(values, meta)
    if errors:
        for msg in errors:
            st.error(msg)
        return
    for msg in warnings:
        st.warning(msg)
    try:
        result = predict(models[model_name], values, threshold)
    except Exception as e:
        st.error(f"The prediction could not be generated: {e}")
        return
    interp = build_interpretation(model_name, result, int(age))
    not_att = result["label"] == 0

    ui.section(2, "Result: what did the model predict?")
    k = st.columns(3, gap="medium")
    with k[0]:
        ui.kpi(model_name, "Model used", "Loaded with its own fitted preprocessing")
    with k[1]:
        ui.kpi("Not attending" if not_att else "Attending", "Predicted status",
               f"Flagged when P(not attending) ≥ {result['threshold']:.2f}")
    with k[2]:
        ui.kpi(f"{result['p_not_attending'] * 100:.1f}%", "Model probability of non-attendance",
               f"Probability band: {interp['band']} · a model score, not a certainty")

    ui.section(3, "Meaning: what this suggests for DepEd")
    ui.notice(ui.md_inline(interp["plain"]), blue=not not_att)
    x, y = st.columns(2, gap="medium")
    with x:
        ui.callout_card("Meaning in context", ui.md_inline(interp["context"]), tint=True)
    with y:
        ui.callout_card("Suggested consideration", ui.md_inline(interp["action"]))
    items = "".join(f"<li>{ui.md_inline(c)}</li>" for c in interp["cautions"])
    ui.notice(f"<ul style='margin:.4rem 0 0 1.1rem;padding:0'>{items}</ul>", lead="Uncertainty and caution.")

    if compare:
        with ui.card("compare"):
            ui.card_title("Same input, all models",
                          "Models can disagree. Rely on the recommended model, and only with human verification.")
            rows = []
            for name, pipe in models.items():
                r = predict(pipe, values, threshold)
                rows.append({"Model": name + (" ⭐" if name == RECOMMENDED_MODEL else ""),
                             "Predicted status": "Not attending" if r["label"] == 0 else "Attending",
                             "P(not attending)": f"{r['p_not_attending'] * 100:.1f}%"})
            st.dataframe(pd.DataFrame(rows), hide_index=True)


# ======================================================================================
# E. About / limitations
# ======================================================================================
def page_about():
    ui.header("Limitations & Responsible Use", "What this tool can and cannot be used for")
    ui.section(1, "Intended and unintended use")
    a, b = st.columns(2, gap="medium")
    with a, ui.card("intended"):
        ui.card_title("Intended use")
        st.write("A decision-support aid for DepEd and education policymakers to explore which profiles resemble "
                 "non-attending individuals in 2022 APIS data, for planning further review, monitoring or "
                 "outreach alongside other evidence.")
    with b, ui.card("unintended"):
        ui.card_title("Not intended for")
        st.markdown(
            "- Automatically deciding who receives or is denied assistance\n"
            "- Any action about an individual without human verification\n"
            "- Concluding that a characteristic *causes* non-attendance\n"
            f"- Screening young children (ages 5–9: recall was only {F['age_min']:.0f}% at threshold {SEL:.2f}, "
            f"{F['age_min_d']:.0f}% at 0.50)\n"
            "- Other years, populations or ages outside 5–24\n"
            "- Treating a probability or threshold as certain")

    ui.section(2, "Known limitations")
    with ui.card("limits"):
        st.markdown(
            f"- **Missed cases:** at threshold 0.50 Logistic Regression misses about {100 - F['te_rec_d']:.0f}% of "
            f"non-attending individuals on the test data ({100 - F['te_rec_s']:.0f}% at {SEL:.2f}).\n"
            f"- **Threshold not final:** {SEL:.2f} was selected on validation data from the training split; "
            "it still needs stakeholder agreement on follow-up capacity and error costs.\n"
            f"- **Age gap:** recall at {SEL:.2f} climbs from {F['age_min']:.0f}% (ages {F['age_min_grp']}) to "
            f"{F['age_max']:.0f}% (ages {F['age_max_grp']}); regional recall ranges from {F['reg_min_v']:.0f}% "
            f"({F['reg_min']}) to {F['reg_max_v']:.0f}% ({F['reg_max']}). Some groups have few non-attending "
            "individuals, so these are signals to investigate, not explanations.\n"
            "- **Data:** one survey year (2022); survey sampling and reporting bias are possible; likely drivers "
            "such as distance to school, disability, health or conflict are not in the predictors; survey "
            "weights were not used in training.\n"
            "- **Drift:** relationships may change after 2022, so retraining and re-validation would be needed.\n"
            f"- **Imbalance:** about {ev.DATASET_FACTS['attending_pct']:.0f}% of individuals attend, so accuracy "
            "overstates usefulness.")

    ui.section(3, "Fairness, privacy and bias")
    with ui.card("fair"):
        st.write("The model uses sex, region, urban/rural and household characteristics, which may encode existing "
                 "inequalities in the survey. Subgroup recall was checked for age, region, sex and urban/rural on the test data. This app "
                 "does not store inputs or personal identifiers; they are used only to produce the on-screen "
                 "prediction.")

    ui.section(4, "Project information")
    with ui.card("info"):
        st.write("Student: Brien Leij O. Mauricio · DS012 · Technological Institute of the Philippines, Manila, "
                 "College of Computer Studies · DSA31S1 Machine Learning, First Semester SY 2026–2027.")
        st.write("Data: Philippine Statistics Authority, Annual Poverty Indicators Survey 2022 "
                 "(https://psada.psa.gov.ph/catalog/APIS/about).")


# ======================================================================================
with st.sidebar:
    ui.brand()
    ui.side_label("APP SECTIONS")
    PAGES = [P_OVERVIEW, P_EVAL, P_PREDICT, P_ABOUT]
    if "page" not in st.session_state:
        st.session_state["page"] = P_OVERVIEW
    for i, name in enumerate(PAGES):
        key = ("navactive_" if name == st.session_state["page"] else "nav_") + str(i)
        if st.button(name, key=key, icon=NAV_ICONS[name], use_container_width=True):
            st.session_state["page"] = name
            st.rerun()
    page = st.session_state["page"]
    sel_model, sel_threshold = RECOMMENDED_MODEL, SEL
    if page == P_PREDICT:
        ui.side_label("MODEL SETTINGS")
        sel_model = st.selectbox(
            "Model", list(MODEL_FILES.keys()), index=list(MODEL_FILES.keys()).index(RECOMMENDED_MODEL),
            format_func=lambda m: f"{m} (recommended)" if m == RECOMMENDED_MODEL else m,
            help="The selected model's own fitted pipeline generates the prediction.")
        sel_threshold = st.select_slider(
            "Decision threshold", options=THRESHOLD_GRID, value=SEL,
            format_func=lambda x: f"{x:.2f}",
            help="A person is flagged as not attending when the model's probability of non-attendance is at or "
                 f"above this value. 0.50 is the conventional default; {SEL:.2f} was selected on validation data.")
    st.caption("Decision-support tool. Not for automated decisions about individuals.")

if page == P_OVERVIEW:
    page_overview()
elif page == P_EVAL:
    page_evaluation()
elif page == P_PREDICT:
    page_prediction(sel_model, sel_threshold)
else:
    page_about()
