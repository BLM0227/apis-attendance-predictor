"""Translate a prediction into DepEd-oriented language. Interpretation depends on the predicted class,
probability band, distance from the threshold, the chosen model and the person's age group."""
from .config import RECOMMENDED_MODEL, age_group
from . import evidence as ev


def probability_band(p_not: float) -> str:
    if p_not < 0.20:
        return "low"
    if p_not < 0.40:
        return "lower-moderate"
    if p_not < 0.60:
        return "borderline"
    return "higher"


def _threshold_row(threshold: float):
    row = ev.threshold_test[(ev.threshold_test["Threshold"] - threshold).abs() < 1e-9]
    return None if row.empty else row.iloc[0]


def build_interpretation(model_name: str, result: dict, age: int) -> dict:
    p_not, thr, label = result["p_not_attending"], result["threshold"], result["label"]
    band = probability_band(p_not)
    grp = age_group(age)
    is_lr = model_name == RECOMMENDED_MODEL
    pct = f"{p_not * 100:.1f}%"

    # ---- plain-language result --------------------------------------------------------------
    if label == 0:
        plain = (f"For this profile, **{model_name}** predicts the person is **likely NOT currently attending "
                 f"school**. Its estimated probability of non-attendance is {pct}, at or above the "
                 f"{thr:.2f} decision threshold.")
    else:
        plain = (f"For this profile, **{model_name}** predicts the person is **likely currently attending "
                 f"school**. Its estimated probability of non-attendance is {pct}, below the "
                 f"{thr:.2f} decision threshold.")

    # ---- meaning in context -----------------------------------------------------------------
    context = ("The result reflects how closely this combination of demographic, household, educational and "
               "income characteristics resembles individuals in the 2022 APIS who were or were not in school. "
               "It is a statistical pattern match for a similar profile, not a finding about this particular "
               "person's circumstances, and it does not explain why they are or are not in school.")

    # ---- possible action --------------------------------------------------------------------
    if label == 0:
        action = ("Treat this as a **prompt to verify**, not a conclusion: check enrolment status through "
                  "school or local records before deciding whether any follow-up or education-support "
                  "programme applies. Profiles like this one could be considered for further review or "
                  "monitoring when planning outreach.")
    else:
        action = ("No follow-up is triggered by this result alone. If there are other reasons for concern "
                  "(for example, reports of absence or dropping out), act on that information directly; "
                  "an 'attending' prediction should not be used to rule anyone out.")

    # ---- uncertainty / caution --------------------------------------------------------------
    F = ev.F
    cautions = []
    near = abs(p_not - thr) < 0.10
    if near:
        cautions.append(f"The probability ({pct}) is within 10 percentage points of the {thr:.2f} threshold, "
                        "so a small change in the inputs could flip the result. Treat it as uncertain.")
    cautions.append(f"Probabilities are model scores (probability band: **{band}**), not certainty.")

    row = _threshold_row(thr)
    if is_lr and row is not None:
        prec, rec = row["Precision (not attending)"], row["Recall (not attending)"]
        cautions.append(
            f"On the held-out test data at threshold {thr:.2f}, Logistic Regression's 'not attending' flags "
            f"were correct about {prec:.0f}% of the time and it found about {rec:.0f}% of people who were "
            f"actually not attending. Roughly {100 - rec:.0f}% of non-attending individuals were missed.")
    elif not is_lr:
        cautions.append("Threshold-level precision/recall were evaluated only for Logistic Regression, "
                        "so no threshold-specific reliability figures are available for this model.")

    if is_lr:
        rec_age = ev.age_recall(grp, thr)
        if rec_age is not None and rec_age < 50:
            cautions.append(
                f"**Age-group warning ({grp}):** at threshold {thr:.2f}, Logistic Regression found only "
                f"{rec_age:.1f}% of actual non-attenders in this age group on the test data. "
                + ("An 'attending' prediction here gives very little assurance."
                   if label == 1 else "A 'not attending' flag is useful, but many similar cases would be missed."))
    elif age <= 15:
        cautions.append("Age-group recall was only measured for Logistic Regression, where it was low "
                        "for younger ages. Treat this model's result for children with extra caution.")

    if model_name == "Gaussian Naive Bayes":
        cautions.append(f"**Not recommended for decisions:** on the test data this model labelled almost "
                        f"everyone as not attending (only {F['nb_prec']:.0f}% of its 'not attending' flags "
                        f"were correct; overall accuracy {F['nb_acc']:.1f}%). Shown for comparison only.")
    if model_name == "Decision Tree":
        cautions.append(f"This model overfit the training data ({F['dt_train_acc']:.2f}% training vs "
                        f"{F['dt_test_acc']:.2f}% test accuracy) and its probabilities are coarse. Prefer the "
                        "recommended model.")

    cautions.append("This is decision-support information. It does not prove causation, guarantee an "
                    "outcome, or replace human review and professional judgement.")

    return {"plain": plain, "context": context, "action": action, "cautions": cautions,
            "band": band, "near_threshold": near, "age_group": grp}
