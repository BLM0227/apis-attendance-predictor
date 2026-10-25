"""Inference helpers. Models are full fitted Pipelines, so preprocessing is never refitted on user input."""
import json
from typing import Optional

import joblib
import pandas as pd
import sklearn

from .config import (CATEGORICAL_COLS, METADATA_PATH, MODEL_FILES, MODELS_DIR, NUMERICAL_COLS,
                     PREDICTOR_COLS)


class ArtifactError(RuntimeError):
    """Raised when model artifacts or metadata are missing/unreadable."""


def load_metadata() -> dict:
    if not METADATA_PATH.exists():
        raise ArtifactError("models/metadata.json not found. Run `python train_models.py` first.")
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def load_models() -> dict:
    models = {}
    for name, fname in MODEL_FILES.items():
        path = MODELS_DIR / fname
        if not path.exists():
            raise ArtifactError(f"Missing model file: models/{fname}. Run `python train_models.py` first.")
        models[name] = joblib.load(path)
    return models


def version_warning(meta: dict) -> Optional[str]:
    trained = meta.get("sklearn_version")
    if trained and trained != sklearn.__version__:
        return (f"Models were trained with scikit-learn {trained} but this environment has "
                f"{sklearn.__version__}. Predictions may be unreliable; reinstall the pinned version "
                f"from requirements.txt or retrain.")
    return None


def validate_inputs(values: dict, meta: dict):
    """Return (errors, warnings). Errors block prediction; warnings are shown alongside the result."""
    errors, warnings = [], []
    for col in PREDICTOR_COLS:
        v = values.get(col)
        if v is None or (isinstance(v, str) and not v.strip()):
            errors.append(f"'{col}' is required.")
    if errors:
        return errors, warnings

    for col in CATEGORICAL_COLS:
        if values[col] not in meta["categories"][col]:
            errors.append(f"'{col}': '{values[col]}' is not a recognised category.")
    for col in NUMERICAL_COLS:
        rng = meta["numeric"][col]
        try:
            x = float(values[col])
        except (TypeError, ValueError):
            errors.append(f"'{col}' must be a number.")
            continue
        if x != x:  # NaN
            errors.append(f"'{col}' must be a number.")
        elif x < rng["min"] or x > rng["max"]:
            errors.append(f"'{col}' must be between {rng['min']:.0f} and {rng['max']:.0f} "
                          f"(the range seen in the training data).")
    if errors:
        return errors, warnings

    age = values["Age as of Last Birthday"]
    if not 5 <= age <= 24:
        errors.append("Age must be between 5 and 24; the model was only trained on that age range.")
    if values["Head: Age as of last birthday"] <= age:
        warnings.append("The household head is the same age or younger than the person. Please double-check "
                        "the ages; this combination is unusual.")
    return errors, warnings


def build_input_frame(values: dict) -> pd.DataFrame:
    """One-row DataFrame with the exact predictor columns/order used in training."""
    return pd.DataFrame([{c: values[c] for c in PREDICTOR_COLS}], columns=PREDICTOR_COLS)


def predict(pipeline, values: dict, threshold: float) -> dict:
    """Predict with a fitted pipeline. Positive decision = 'not attending' when P(not attending) >= threshold."""
    frame = build_input_frame(values)
    classes = list(pipeline.classes_)
    proba = pipeline.predict_proba(frame)[0]
    p_not = float(proba[classes.index(0)])
    label = 0 if p_not >= threshold else 1
    return {"label": label, "p_not_attending": p_not, "p_attending": 1 - p_not, "threshold": threshold}
