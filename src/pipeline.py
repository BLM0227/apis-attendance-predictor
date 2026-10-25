"""Model/pipeline builders shared by training and evaluation (single source of truth)."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .config import CATEGORICAL_COLS, NUMERICAL_COLS, RANDOM_STATE


def build_preprocessor() -> ColumnTransformer:
    cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    num = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    return ColumnTransformer([("categorical", cat, CATEGORICAL_COLS), ("numerical", num, NUMERICAL_COLS)])


def build_models() -> dict:
    """Fresh, unfitted pipelines. Preprocessing is inside each pipeline, so it is refitted per CV fold."""
    return {
        "Logistic Regression": Pipeline([("preprocessor", build_preprocessor()),
                                         ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))]),
        "Decision Tree": Pipeline([("preprocessor", build_preprocessor()),
                                   ("classifier", DecisionTreeClassifier(random_state=RANDOM_STATE))]),
        "Gaussian Naive Bayes": Pipeline([("preprocessor", build_preprocessor()),
                                          ("classifier", GaussianNB())]),
    }
