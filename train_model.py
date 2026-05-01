"""
train_model.py
--------------
Loads the Titanic dataset, trains a RandomForestClassifier
with a preprocessing pipeline, evaluates accuracy,
and saves the trained pipeline to models/titanic_model.pkl.
"""

import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report

# ── Paths ────────────────────────────────────────────────────────────────────
CSV_PATH   = "data/titanic.csv"
MODEL_PATH = "models/titanic_model.pkl"

# ── Column definitions ────────────────────────────────────────────────────────
TARGET_COLUMN      = "Survived"
NUMERIC_FEATURES   = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
CATEGORIC_FEATURES = ["Sex", "Embarked"]
ALL_FEATURES       = NUMERIC_FEATURES + CATEGORIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
      - imputes + passes numeric columns through
      - imputes + ordinal-encodes categorical columns
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
        # No scaling needed for tree-based models
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer,  NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORIC_FEATURES)
    ])

    return preprocessor


def train() -> None:
    # ── Load data ────────────────────────────────────────────────────────────
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"Dataset not found at '{CSV_PATH}'.\n"
            "Download titanic.csv from:\n"
            "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        )

    print("[TRAIN] Loading Titanic dataset …")
    df = pd.read_csv(CSV_PATH, sep=None, engine="python")

    # Drop rows where target is missing
    df = df.dropna(subset=[TARGET_COLUMN])

    X = df[ALL_FEATURES]
    y = df[TARGET_COLUMN]

    print(f"[TRAIN] Dataset size: {len(df)} rows | "
          f"Survived: {y.sum()} | Did not survive: {(y == 0).sum()}")

    # ── Train / test split ────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Build full pipeline ───────────────────────────────────────────────────
    pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("classifier",   RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    # ── Fit ───────────────────────────────────────────────────────────────────
    print("[TRAIN] Training RandomForestClassifier …")
    pipeline.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n[TRAIN] ── Model Evaluation ──────────────────────────────")
    print(f"[TRAIN] Accuracy : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"\n[TRAIN] Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=["Did not survive", "Survived"]))

    # ── Save model ────────────────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)

    print(f"[TRAIN] Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train()
