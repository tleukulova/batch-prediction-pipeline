"""
batch_predict.py
----------------
Core batch prediction script:
  1. Connects to the SQLite database.
  2. Reads all rows from input_data that have NOT yet been predicted.
  3. Loads the trained model pipeline from disk.
  4. Generates predictions.
  5. Writes results (prediction + timestamp) into the predictions table.

Duplicate prevention:
  The predictions table has a UNIQUE constraint on input_id.
  This script only reads input rows that have no matching entry in predictions,
  so re-running the script is always safe — existing predictions are never
  overwritten or duplicated.
"""

import sqlite3
import pickle
import pandas as pd
from datetime import datetime, timezone

# ── Paths ────────────────────────────────────────────────────────────────────
DB_PATH    = "database/predictions.db"
MODEL_PATH = "models/titanic_model.pkl"

# ── Feature columns (must match train_model.py) ───────────────────────────────
FEATURE_COLUMNS = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]


def load_unpredicted_data(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Return rows from input_data that do not yet have a prediction.
    Uses a LEFT JOIN so only truly new rows are returned.
    """
    query = """
        SELECT i.id, i.Pclass, i.Sex, i.Age, i.SibSp, i.Parch, i.Fare, i.Embarked
        FROM   input_data AS i
        LEFT JOIN predictions AS p ON i.id = p.input_id
        WHERE  p.input_id IS NULL
    """
    df = pd.read_sql_query(query, conn)
    return df


def load_model(path: str):
    """Load the trained scikit-learn pipeline from a pickle file."""
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


def save_predictions(conn: sqlite3.Connection, input_ids: list, preds: list) -> None:
    """Insert prediction rows into the predictions table."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    cursor = conn.cursor()

    rows = [(int(iid), int(pred), timestamp) for iid, pred in zip(input_ids, preds)]

    cursor.executemany("""
        INSERT OR IGNORE INTO predictions (input_id, prediction, prediction_timestamp)
        VALUES (?, ?, ?)
    """, rows)

    conn.commit()
    return len(rows)


def run_batch_prediction() -> None:
    """Execute one full batch prediction cycle."""
    print(f"\n[BATCH] ── Starting batch prediction run ──────────────────────")
    print(f"[BATCH] Time : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── Connect to database ────────────────────────────────────────────────
    conn = sqlite3.connect(DB_PATH)

    # ── Load only unpredicted rows ─────────────────────────────────────────
    df = load_unpredicted_data(conn)

    if df.empty:
        print("[BATCH] No new rows to predict. All input_data rows are already predicted.")
        conn.close()
        return

    print(f"[BATCH] Found {len(df)} unpredicted row(s). Running model …")

    # Separate ids from features
    input_ids = df["id"].tolist()
    X = df[FEATURE_COLUMNS]

    # ── Load model ─────────────────────────────────────────────────────────
    model = load_model(MODEL_PATH)

    # ── Generate predictions ───────────────────────────────────────────────
    predictions = model.predict(X).tolist()

    survived_count     = sum(predictions)
    not_survived_count = len(predictions) - survived_count
    print(f"[BATCH] Predictions: {survived_count} survived | {not_survived_count} did not survive")

    # ── Save to database ───────────────────────────────────────────────────
    inserted = save_predictions(conn, input_ids, predictions)
    print(f"[BATCH] Saved {inserted} prediction(s) to database.")

    conn.close()
    print(f"[BATCH] ── Batch prediction complete ──────────────────────────\n")


if __name__ == "__main__":
    run_batch_prediction()
