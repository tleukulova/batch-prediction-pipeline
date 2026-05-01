


"""
create_database.py
------------------
Creates the SQLite database, defines the schema,
and loads sample Titanic rows into the input_data table.
 
Tables created:
  - input_data   : passenger features (no target column)
  - predictions  : model output + timestamp
"""
 
import sqlite3
import pandas as pd
import os
 
# ── Paths ────────────────────────────────────────────────────────────────────
DB_PATH  = "database/predictions.db"
CSV_PATH = "data/titanic.csv"
 
# ── Feature columns loaded into the database ─────────────────────────────────
FEATURE_COLUMNS = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
 
 
def create_tables(conn: sqlite3.Connection) -> None:
    """Create input_data and predictions tables if they do not exist."""
    cursor = conn.cursor()
 
    # input_data: one row per passenger, no target column
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS input_data (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            Pclass   INTEGER,
            Sex      TEXT,
            Age      REAL,
            SibSp    INTEGER,
            Parch    INTEGER,
            Fare     REAL,
            Embarked TEXT
        )
    """)
 
    # predictions: stores model output linked to input row id
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            input_id             INTEGER UNIQUE,
            prediction           INTEGER,
            prediction_timestamp TEXT,
            FOREIGN KEY (input_id) REFERENCES input_data(id)
        )
    """)
 
    conn.commit()
    print("[DB] Tables 'input_data' and 'predictions' are ready.")
 
 
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename DataFrame columns to match expected capitalisation.
    Works whether the CSV uses 'Pclass', 'pclass', or 'PCLASS'.
    """
    col_map = {col.lower(): col for col in df.columns}
    rename = {}
    for expected in FEATURE_COLUMNS:
        lower = expected.lower()
        if lower in col_map and col_map[lower] != expected:
            rename[col_map[lower]] = expected
    if rename:
        print(f"[DB] Renaming columns to match expected names: {rename}")
        df = df.rename(columns=rename)
    return df
 
 
def load_input_data(conn: sqlite3.Connection) -> None:
    """Load Titanic CSV into input_data table (features only, no target)."""
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"Dataset not found at '{CSV_PATH}'.\n"
            "Please download titanic.csv and place it in the data/ folder.\n"
            "Source: https://raw.githubusercontent.com/datasciencedboys/"
            "datasets/master/titanic.csv"
        )
 
    df = pd.read_csv(CSV_PATH, sep=None, engine="python")
 
    # Show actual columns so the user can debug if needed
    print(f"[DB] Columns found in CSV: {df.columns.tolist()}")
 
    # Normalise column names (handles lowercase / mixed-case CSV files)
    df = normalize_columns(df)
 
    # Verify all required columns exist after normalisation
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise KeyError(
            f"Required columns missing from CSV: {missing}\n"
            f"Columns available: {df.columns.tolist()}"
        )
 
    # Keep only the feature columns we need
    df = df[FEATURE_COLUMNS].copy()
 
    # Check if table already has data — avoid duplicates on re-run
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM input_data")
    row_count = cursor.fetchone()[0]
 
    if row_count > 0:
        print(f"[DB] input_data already contains {row_count} rows. Skipping insert.")
        return
 
    # Write features to database
    df.to_sql("input_data", conn, if_exists="append", index=False)
    print(f"[DB] Loaded {len(df)} passenger records into input_data.")
 
 
def main() -> None:
    # Create database/ directory if it does not exist
    os.makedirs("database", exist_ok=True)
 
    print("[DB] Connecting to database …")
    conn = sqlite3.connect(DB_PATH)
 
    create_tables(conn)
    load_input_data(conn)
 
    conn.close()
    print(f"[DB] Database ready at: {DB_PATH}")
 
 
if __name__ == "__main__":
    main()