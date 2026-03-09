import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

try:
    import snowflake.connector
    from snowflake.connector.pandas_tools import write_pandas
    SNOWFLAKE_AVAILABLE = True
except ImportError:
    SNOWFLAKE_AVAILABLE = False
    print("snowflake-connector-python not installed — running in CSV-only mode.")


SNOWFLAKE_CONFIG = {
    "account":   os.environ.get("SNOWFLAKE_ACCOUNT"),
    "user":      os.environ.get("SNOWFLAKE_USER"),
    "password":  os.environ.get("SNOWFLAKE_PASSWORD"),
    "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
    "database":  os.environ.get("SNOWFLAKE_DATABASE", "LLM_OPTIMIZATION"),
    "schema":    os.environ.get("SNOWFLAKE_SCHEMA", "PUBLIC"),
}


def get_connection():
    if not all([SNOWFLAKE_CONFIG["account"], SNOWFLAKE_CONFIG["user"], SNOWFLAKE_CONFIG["password"]]):
        raise ValueError("Missing Snowflake credentials in .env file.")
    return snowflake.connector.connect(**SNOWFLAKE_CONFIG)


def initialize_schema():
    if not SNOWFLAKE_AVAILABLE:
        return
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {SNOWFLAKE_CONFIG['database']}")
        cursor.execute(f"USE DATABASE {SNOWFLAKE_CONFIG['database']}")
        cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {SNOWFLAKE_CONFIG['schema']}")
        cursor.execute(f"USE SCHEMA {SNOWFLAKE_CONFIG['schema']}")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS simulation_results (
                id                INTEGER AUTOINCREMENT PRIMARY KEY,
                model_name        VARCHAR(100),
                prompt_strategy   VARCHAR(100),
                temperature       FLOAT,
                accuracy          FLOAT,
                cost_per_request  FLOAT,
                latency_ms        FLOAT,
                objective_score   FLOAT,
                sla_compliant     BOOLEAN,
                run_timestamp     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS optimization_runs (
                run_id             INTEGER AUTOINCREMENT PRIMARY KEY,
                run_timestamp      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_configs      INTEGER,
                pareto_configs     INTEGER,
                best_accuracy      FLOAT,
                min_cost           FLOAT,
                sla_compliance_pct FLOAT,
                notes              VARCHAR(500)
            )
        """)
        conn.commit()
        cursor.close()
        conn.close()
        print("Snowflake schema initialized.")
    except Exception as e:
        print(f"Schema init failed: {e}")


def save_results_to_snowflake(df: pd.DataFrame, notes: str = "") -> bool:
    if not SNOWFLAKE_AVAILABLE:
        print("Snowflake unavailable — saved to CSV only.")
        return False
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(f"USE DATABASE {SNOWFLAKE_CONFIG['database']}")
        cursor.execute(f"USE SCHEMA {SNOWFLAKE_CONFIG['schema']}")
        for _, row in df.iterrows():
            cursor.execute("""
                INSERT INTO simulation_results
                (model_name, prompt_strategy, temperature, accuracy,
                 cost_per_request, latency_ms, objective_score, sla_compliant)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                str(row.get("model_name", "unknown")),
                str(row.get("prompt_strategy", "default")),
                float(row.get("temperature", 0.0)),
                float(row.get("accuracy", 0.0)),
                float(row.get("cost_per_request", 0.0)),
                float(row.get("latency_ms", 0.0)) if pd.notna(row.get("latency_ms")) else None,
                float(row.get("objective_score", 0.0)),
                bool(row.get("sla_compliant", False))
            ))
        sla_pct = df["sla_compliant"].mean() * 100 if "sla_compliant" in df.columns else 0
        cursor.execute("""
            INSERT INTO optimization_runs
            (total_configs, pareto_configs, best_accuracy, min_cost, sla_compliance_pct, notes)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            len(df),
            int(df["sla_compliant"].sum()) if "sla_compliant" in df.columns else 0,
            float(df["accuracy"].max()),
            float(df["cost_per_request"].min()),
            round(sla_pct, 2),
            notes or f"Auto-run at {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        ))
        conn.commit()
        cursor.close()
        conn.close()
        print(f"Saved {len(df)} rows to Snowflake successfully.")
        return True
    except Exception as e:
        print(f"Snowflake save failed: {e} — falling back to CSV.")
        return False


def load_results_from_snowflake(limit: int = 500) -> pd.DataFrame:
    if not SNOWFLAKE_AVAILABLE:
        return pd.DataFrame()
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(f"USE DATABASE {SNOWFLAKE_CONFIG['database']}")
        cursor.execute(f"USE SCHEMA {SNOWFLAKE_CONFIG['schema']}")
        df = pd.read_sql(f"""
            SELECT * FROM simulation_results
            ORDER BY run_timestamp DESC
            LIMIT {limit}
        """, conn)
        conn.close()
        print(f"Loaded {len(df)} records from Snowflake.")
        return df
    except Exception as e:
        print(f"Snowflake load failed: {e}")
        return pd.DataFrame()


def get_run_history() -> pd.DataFrame:
    if not SNOWFLAKE_AVAILABLE:
        return pd.DataFrame()
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(f"USE DATABASE {SNOWFLAKE_CONFIG['database']}")
        cursor.execute(f"USE SCHEMA {SNOWFLAKE_CONFIG['schema']}")
        df = pd.read_sql("""
            SELECT * FROM optimization_runs
            ORDER BY run_timestamp DESC
        """, conn)
        conn.close()
        return df
    except Exception as e:
        print(f"Run history load failed: {e}")
        return pd.DataFrame()


def get_best_historical_config() -> dict:
    if not SNOWFLAKE_AVAILABLE:
        return {}
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(f"USE DATABASE {SNOWFLAKE_CONFIG['database']}")
        cursor.execute(f"USE SCHEMA {SNOWFLAKE_CONFIG['schema']}")
        df = pd.read_sql("""
            SELECT * FROM simulation_results
            WHERE sla_compliant = TRUE
            ORDER BY objective_score DESC
            LIMIT 1
        """, conn)
        conn.close()
        return df.iloc[0].to_dict() if not df.empty else {}
    except Exception as e:
        print(f"Best config query failed: {e}")
        return {}


initialize_schema()
