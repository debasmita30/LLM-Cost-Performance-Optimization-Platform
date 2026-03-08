import os
import pandas as pd

try:
    import snowflake.connector
    SNOWFLAKE_AVAILABLE = True
except ImportError:
    SNOWFLAKE_AVAILABLE = False


def get_connection():
    return snowflake.connector.connect(
        account=os.environ.get("SNOWFLAKE_ACCOUNT"),
        user=os.environ.get("SNOWFLAKE_USER"),
        password=os.environ.get("SNOWFLAKE_PASSWORD"),
        warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
        database=os.environ.get("SNOWFLAKE_DATABASE", "LLM_OPTIMIZATION"),
        schema=os.environ.get("SNOWFLAKE_SCHEMA", "PUBLIC")
    )


def save_results_to_snowflake(df: pd.DataFrame):
    if not SNOWFLAKE_AVAILABLE:
        print("Snowflake not available — saving to CSV only.")
        return
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS simulation_results (
                model_name        VARCHAR,
                prompt_strategy   VARCHAR,
                temperature       FLOAT,
                accuracy          FLOAT,
                cost_per_request  FLOAT,
                latency_ms        FLOAT,
                objective_score   FLOAT,
                sla_compliant     BOOLEAN,
                run_timestamp     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        for _, row in df.iterrows():
            cursor.execute("""
                INSERT INTO simulation_results
                (model_name, prompt_strategy, temperature, accuracy,
                 cost_per_request, latency_ms, objective_score, sla_compliant)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                row.get("model_name"),
                row.get("prompt_strategy"),
                row.get("temperature"),
                row.get("accuracy"),
                row.get("cost_per_request"),
                row.get("latency_ms"),
                row.get("objective_score"),
                bool(row.get("sla_compliant", False))
            ))
        conn.commit()
        cursor.close()
        conn.close()
        print(f"Saved {len(df)} rows to Snowflake.")
    except Exception as e:
        print(f"Snowflake save failed: {e} — falling back to CSV.")


def load_results_from_snowflake() -> pd.DataFrame:
    if not SNOWFLAKE_AVAILABLE:
        return pd.DataFrame()
    try:
        conn = get_connection()
        df = pd.read_sql("SELECT * FROM simulation_results ORDER BY run_timestamp DESC", conn)
        conn.close()
        return df
    except Exception as e:
        print(f"Snowflake load failed: {e}")
        return pd.DataFrame()
