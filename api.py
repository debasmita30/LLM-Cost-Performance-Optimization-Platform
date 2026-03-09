import os
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from snowflake_connector import (
    load_results_from_snowflake,
    get_run_history,
    get_best_historical_config,
    SNOWFLAKE_AVAILABLE,
    get_connection
)

app = FastAPI(
    title="LLM Cost-Performance Optimization API",
    description="Backend API for LLM optimization platform with Snowflake integration",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "online", "service": "LLM Optimization API"}

@app.get("/health")
def health():
    return {"status": "online", "version": "1.0"}

@app.get("/snowflake/status")
def snowflake_status():
    if not SNOWFLAKE_AVAILABLE:
        return {"connected": False, "reason": "package not installed"}
    try:
        conn = get_connection()
        conn.close()
        return {"connected": True}
    except Exception as e:
        return {"connected": False, "reason": str(e)}

@app.get("/results")
def get_results():
    df = load_results_from_snowflake()
    if df.empty:
        try:
            df = pd.read_csv("results/simulation_results.csv")
        except Exception:
            return []
    return df.to_dict(orient="records")

@app.get("/run-history")
def run_history():
    df = get_run_history()
    if df.empty:
        return []
    return df.to_dict(orient="records")

@app.get("/best-config")
def best_config():
    return get_best_historical_config()
