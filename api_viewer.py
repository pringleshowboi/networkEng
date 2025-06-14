from fastapi import FastAPI
from viewer import get_metrics_dataframe, get_decisions_dataframe
import pandas as pd

app = FastAPI()

@app.get("/viewer/metrics")
def metrics():
    df = get_metrics_dataframe()
    return df.to_dict(orient='records')

@app.get("/viewer/decisions")
def decisions():
    df = get_decisions_dataframe()
    return df.to_dict(orient='records')

@app.post("/viewer/save")
def save():
    success = save_data_for_powerbi()
    return {"saved": success}
