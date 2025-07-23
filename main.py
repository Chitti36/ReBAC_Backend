from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import pandas as pd
import os

from train_model import load_data, train_decision_tree, save_model
from policy_utils import load_model as load_saved_model, extract_rules, format_rules
from fp_analysis import detect_false_positives

app = FastAPI()

DATA_PATH = "data/uploaded.csv"
MODEL_PATH = "model/model.joblib"

@app.get("/")
def root():
    return {"message": "ReBAC Policy ML Backend is running 🚀"}

@app.post("/train")
async def train(file: UploadFile = File(...)):
    try:
        os.makedirs("data", exist_ok=True)
        os.makedirs("model", exist_ok=True)

        with open(DATA_PATH, "wb") as f:
            contents = await file.read()
            f.write(contents)

        X, y, _ = load_data(DATA_PATH)
        model, auc, _ = train_decision_tree(X, y)
        save_model(model, X.columns.tolist(), MODEL_PATH)

        return {"message": "Trained", "roc_auc": auc}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# ✅ Fixed: Added missing route decorator
@app.get("/rules")
def get_rules():
    try:
        model, features = load_saved_model(MODEL_PATH)
        rules = extract_rules(model, features)
        formatted = format_rules(rules)
        return {"rules": formatted}
    except Exception as e:
        print("Error in /rules:", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/false_positives")
def get_false_positives():
    try:
        df = pd.read_csv(DATA_PATH)
        X, y, _ = load_data(DATA_PATH)
        model, _ = load_saved_model(MODEL_PATH)
        fps = detect_false_positives(model, X, y, df)
        result = fps.to_dict(orient="records")
        return {"false_positives": result}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
