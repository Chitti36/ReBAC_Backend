from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import pandas as pd
import os
import shutil

from train_model import load_data, train_decision_tree, save_model
from policy_utils import load_model, extract_rules, format_rules
from fp_analysis import detect_false_positives


app = FastAPI()

DATA_PATH = "backend/data/uploaded.csv"
MODEL_PATH = "backend/model/model.joblib"

@app.get("/")
def root():
    return {"message": "ReBAC Policy ML Backend is running 🚀"}

'''@app.post("/train")
async def train(file: UploadFile = File(...)):
    # Save uploaded file
    with open(DATA_PATH, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Load + train model
    try:
        X, y, feature_names = load_data(DATA_PATH)
        model, auc, features = train_decision_tree(X, y)
        save_model(model, features, MODEL_PATH)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

    return {"message": "Model trained successfully", "roc_auc": round(auc, 4)}
'''
@app.post("/train")
async def train(file: UploadFile = File(...)):
    try:
        # ✅ Make sure 'backend/data' folder exists
        os.makedirs("backend/data", exist_ok=True)

        # ✅ Save uploaded file to backend/data/uploaded.csv
        with open(DATA_PATH, "wb") as f:
            contents = await file.read()
            f.write(contents)

        # ✅ Load, train, save
        X, y, _ = load_data(DATA_PATH)
        model, auc, _ = train_decision_tree(X, y)
        save_model(model, X.columns.tolist(), MODEL_PATH)

        return {"message": "Trained", "roc_auc": auc}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/rules")
def get_rules():
    try:
        model, features = load_saved_model(MODEL_PATH)
        rules = extract_rules(model, features)
        formatted = format_rules(rules)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

    return {"rules": formatted}

@app.get("/false_positives")
def get_false_positives():
    try:
        df = pd.read_csv(DATA_PATH)
        X, y, _ = load_data(DATA_PATH)
        model, _ = load_saved_model(MODEL_PATH)
        fps = detect_false_positives(model, X, y, df)
        result = fps.to_dict(orient="records")
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

    return {"false_positives": result}
