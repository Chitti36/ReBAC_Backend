import pandas as pd
import joblib

def load_model(model_path="model.joblib"):
    saved = joblib.load(model_path)
    return saved['model'], saved['features']

def detect_false_positives(model, X, y, original_df):
    preds = model.predict(X)
    false_positives = []

    for i in range(len(y)):
        actual = y.iloc[i]
        predicted = preds[i]

        if actual == 0 and predicted == 1:  # False Positive (Denied → Predicted Allowed)
            sample = X.iloc[i].to_dict()
            sample['Index'] = y.index[i]
            sample['Actual'] = 'DENY'
            sample['Predicted'] = 'ALLOW'
            false_positives.append(sample)

    return pd.DataFrame(false_positives)
