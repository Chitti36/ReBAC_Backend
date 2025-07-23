import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna()

    if 'Access' not in df.columns:
        raise ValueError("Expected 'Access' column for labels")

    X = df.drop('Access', axis=1)
    y = df['Access'].apply(lambda x: 1 if x.lower() == 'yes' else 0)

    return X, y, df.columns.tolist()

def train_decision_tree(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Grid Search for best parameters
    param_grid = {
        'max_depth': [4, 6, 8, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    model = grid.best_estimator_

    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, y_pred)

    return model, auc, X_train.columns.tolist()

'''
def save_model(model, feature_names, model_path="model.joblib"):
    joblib.dump({
        'model': model,
        'features': feature_names
    }, model_path)
    '''
def save_model(model, features, path="backend/model/model.joblib"):
    os.makedirs(os.path.dirname(path), exist_ok=True)  # 🔥 Ensure model folder exists
    joblib.dump({"model": model, "features": features}, path)
