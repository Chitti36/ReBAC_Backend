# 🔧 ReBAC Policy Rule Extractor – Backend

This is the **FastAPI backend** service for extracting interpretable access control rules using a Decision Tree classifier trained on ReBAC-style datasets.

🔗 **Deployable on Render**  
The backend is designed to be easily hosted using [Render.com](https://render.com), and integrates seamlessly with a Streamlit frontend.

## 🚀 Features

- 📥 Accept CSV upload to train access control model
- 🤖 Trains Decision Tree with hyperparameter tuning
- 📜 Extracts IF-THEN rules from trained model
- 🚨 Identifies false positives (misclassified access)



## 📦 Installation

```bash
pip install -r requirements.txt
```


## ▶️ Run the Server Locally

```bash
uvicorn main:app --reload
```

Access the backend at: [http://localhost:8000](http://localhost:8000)

## ⚙️ API Endpoints

| Endpoint                | Method | Description                         |
|------------------------|--------|-------------------------------------|
| `/`                    | GET    | Backend health check                |
| `/train`               | POST   | Upload CSV and train model          |
| `/rules`               | GET    | Get extracted access control rules  |
| `/false_positives`     | GET    | List all detected false positives   |

## 🧪 Sample Input Format

```csv
User_A, User_B, Relation, Resource, Access
1, 0, 1, 0, Yes
0, 1, 0, 1, No
```

- `Access` should be `Yes` or `No`
- Other fields are binary indicators

## 🚀 Deploying on Render (Optional)

To deploy on Render:
1. Connect repo to [Render](https://render.com)
2. Select **Web Service** (Python)
3. Start command:

```bash
uvicorn main:app --host 0.0.0.0 --port 10000
```

## 🔗 Frontend Integration

This backend is consumed by the frontend app at [`rebac-frontend`](https://github.com/your-user/rebac-frontend), which must point to this backend's deployed URL.

## 👤 Author

Developed by **Ruthik Chitti** under academic guidance.
