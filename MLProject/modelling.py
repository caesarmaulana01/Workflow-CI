import argparse
from pathlib import Path
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# === 1. Parsing argumen dari MLflow Project ===
parser = argparse.ArgumentParser()
parser.add_argument("n_estimators", type=int)
parser.add_argument("max_depth", type=int)
parser.add_argument("dataset", type=str)
args = parser.parse_args()

n_estimators = args.n_estimators
max_depth = args.max_depth
dataset_file = args.dataset

# === 2. Set MLflow Tracking ===
os.environ["MLFLOW_TRACKING_USERNAME"] = "caesarmaulana01"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "7d33c8c359b3f97629ce7041aab248242015356f"
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/caesarmaulana01/Membangun_model.mlflow"
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("Employee Attrition Modeling")

# === 3. Init Dagshub ===
dagshub.init(repo_owner='caesarmaulana01', repo_name='mlsystems_employee_attrition', mlflow=True)

# === 4. Load Dataset ===
base_path = Path(__file__).resolve().parent
csv_path = base_path / dataset_file
data = pd.read_csv(csv_path)

X = data.drop(columns=['Attrition'])
y = data['Attrition']

# === 5. Split data ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=7, stratify=y
)

# === 6. Aktifkan autolog ===
mlflow.autolog()

# === 7. Jalankan model (contoh: RandomForest & XGBoost) ===
models = [
    ('RandomForest', RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=7)),
    ('XGBoost', XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, use_label_encoder=False, eval_metric='logloss', random_state=7))
]

for name, model in models:
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)

        mlflow.log_param("model_name", name)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("dataset", dataset_file)

        mlflow.log_metric("manual_precision", precision)
        mlflow.log_metric("manual_recall", recall)

        print(f"[{name}] Precision: {precision:.4f} | Recall: {recall:.4f}")
