import argparse
from pathlib import Path
import pandas as pd
import mlflow
import mlflow.sklearn
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def main():
    # === 1. Parsing arguments ===
    parser = argparse.ArgumentParser()
    parser.add_argument("n_estimators", type=int)
    parser.add_argument("max_depth", type=int)
    parser.add_argument("dataset", type=str)
    args = parser.parse_args()

    n_estimators = args.n_estimators
    max_depth = args.max_depth
    dataset_file = args.dataset
    
    # === 3. Load Dataset ===
    base_path = Path(__file__).resolve().parent
    csv_path = base_path / dataset_file
    
    if not csv_path.exists():
        csv_path = base_path.parent / dataset_file
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset not found at: {csv_path}")
    
    print(f"Loading dataset from: {csv_path}")
    data = pd.read_csv(csv_path)

    X = data.drop(columns=['Attrition'])
    y = data['Attrition']

    # === 4. Split data ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=7, stratify=y
    )

    # === 5. Enable autologging ===
    mlflow.sklearn.autolog()

    # === 6. Run models ===
    models = [
        ('RandomForest', RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=7)),
        ('Logistic Regression', LogisticRegression(
            solver='liblinear',
            random_state=7,
            class_weight='balanced'))
    ]

    for name, model in models:
        with mlflow.start_run(run_name=name):
            print(f"\n=== Training {name} ===")
            
            # Log parameters
            mlflow.log_params({
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "model_type": name
            })

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                input_example=input_example
            )
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            
            # Log metrics
            mlflow.log_metrics({
                "precision": precision,
                "recall": recall
            })
            
            print(f"{name} - Precision: {precision:.4f}, Recall: {recall:.4f}")

if __name__ == "__main__":
    main()