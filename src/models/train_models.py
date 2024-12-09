import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import mlflow
import mlflow.sklearn

def load_data():
    data = pd.read_csv("../data/processed/data_combined_vif.csv")
    X = data.drop("custcat", axis=1)
    y = data["custcat"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred)

if __name__ == "__main__":
    mlflow.set_experiment("custcat_classification")

    X_train, X_test, y_train, y_test = load_data()

    with mlflow.start_run():
        model = train_model(X_train, y_train)
        
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", model.n_estimators)
        
        evaluation = evaluate_model(model, X_test, y_test)
        mlflow.log_metric("accuracy", model.score(X_test, y_test))
        
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        print(evaluation)