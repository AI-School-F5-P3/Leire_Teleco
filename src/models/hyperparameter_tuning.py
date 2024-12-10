import os
import joblib
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Configurar el directorio base del proyecto
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def prepare_data(data):
    """
    Prepara los datos aplicando la misma transformación que en train_models.py
    """
    X = data.drop(columns=["custcat"])
    y = data["custcat"]

    # Preparar características categóricas
    categorical_features = ['region', 'marital_label', 'gender_label', 'retire_label']
    
    # One-hot encoding para variables categóricas
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_features = encoder.fit_transform(X[categorical_features])
    
    # Convertir a DataFrame
    encoded_df = pd.DataFrame(
        encoded_features, 
        columns=encoder.get_feature_names_out(categorical_features)
    )
    
    # Seleccionar características numéricas
    numeric_columns = ['tenure', 'age', 'address', 'income', 'ed', 'employ', 'reside']
    x_numeric = X[numeric_columns]
    
    # Escalar características numéricas
    scaler = StandardScaler()
    x_numeric_scaled = scaler.fit_transform(x_numeric)
    x_numeric_scaled = pd.DataFrame(x_numeric_scaled, columns=numeric_columns)
    
    # Combinar características
    x_prepared = pd.concat([x_numeric_scaled, encoded_df], axis=1)
    
    return x_prepared, y

def objective_logistic_regression(trial, x, y):
    C = trial.suggest_float("C", 0.01, 10.0, log=True)
    solver = trial.suggest_categorical("solver", ["lbfgs", "saga"])
    max_iter = trial.suggest_int("max_iter", 500, 1000)

    model = LogisticRegression(
        random_state=42,
        solver=solver,
        C=C,
        max_iter=max_iter,
    )

    scores = cross_val_score(
        model, x, y, cv=5, scoring=make_scorer(f1_score, average="weighted")
    )
    return scores.mean()

def objective_random_forest(trial, x, y):
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 5, 50)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42,
    )

    scores = cross_val_score(
        model, x, y, cv=5, scoring=make_scorer(f1_score, average="weighted")
    )
    return scores.mean()

def objective_gradient_boosting(trial, x, y):
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, step=0.01)
    max_depth = trial.suggest_int("max_depth", 3, 50)

    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=42,
    )

    scores = cross_val_score(
        model, x, y, cv=5, scoring=make_scorer(f1_score, average="weighted")
    )
    return scores.mean()

if __name__ == "__main__":
    # Cargar y preparar datos
    train_path = os.path.join(BASE_DIR, "data", "processed", "data_split_train.csv")
    print(f"Cargando datos desde: {train_path}")
    train_data = pd.read_csv(train_path)
    
    # Preparar datos
    X, y = prepare_data(train_data)

    # Crear directorio para resultados
    results_dir = os.path.join(BASE_DIR, "results", "hyperparameter_tuning")
    os.makedirs(results_dir, exist_ok=True)

    # Optimización para Logistic Regression
    print("Optimizando Logistic Regression...")
    lr_study = optuna.create_study(direction="maximize")
    lr_study.optimize(lambda trial: objective_logistic_regression(trial, X, y), n_trials=50)
    print("Mejores hiperparámetros para Logistic Regression:")
    print(lr_study.best_params)
    print(f"Mejor puntuación: {lr_study.best_value:.4f}")

    # Optimización para Random Forest
    print("\nOptimizando Random Forest...")
    rf_study = optuna.create_study(direction="maximize")
    rf_study.optimize(lambda trial: objective_random_forest(trial, X, y), n_trials=50)
    print("Mejores hiperparámetros para Random Forest:")
    print(rf_study.best_params)
    print(f"Mejor puntuación: {rf_study.best_value:.4f}")

    # Optimización para Gradient Boosting
    print("\nOptimizando Gradient Boosting...")
    gb_study = optuna.create_study(direction="maximize")
    gb_study.optimize(lambda trial: objective_gradient_boosting(trial, X, y), n_trials=50)
    print("Mejores hiperparámetros para Gradient Boosting:")
    print(gb_study.best_params)
    print(f"Mejor puntuación: {gb_study.best_value:.4f}")

    # Guardar resultados
    joblib.dump(lr_study, os.path.join(results_dir, "logistic_regression_study.pkl"))
    joblib.dump(rf_study, os.path.join(results_dir, "random_forest_study.pkl"))
    joblib.dump(gb_study, os.path.join(results_dir, "gradient_boosting_study.pkl"))

    # Guardar resumen en archivo de texto
    with open(os.path.join(results_dir, "optimization_results.txt"), "w") as f:
        f.write("RESULTADOS DE OPTIMIZACIÓN DE HIPERPARÁMETROS\n")
        f.write("===========================================\n\n")
        
        f.write("Logistic Regression\n")
        f.write("-----------------\n")
        f.write(f"Mejores hiperparámetros: {lr_study.best_params}\n")
        f.write(f"Mejor puntuación: {lr_study.best_value:.4f}\n\n")
        
        f.write("Random Forest\n")
        f.write("------------\n")
        f.write(f"Mejores hiperparámetros: {rf_study.best_params}\n")
        f.write(f"Mejor puntuación: {rf_study.best_value:.4f}\n\n")
        
        f.write("Gradient Boosting\n")
        f.write("----------------\n")
        f.write(f"Mejores hiperparámetros: {gb_study.best_params}\n")
        f.write(f"Mejor puntuación: {gb_study.best_value:.4f}\n")

    print(f"\nResultados guardados en {results_dir}")