import os
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score
from sklearn.preprocessing import OneHotEncoder

# Configurar el directorio base del proyecto
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def perform_cross_validation(x, y, model, scoring, cv):
    """
    Realiza validación cruzada para un modelo dado.

    Args:
        X (pd.DataFrame): Características.
        y (pd.Series): Etiqueta objetivo.
        model: Modelo a validar.
        scoring: Métrica de evaluación.
        cv: Número de folds en la validación cruzada.

    Returns:
        list: Puntajes de validación cruzada.
    """
    print(f"Realizando validación cruzada con {cv} folds...")
    scores = cross_val_score(model, x, y, scoring=scoring, cv=cv)
    print(f"Puntajes de validación cruzada: {scores}")
    print(f"Puntaje promedio: {scores.mean():.4f}")
    print(f"Desviación estándar: {scores.std():.4f}")
    return scores

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
    
    # Combinar características
    x_prepared = pd.concat([x_numeric, encoded_df], axis=1)
    
    return x_prepared, y

if __name__ == "__main__":
    # Cargar datos usando ruta absoluta
    data_path = os.path.join(BASE_DIR, "data", "processed", "data_split_train.csv")
    print(f"Cargando datos desde: {data_path}")
    data = pd.read_csv(data_path)

    # Preparar datos
    X, y = prepare_data(data)

    # Modelo a validar
    model = RandomForestClassifier(random_state=42)

    # Métrica y configuración de validación cruzada
    scoring = make_scorer(f1_score, average="weighted")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Ejecutar validación cruzada
    scores = perform_cross_validation(X, y, model, scoring, cv)

    # Guardar resultados
    results_dir = os.path.join(BASE_DIR, "results", "cross_validation")
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, "cv_results.txt"), "w") as f:
        f.write("Resultados de Validación Cruzada - Random Forest\n")
        f.write("=============================================\n")
        f.write(f"Número de folds: {cv.n_splits}\n")
        f.write(f"Puntajes individuales: {', '.join([f'{score:.4f}' for score in scores])}\n")
        f.write(f"Puntaje promedio: {scores.mean():.4f}\n")
        f.write(f"Desviación estándar: {scores.std():.4f}\n")
