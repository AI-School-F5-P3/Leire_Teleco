import pandas as pd
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_ensemble_model(x_train, y_train, x_val, y_val, output_path):
    """
    Entrena un modelo ensemble (Voting Classifier).

    Args:
        X_train (pd.DataFrame): Características de entrenamiento.
        y_train (pd.Series): Etiquetas de entrenamiento.
        X_val (pd.DataFrame): Características de validación.
        y_val (pd.Series): Etiquetas de validación.
        output_path (str): Ruta para guardar el modelo entrenado.

    Returns:
        None
    """
    # Definir modelos base
    model1 = LogisticRegression(random_state=42, multi_class='multinomial', solver='lbfgs')
    model2 = RandomForestClassifier(random_state=42)
    model3 = GradientBoostingClassifier(random_state=42)

    # Voting Classifier
    ensemble = VotingClassifier(
        estimators=[
            ('lr', model1),
            ('rf', model2),
            ('gb', model3)
        ],
        voting='hard'  # Opción 'soft' si los modelos soportan probabilidades
    )

    print("Entrenando modelo ensemble...")
    ensemble.fit(x_train, y_train)

    # Validar el modelo
    y_val_pred = ensemble.predict(x_val)
    acc = accuracy_score(y_val, y_val_pred)
    print(f"Accuracy en validación: {acc:.2f}")

    # Guardar modelo
    import joblib
    joblib.dump(ensemble, output_path)
    print(f"Modelo ensemble guardado en {output_path}")

if __name__ == "__main__":
    # Cargar datos
    train_path = "../data/processed/data_split_train.csv"
    val_path = "../data/processed/data_split_val.csv"
    train_data = pd.read_csv(train_path)
    val_data = pd.read_csv(val_path)

    x_train = train_data.drop(columns=["custcat"])
    y_train = train_data["custcat"]
    x_val = val_data.drop(columns=["custcat"])
    y_val = val_data["custcat"]

    # Entrenar modelo ensemble
    output_path = "../models/ensemble_model.pkl"
    train_ensemble_model(x_train, y_train, x_val, y_val, output_path)
