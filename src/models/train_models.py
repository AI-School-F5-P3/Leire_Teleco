import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def train_and_save_models(x_train, y_train, x_val, y_val, models, output_dir):
    """
    Entrena múltiples modelos con datos de entrenamiento y validación, y los guarda.
    """
    os.makedirs(output_dir, exist_ok=True)

    for name, model in models.items():
        print(f"Entrenando modelo: {name}")
        # Entrenar el modelo
        model.fit(x_train, y_train)

        # Validar el modelo
        y_val_pred = model.predict(x_val)
        acc = accuracy_score(y_val, y_val_pred)
        print(f"Accuracy en validación para {name}: {acc:.2f}")

        # Guardar el modelo entrenado
        model_path = f"{output_dir}/{name.replace(' ', '_').lower()}_model.pkl"
        joblib.dump(model, model_path)
        print(f"Modelo {name} guardado en {model_path}\n")

if __name__ == "__main__":
    # Rutas relativas desde src/models
    train_path = "/home/leire/Documentos/Cursos/F5/Proyectos/refuerzo/repositorio/Leire_Teleco/data/processed/data_split_train.csv"
    val_path = "/home/leire/Documentos/Cursos/F5/Proyectos/refuerzo/repositorio/Leire_Teleco/data/processed/data_split_val.csv"
    output_dir = "/home/leire/Documentos/Cursos/F5/Proyectos/refuerzo/repositorio/Leire_Teleco/models"

    # Modelos a entrenar
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, multi_class='multinomial', solver='lbfgs'),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    # Cargar datos
    print(f"Cargando datos de entrenamiento desde: {train_path}")
    train_data = pd.read_csv(train_path)
    print(f"Columnas en el conjunto de entrenamiento: {train_data.columns.tolist()}")

    print(f"Cargando datos de validación desde: {val_path}")
    val_data = pd.read_csv(val_path)

    # Separar características (X) y etiquetas (y)
    X_train = train_data.drop(columns=["custcat"])
    y_train = train_data["custcat"]
    X_val = val_data.drop(columns=["custcat"])
    y_val = val_data["custcat"]

    # Columnas categóricas a codificar
    categorical_features = ['region', 'marital_label', 'gender_label', 'retire_label']

    # Preparar el codificador para variables categóricas
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    
    # Aplicar One-Hot Encoding a las características categóricas
    encoded_features_train = encoder.fit_transform(X_train[categorical_features])
    encoded_features_val = encoder.transform(X_val[categorical_features])

    # Convertir a DataFrames con nombres de columnas
    encoded_df_train = pd.DataFrame(
        encoded_features_train, 
        columns=encoder.get_feature_names_out(categorical_features)
    )
    encoded_df_val = pd.DataFrame(
        encoded_features_val, 
        columns=encoder.get_feature_names_out(categorical_features)
    )

    # Eliminar columnas categóricas originales y numéricas no deseadas
    numeric_columns = [
        'tenure', 'age', 'address', 'income', 'ed', 'employ', 'reside'
    ]
    X_train_numeric = X_train[numeric_columns]
    X_val_numeric = X_val[numeric_columns]

    # Combinar características numéricas con características codificadas
    X_train_encoded = pd.concat([X_train_numeric, encoded_df_train], axis=1)
    X_val_encoded = pd.concat([X_val_numeric, encoded_df_val], axis=1)

    # Entrenamiento Sin Variables Combinadas
    print("\n--- Entrenando Sin Variables Combinadas ---")
    train_and_save_models(
        X_train_encoded,
        y_train,
        X_val_encoded,
        y_val,
        models,
        f"{output_dir}/no_combined"
    )

    # Entrenamiento Con Variables Combinadas
    print("\n--- Entrenando Con Variables Combinadas ---")
    train_and_save_models(
        X_train_encoded,
        y_train,
        X_val_encoded,
        y_val,
        models,
        f"{output_dir}/combined"
    )
