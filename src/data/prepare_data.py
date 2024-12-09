from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os

# Cargar datos
data = pd.read_csv('/home/leire/Documentos/Cursos/F5/Proyectos/refuerzo/repositorio/Leire_Teleco/data/processed/data_cleaned.csv')

# Separar numéricas y categóricas
numeric_cols = data.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = data.select_dtypes(include=["object", "category"]).columns

# Imputar columnas numéricas con la media
numeric_imputer = SimpleImputer(strategy='mean')
data[numeric_cols] = numeric_imputer.fit_transform(data[numeric_cols])

# Imputar columnas categóricas con la moda
categorical_imputer = SimpleImputer(strategy='most_frequent')
data[categorical_cols] = categorical_imputer.fit_transform(data[categorical_cols])

print("Valores faltantes imputados.")

# Normalización
scaler = StandardScaler()
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

# Guardado
processed_path = "../data/processed/"
os.makedirs(processed_path, exist_ok=True)

# Guarda el archivo
file_path = os.path.join(processed_path, "data_prepared.csv")
data.to_csv(file_path, index=False)

print(f"Datos procesados guardados en: {file_path}")