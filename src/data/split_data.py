import os
import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(input_path, train_path, val_path, test_path, val_size=0.2, test_size=0.2, random_state=42):
    """
    Divide los datos en entrenamiento, validación y test.
    """
    # Verificar si el archivo de entrada existe
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"El archivo {input_path} no existe. Verifica la ruta o ejecuta el preprocesamiento primero.")
    
    # Cargar datos
    print(f"Cargando datos desde: {input_path}")
    data = pd.read_csv(input_path)
    print(f"Datos cargados: {data.shape[0]} filas, {data.shape[1]} columnas.")
    
    # División inicial en entrenamiento y test
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    
    # División adicional del entrenamiento en entrenamiento y validación
    train_data, val_data = train_test_split(train_data, test_size=val_size, random_state=random_state)

    # Crear carpetas de salida si no existen
    os.makedirs(os.path.dirname(train_path), exist_ok=True)

    # Guardar conjuntos
    train_data.to_csv(train_path, index=False)
    val_data.to_csv(val_path, index=False)
    test_data.to_csv(test_path, index=False)

    print(f"Datos divididos y guardados en: {train_path}, {val_path}, {test_path}")

if __name__ == "__main__":
    # Ruta absoluta del archivo de entrada
    input_path = "/home/leire/Documentos/Cursos/F5/Proyectos/refuerzo/repositorio/Leire_Teleco/data/processed/data_cleaned.csv"

    # Rutas de salida
    train_path = "/home/leire/Documentos/Cursos/F5/Proyectos/refuerzo/repositorio/Leire_Teleco/data/processed/data_split_train.csv"
    val_path = "/home/leire/Documentos/Cursos/F5/Proyectos/refuerzo/repositorio/Leire_Teleco/data/processed/data_split_val.csv"
    test_path = "/home/leire/Documentos/Cursos/F5/Proyectos/refuerzo/repositorio/Leire_Teleco/data/processed/data_split_test.csv"

    split_data(input_path, train_path, val_path, test_path)
