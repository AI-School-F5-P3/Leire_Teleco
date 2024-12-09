import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

def combine_features(data):
    """
    Crea variables combinadas en el dataset.
    """
    data["age_tenure_ratio"] = data["age"] / (data["tenure"] + 1)
    data["income_employ_product"] = data["income"] * data["employ"]
    return data

def calculate_vif(data, numeric_cols, threshold=10):
    """
    Calcula el VIF y elimina variables con VIF > threshold.
    """
    vif_data = pd.DataFrame()
    vif_data["feature"] = numeric_cols
    vif_data["VIF"] = [variance_inflation_factor(data[numeric_cols].values, i) for i in range(len(numeric_cols))]

    high_vif_features = vif_data[vif_data["VIF"] > threshold]["feature"].tolist()
    print(f"Eliminando variables con VIF > {threshold}: {high_vif_features}")

    data_cleaned = data.drop(columns=high_vif_features)
    return data_cleaned, vif_data

if __name__ == "__main__":
    input_path = "../data/processed/data_prepared.csv"
    output_path = "../data/processed/data_combined_vif.csv"

    # Cargar datos
    data = pd.read_csv(input_path)

    # Crear variables combinadas
    data = combine_features(data)

    # Calcular y filtrar VIF
    numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns.tolist()
    data_cleaned, vif_table = calculate_vif(data, numeric_cols)

    # Guardar resultados
    data_cleaned.to_csv(output_path, index=False)
    vif_table.to_csv("../results/vif_analysis.csv", index=False)
    print(f"Datos procesados guardados en {output_path}")

