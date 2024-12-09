import pandas as pd

def combine_features(input_path, output_path):
    
    data = pd.read_csv(input_path)

    # Crear variables combinadas
    data["age_tenure_ratio"] = data["age"] / (data["tenure"] + 1)
    data["income_employ_product"] = data["income"] * data["employ"]

    # Guardar dataset con combinaciones
    data.to_csv(output_path, index=False)
    print(f"Datos con variables combinadas guardados en {output_path}")

if __name__ == "__main__":
    input_path = "../data/processed/data_prepared.csv"
    output_path = "../data/processed/data_combined.csv"
    combine_features(input_path, output_path)
