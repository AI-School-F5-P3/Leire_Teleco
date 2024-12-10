import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import OneHotEncoder

# Directorio base del proyecto
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

def evaluate_model(test_path, model_path):
    """
    Evalúa un modelo usando datos de test y guarda un informe.
    
    Args:
        test_path (str): Ruta del archivo de test.
        model_path (str): Ruta del modelo entrenado.
    """
    # Asegurar directorios
    os.makedirs(os.path.join(RESULTS_DIR, "visualizations"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "performance_reports"), exist_ok=True)
    
    # Rutas de salida
    plot_path = os.path.join(RESULTS_DIR, "visualizations", f"{os.path.basename(model_path).replace('.pkl', '')}_confusion_matrix.png")
    report_path = os.path.join(RESULTS_DIR, "performance_reports", f"{os.path.basename(model_path).replace('.pkl', '')}_metrics_report.csv")
    
    # Cargar datos de test
    test_data = pd.read_csv(test_path)
    x_test = test_data.drop(columns=["custcat"])
    y_test = test_data["custcat"]
    
    # Preparar características categóricas para la codificación
    categorical_features = ['region', 'marital_label', 'gender_label', 'retire_label']
    
    # Realizar one-hot encoding de las características categóricas
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    
    # Aplicar one-hot encoding a las características categóricas
    encoded_features_test = encoder.fit_transform(x_test[categorical_features])
    
    # Convertir a DataFrame con nombres de columnas
    encoded_df_test = pd.DataFrame(
        encoded_features_test, 
        columns=encoder.get_feature_names_out(categorical_features)
    )
    
    # Seleccionar características numéricas
    numeric_columns = [
        'tenure', 'age', 'address', 'income', 'ed', 'employ', 'reside'
    ]
    x_test_numeric = x_test[numeric_columns]
    
    # Combinar características numéricas con características codificadas
    x_test_encoded = pd.concat([x_test_numeric, encoded_df_test], axis=1)
    
    # Cargar el modelo entrenado
    model = joblib.load(model_path)
    
    # Predicciones
    y_pred = model.predict(x_test_encoded)
    
    # Informe de métricas
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(report_path, index=True)
    print(f"Informe guardado en {report_path}")
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    print("Matriz de Confusión:")
    print(cm)
    
    # Guardar la matriz de confusión como gráfico
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title(f"Matriz de Confusión - {os.path.basename(model_path)}")
    plt.savefig(plot_path)
    print(f"Matriz de confusión guardada en {plot_path}")
    plt.close()  # Cerrar la figura para liberar memoria

    # Determinar si es un modelo con variables combinadas basado en la ruta
    is_combined = "combined" in model_path and "no_combined" not in model_path
    model_type = "Variables Combinadas" if is_combined else "Variables Simples"

    # Generar resumen de modelo
    with open(os.path.join(RESULTS_DIR, "performance_reports", "model_summary.txt"), "a") as summary_file:
        summary_file.write("\n" + "="*50 + "\n")
        summary_file.write(f"Modelo: {os.path.basename(model_path)}\n")
        summary_file.write(f"Tipo: {model_type}\n")
        summary_file.write(f"Accuracy: {report['accuracy']:.4f}\n")
        summary_file.write(f"Macro Avg F1-Score: {report['macro avg']['f1-score']:.4f}\n")
        summary_file.write(f"Weighted Avg F1-Score: {report['weighted avg']['f1-score']:.4f}\n")
        summary_file.write("="*50 + "\n")

if __name__ == "__main__":
    test_path = os.path.join(BASE_DIR, "data", "processed", "data_split_test.csv")
    
    # Limpiar el archivo de resumen anterior si existe
    summary_file_path = os.path.join(RESULTS_DIR, "performance_reports", "model_summary.txt")
    with open(summary_file_path, "w") as summary_file:
        summary_file.write("RESUMEN DE EVALUACIÓN DE MODELOS\n")
        summary_file.write("Fecha: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
    
    # Modelos sin variables combinadas
    simple_model_paths = [
        os.path.join(BASE_DIR, "models", "no_combined", "logistic_regression_model.pkl"),
        os.path.join(BASE_DIR, "models", "no_combined", "random_forest_model.pkl"),
        os.path.join(BASE_DIR, "models", "no_combined", "gradient_boosting_model.pkl")
    ]
    
    # Modelos con variables combinadas
    combined_model_paths = [
        os.path.join(BASE_DIR, "models", "combined", "logistic_regression_model.pkl"),
        os.path.join(BASE_DIR, "models", "combined", "random_forest_model.pkl"),
        os.path.join(BASE_DIR, "models", "combined", "gradient_boosting_model.pkl")
    ]

    # Evaluar modelos sin variables combinadas
    with open(summary_file_path, "a") as summary_file:
        summary_file.write("\nMODELOS CON VARIABLES SIMPLES\n")
        summary_file.write("-" * 30 + "\n")
    
    print("\n=== Evaluación de Modelos sin Variables Combinadas ===")
    for model_path in simple_model_paths:
        print(f"\nEvaluando modelo: {os.path.basename(model_path)}")
        evaluate_model(test_path, model_path)

    # Evaluar modelos con variables combinadas
    with open(summary_file_path, "a") as summary_file:
        summary_file.write("\nMODELOS CON VARIABLES COMBINADAS\n")
        summary_file.write("-" * 30 + "\n")
    
    print("\n=== Evaluación de Modelos con Variables Combinadas ===")
    for model_path in combined_model_paths:
        print(f"\nEvaluando modelo: {os.path.basename(model_path)}")
        evaluate_model(test_path, model_path)