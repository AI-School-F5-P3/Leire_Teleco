import optuna
import matplotlib.pyplot as plt

def visualize_study_results(study_path, model_name):
    """
    Carga un estudio de Optuna y genera visualizaciones.

    Args:
        study_path (str): Ruta al archivo del estudio Optuna (.pkl).
        model_name (str): Nombre del modelo para la visualización.

    Returns:
        None
    """
    # Cargar el estudio
    study = optuna.study.load_study_from_pickle(study_path)

    print(f"Mejores hiperparámetros para {model_name}:")
    print(study.best_params)
    print(f"Mejor puntuación: {study.best_value:.4f}\n")

    # Visualización de la optimización
    fig_optimization = optuna.visualization.plot_optimization_history(study)
    fig_optimization.show()

    # Visualización de importancia de hiperparámetros
    fig_importance = optuna.visualization.plot_param_importances(study)
    fig_importance.show()

if __name__ == "__main__":
    # Estudios a visualizar
    studies = {
        "Logistic Regression": "../results/logistic_regression_study.pkl",
        "Random Forest": "../results/random_forest_study.pkl",
        "Gradient Boosting": "../results/gradient_boosting_study.pkl",
    }

    for model_name, study_path in studies.items():
        visualize_study_results(study_path, model_name)
