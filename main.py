import subprocess
import logging
from datetime import datetime
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)

def execute_script(script_path):
    """
    Ejecuta un script de Python y maneja errores.

    Args:
        script_path (str): Ruta al archivo del script.
    """
    try:
        logging.info(f"Ejecutando: {script_path}")
        subprocess.run(["python", script_path], check=True)
        logging.info(f"Completado: {script_path}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error al ejecutar {script_path}: {e}")
        return False
    except Exception as e:
        logging.error(f"Error inesperado en {script_path}: {e}")
        return False

def run_pipeline():
    """
    Ejecuta todo el pipeline de machine learning en orden.
    """
    start_time = datetime.now()
    logging.info("Iniciando pipeline de machine learning")

    scripts = [
        "src/data/prepare_data.py",         # Preprocesamiento
        "src/data/feature_engineering.py",   # Variables combinadas
        "src/data/split_data.py",           # División de datos
        "src/models/cross_validation.py",    # Validación cruzada
        "src/models/hyperparameter_tuning.py", # Ajuste de hiperparámetros
        "src/models/train_models.py",        # Entrenamiento de modelos
        "src/models/ensemble_models.py",     # Modelos ensemble
        "src/models/evaluate_models.py",     # Evaluación de modelos
        "src/utils/visualize_tuning.py"      # Visualización de hiperparámetros
    ]

    for script in scripts:
        if not execute_script(script):
            logging.error("Pipeline interrumpido debido a un error")
            break

    end_time = datetime.now()
    duration = end_time - start_time
    logging.info(f"Pipeline completado en {duration}")

if __name__ == "__main__":
    run_pipeline()