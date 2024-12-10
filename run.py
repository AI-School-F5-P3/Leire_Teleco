import subprocess
import sys
from pathlib import Path
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

def run_api():
    """Ejecutar la API FastAPI"""
    try:
        logging.info("Iniciando API...")
        subprocess.Popen([sys.executable, "-m", "uvicorn", "src.api.app:app", "--reload"])
        logging.info("API iniciada correctamente")
    except Exception as e:
        logging.error(f"Error al iniciar API: {e}")

def run_frontend():
    """Ejecutar el frontend Streamlit"""
    try:
        logging.info("Iniciando frontend...")
        subprocess.Popen([sys.executable, "-m", "streamlit", "run", "src/frontend/streamlit_app.py"])
        logging.info("Frontend iniciado correctamente")
    except Exception as e:
        logging.error(f"Error al iniciar frontend: {e}")

if __name__ == "__main__":
    logging.info("Iniciando aplicación...")
    run_api()
    run_frontend()