# 📊 Proyecto de Machine Learning: Clasificación Multiclase

¡Bienvenido a este proyecto de Machine Learning! Este repositorio contiene el desarrollo completo de un modelo de clasificación multiclase, desde el preprocesamiento de datos hasta la implementación del modelo entrenado en una API. Aquí encontrarás herramientas para analizar, entrenar, evaluar y monitorear modelos de forma eficiente. 🚀

---

## 🛠️ Estructura del Proyecto


project/
│
├── data/
│   ├── raw/                  # Datos crudos
│   │   └── dataset.csv       
│   └── processed/            # Datos procesados y divididos
│       ├── data_cleaned.csv  
│       ├── data_combined.csv
│       └── data_split.csv
│
├── src/
│   ├── data/                 # Preprocesamiento y división
│   │   ├── prepare_data.py
│   │   ├── feature_engineering.py
│   │   └── split_data.py
│   ├── models/               # Modelos y evaluación
│   │   ├── train_models.py
│   │   ├── evaluate_models.py
│   │   ├── hyperparameter_tuning.py
│   │   ├── ensemble_models.py
│   │   └── cross_validation.py
│   ├── utils/                # Utilidades generales
│   │   ├── visualization.py
│   │   ├── metrics.py
│   │   └── data_drift.py
│   ├── api/                  # Implementación de API
│   │   └── app.py
│   └── monitoring/           # Monitoreo de producción
│       └── monitor_model.py
│
├── notebooks/                # Exploración inicial
│   └── exploratory_data_analysis.ipynb
│
├── tests/                    # Pruebas unitarias
│   ├── test_data.py
│   ├── test_models.py
│   └── test_api.py
│
├── mlruns/                   # Tracking de experimentos con MLflow
│
├── config/                   # Configuraciones generales
│   └── model_config.yaml
│
├── results/                  # Informes y visualizaciones
│   ├── performance_reports/
│   └── visualizations/
│
├── docs/                     # Documentación adicional
│
├── requirements.txt          # Dependencias
├── Dockerfile                # Contenerización
├── README.md                 # Este archivo
└── main.py                   # Orquestador principal
```

---

## ⚙️ Instalación y Configuración

1. **Clonar el Repositorio:**

   git clone https://github.com/tu_usuario/tu_repositorio.git
   cd tu_repositorio
   

2. **Crear un Entorno Virtual:**
   
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   

3. **Instalar Dependencias:**
   
   pip install -r requirements.txt


4. **(Opcional) Configurar MLflow:**
   Asegúrate de que MLflow esté configurado para tracking de experimentos:
   
   mlflow ui


---

## 🚀 Cómo Usar

### 1. **Preprocesar los Datos:**
   
   python src/data/prepare_data.py
   

### 2. **Ingeniería de Características:**
   
   python src/data/feature_engineering.py

### 3. **Dividir los Datos:**
   
   python src/data/split_data.py


### 4. **Entrenar el Modelo:**
   
   python src/models/train_models.py

### 5. **Evaluar el Modelo:**

   python src/models/evaluate_models.py


### 6. **Optimizar Hiperparámetros:**
   
   python src/models/hyperparameter_tuning.py


### 7. **Exponer el Modelo en una API:**

   uvicorn src.api.app:app --reload


---

## 🧪 Funcionalidades Principales

1. **Preprocesamiento de Datos**:
   - Limpieza, normalización y manejo de valores faltantes.
2. **Ingeniería de Características**:
   - Creación de nuevas columnas basadas en combinaciones y transformaciones.
3. **Entrenamiento de Modelos**:
   - Soporte para modelos estándar y ensemble (Random Forest, Gradient Boosting).
4. **Evaluación Detallada**:
   - Métricas como precisión, recall, F1 y matriz de confusión.
5. **Optimización de Hiperparámetros**:
   - Uso de Optuna para búsquedas rápidas y eficientes.
6. **Tracking de Experimentos**:
   - Registro de métricas, parámetros y artefactos con MLflow.
7. **Monitoreo de Producción**:
   - Detección de drift en los datos y monitoreo del rendimiento.
8. **API REST**:
   - Predicciones en tiempo real mediante FastAPI.

---

## 📈 Resultados y Visualizaciones

Los gráficos y resultados del modelo se guardan en `results/`.

- **Gráficos Generados:**
  - Matriz de Confusión.
  - Curva ROC.
  - Análisis de Importancia de Características.

---

## 🛡️ Pruebas

1. **Pruebas de Preprocesamiento:**
   
   pytest tests/test_data.py
   

2. **Pruebas de Modelos:**

   pytest tests/test_models.py

3. **Pruebas de API:**

   pytest tests/test_api.py


---

## 🐳 Docker

Para ejecutar el proyecto en un contenedor:

1. **Construir la Imagen:**
   ```bash
   docker build -t ml_project .
   ```

2. **Ejecutar el Contenedor:**
   ```bash
   docker run -p 8000:8000 ml_project
   ```

---

## 📂 Documentación Adicional

Consulta la carpeta `docs/` para información técnica detallada sobre cada componente del proyecto.

---

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Si tienes ideas o mejoras, no dudes en abrir un issue o enviar un pull request.

---

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo `LICENSE` para más detalles.

