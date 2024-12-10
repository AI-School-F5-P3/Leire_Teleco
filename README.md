# 📊 Proyecto de Machine Learning: Clasificación Multiclase

¡Bienvenido a este proyecto de Machine Learning! Este repositorio contiene el desarrollo completo de un modelo de clasificación multiclase, desde el preprocesamiento de datos hasta la implementación del modelo entrenado en una API. Aquí encontrarás herramientas para analizar, entrenar, evaluar y monitorear modelos de forma eficiente. 🚀

---
project/
│
├── data/                     # Todos nuestros datos viven aquí
│   ├── raw/                 # Datos sin procesar
│   └── processed/           # Datos listos para el modelado
│
├── src/                      # El corazón de nuestro código
│   ├── data/                # Scripts de procesamiento de datos
│   ├── models/              # Implementación de modelos
│   ├── utils/               # Herramientas útiles
│   ├── api/                 # Nuestra API
│   └── frontend/            # Interfaz de usuario
│
├── notebooks/                # Análisis exploratorio
├── tests/                    # Pruebas unitarias
├── results/                  # Resultados y visualizaciones
└── docs/                     # Documentación

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





---

## 🚀 Cómo Usar

1.Procesa los datos y entrena los modelos:

python3 main.py

2.Inicia la aplicación:

python3 run.py
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

