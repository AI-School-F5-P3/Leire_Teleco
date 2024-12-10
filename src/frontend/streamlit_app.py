import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from pathlib import Path
import sys

# Añadir el directorio src al path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.utils.visualization import plot_feature_importance
from src.utils.metrics import calculate_model_metrics

def main():
    st.set_page_config(page_title="Telco Customer Predictor", layout="wide")
    
    st.title("Telco Customer Category Prediction")
    
    # Sidebar con opciones
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Predict", "Model Performance", "Data Analysis"])
    
    if page == "Predict":
        show_prediction_page()
    elif page == "Model Performance":
        show_performance_page()
    else:
        show_analysis_page()

def show_prediction_page():
    st.header("Make a Prediction")
    
    # Formulario de entrada
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            region = st.text_input("Region")
            tenure = st.number_input("Tenure", min_value=0.0)
            age = st.number_input("Age", min_value=0)
            marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            
        with col2:
            income = st.number_input("Income", min_value=0.0)
            gender = st.selectbox("Gender", ["Male", "Female"])
            education = st.number_input("Education Level", min_value=0)
            
        submitted = st.form_submit_button("Predict")
        
        if submitted:
            # Preparar datos para la API
            data = {
                "region": region,
                "tenure": tenure,
                "age": age,
                "marital": marital,
                "income": income,
                "gender": gender,
                "ed": education
            }
            
            # Hacer predicción
            response = requests.post("http://localhost:8000/predict", json=data)
            
            if response.status_code == 200:
                result = response.json()
                st.success(f"Predicted Category: {result['prediction']}")
                st.info(f"Prediction Probability: {result['probability']:.2f}")
            else:
                st.error("Error making prediction")

def show_performance_page():
    st.header("Model Performance Metrics")
    
    # Cargar métricas del modelo
    metrics = calculate_model_metrics()  # Función que debes implementar
    
    # Mostrar métricas
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{metrics['accuracy']:.2f}")
    col2.metric("F1 Score", f"{metrics['f1_score']:.2f}")
    col3.metric("ROC AUC", f"{metrics['roc_auc']:.2f}")
    
    # Mostrar matriz de confusión
    st.subheader("Confusion Matrix")
    st.plotly_chart(metrics['confusion_matrix'])

def show_analysis_page():
    st.header("Data Analysis")
    
    # Cargar datos procesados
    data = pd.read_csv(project_root / "data" / "processed" / "data_cleaned.csv")
    
    # Visualizaciones
    st.subheader("Feature Distributions")
    feature = st.selectbox("Select Feature", data.columns)
    fig = px.histogram(data, x=feature)
    st.plotly_chart(fig)
    
    # Importancia de características
    st.subheader("Feature Importance")
    importance_plot = plot_feature_importance()  # Función que debes implementar
    st.plotly_chart(importance_plot)

if __name__ == "__main__":
    main()