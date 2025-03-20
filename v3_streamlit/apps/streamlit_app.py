#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Aplicación Streamlit para visualizar los resultados del análisis de 
indicadores económicos: Tipo de Cambio, Tasa de Interés e Inflación.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuración de la página
st.set_page_config(
    page_title="Análisis Económico: Tipo de Cambio, Tasa de Interés e Inflación",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título y descripción
st.title("📊 Análisis de Indicadores Económicos Mexicanos")
st.markdown("""
Esta aplicación visualiza la relación entre tres importantes indicadores económicos de México:
- 💱 **Tipo de cambio PESO/USD**
- 💹 **Tasa de interés**
- 📈 **Inflación**

Los datos provienen del Banco de México y el INEGI, procesados a través de un ETL y almacenados en AWS.
""")

# Cargar datos
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('indicadores_economicos.csv')
        # Convertir la columna de fecha
        df['date'] = pd.to_datetime(df['date'])
        # Asegurar que todas las columnas numéricas sean float64 para compatibilidad con Arrow
        for col in ['tipo_de_cambio', 'tasa_de_interes', 'inflacion']:
            df[col] = df[col].astype('float64')
        return df
    except FileNotFoundError:
        st.error("No se encontró el archivo de datos. Por favor ejecuta el análisis primero.")
        return None

@st.cache_data
def load_results():
    try:
        df = pd.read_csv('resultados_regresiones.csv')
        # Asegurar que las columnas numéricas sean float64
        for col in ['Coeficiente', 'Intercepto', 'R²', 'MSE']:
            df[col] = df[col].astype('float64')
        return df
    except FileNotFoundError:
        return None

# Cargar datos
df = load_data()
resultados = load_results()

if df is not None:
    # Sidebar para filtrar por fechas
    st.sidebar.header("Filtros")
    
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    
    start_date = st.sidebar.date_input("Fecha de inicio", min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("Fecha de fin", max_date, min_value=min_date, max_value=max_date)
    
    # Filtrar datos
    mask = (df['date'] >= pd.Timestamp(start_date)) & (df['date'] <= pd.Timestamp(end_date))
    filtered_df = df.loc[mask].copy()  # Usar copy() para evitar SettingWithCopyWarning
    
    # Mostrar datos filtrados
    with st.expander("📋 Ver datos"):
        st.dataframe(filtered_df)
        
        # Botón para descargar datos
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Descargar datos como CSV",
            data=csv,
            file_name="indicadores_economicos_filtrados.csv",
            mime="text/csv",
        )
    
    # Dividir la página en pestañas
    tab1, tab2, tab3 = st.tabs(["📈 Series de Tiempo", "🔄 Correlaciones", "📊 Regresiones"])
    
    with tab1:
        st.header("Series de Tiempo")
        
        # Crear gráfico de series de tiempo con Plotly
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                           subplot_titles=("Tipo de Cambio PESO/USD", "Tasa de Interés (%)", "Inflación (%)"))
        
        # Tipo de cambio
        fig.add_trace(go.Scatter(
            x=filtered_df['date'], 
            y=filtered_df['tipo_de_cambio'],
            mode='lines',
            name='Tipo de Cambio',
            line=dict(color='blue')
        ), row=1, col=1)
        
        # Tasa de interés
        fig.add_trace(go.Scatter(
            x=filtered_df['date'], 
            y=filtered_df['tasa_de_interes'],
            mode='lines',
            name='Tasa de Interés',
            line=dict(color='red')
        ), row=2, col=1)
        
        # Inflación
        fig.add_trace(go.Scatter(
            x=filtered_df['date'], 
            y=filtered_df['inflacion'],
            mode='lines',
            name='Inflación',
            line=dict(color='green')
        ), row=3, col=1)
        
        # Actualizar diseño
        fig.update_layout(height=800, width=1000, showlegend=False)
        
        # Mostrar gráfico
        st.plotly_chart(fig, use_container_width=True)
        
        # Estadísticas descriptivas
        st.subheader("Estadísticas Descriptivas")
        st.dataframe(filtered_df.describe())
    
    with tab2:
        st.header("Correlaciones entre Variables")
        
        # Calcular matriz de correlación
        corr = filtered_df[['tipo_de_cambio', 'tasa_de_interes', 'inflacion']].corr()
        
        # Mostrar matriz de correlación
        fig_corr = px.imshow(
            corr, 
            text_auto=True, 
            color_continuous_scale='RdBu_r',
            labels=dict(color="Correlación")
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Explicación de los resultados
        st.subheader("Interpretación de Correlaciones")
        st.markdown("""
        La matriz de correlación muestra la fuerza y dirección de la relación lineal entre las variables:
        
        - **Correlación positiva (cercana a 1)**: Indica que cuando una variable aumenta, la otra también tiende a aumentar.
        - **Correlación negativa (cercana a -1)**: Indica que cuando una variable aumenta, la otra tiende a disminuir.
        - **Correlación cercana a 0**: Indica poca o ninguna relación lineal entre las variables.
        """)
    
    with tab3:
        st.header("Regresiones Lineales")
        
        # Seleccionar variables para la regresión
        st.subheader("Crear Regresión Personalizada")
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_var = st.selectbox(
                "Variable independiente (X):",
                options=['tipo_de_cambio', 'tasa_de_interes', 'inflacion'],
                index=0
            )
        
        with col2:
            y_var = st.selectbox(
                "Variable dependiente (Y):",
                options=['tipo_de_cambio', 'tasa_de_interes', 'inflacion'],
                index=1
            )
        
        if x_var != y_var:
            # Función para realizar y mostrar la regresión
            def show_regression(x, y, x_label, y_label):
                # Convertir a numpy arrays
                x_arr = np.array(x).reshape(-1, 1)
                y_arr = np.array(y)
                
                # Crear y ajustar el modelo
                model = LinearRegression()
                model.fit(x_arr, y_arr)
                
                # Predecir valores
                y_pred = model.predict(x_arr)
                
                # Calcular métricas
                r2 = r2_score(y_arr, y_pred)
                mse = mean_squared_error(y_arr, y_pred)
                
                # Crear gráfico con Plotly
                fig = px.scatter(
                    filtered_df, x=x, y=y,
                    labels={x: x_label, y: y_label},
                    title=f'Regresión: {y_label} ~ {x_label} (R² = {r2:.4f})'
                )
                
                # Añadir línea de regresión
                sorted_x = filtered_df[x].sort_values()
                fig.add_traces(
                    go.Scatter(
                        x=sorted_x,
                        y=model.predict(sorted_x.values.reshape(-1, 1)),
                        mode='lines',
                        name='Regresión',
                        line=dict(color='red', width=2)
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Mostrar ecuación y métricas
                st.markdown(f"""
                **Ecuación de regresión:**  
                {y_label} = {model.coef_[0]:.4f} × {x_label} + {model.intercept_:.4f}
                
                **Métricas del modelo:**
                - **R²:** {r2:.4f} (Porcentaje de varianza explicada)
                - **Error cuadrático medio:** {mse:.4f}
                - **Coeficiente:** {model.coef_[0]:.4f}
                - **Intercepto:** {model.intercept_:.4f}
                """)
                
                return r2, mse, model.coef_[0], model.intercept_
            
            # Mapeo de nombres para etiquetas
            var_labels = {
                'tipo_de_cambio': 'Tipo de Cambio (MXN/USD)',
                'tasa_de_interes': 'Tasa de Interés (%)',
                'inflacion': 'Inflación (%)'
            }
            
            # Mostrar regresión
            try:
                r2, mse, coef, intercept = show_regression(
                    filtered_df[x_var],
                    filtered_df[y_var],
                    var_labels[x_var],
                    var_labels[y_var]
                )
                
                # Mostrar resultados de las regresiones predefinidas si están disponibles
                if resultados is not None:
                    st.subheader("Resumen de Regresiones")
                    st.dataframe(resultados)
            except Exception as e:
                st.error(f"Error al calcular la regresión: {e}")
        else:
            st.warning("Por favor selecciona diferentes variables para X y Y.")
else:
    st.error("No se pudieron cargar los datos. Por favor ejecuta el análisis primero (notebooks/analysis.py).")

# Footer
st.markdown("---")
st.markdown("""
**Tarea 06 - Análisis Económico con ETL/ELT y AWS**  
Desarrollado con Streamlit, Pandas, Scikit-learn y Plotly.
""") 