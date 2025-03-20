#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Aplicación Streamlit simplificada para visualizar los resultados del análisis de 
indicadores económicos: Tipo de Cambio, Tasa de Interés e Inflación.
"""

import streamlit as st
import pandas as pd
import os

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

# Verificar si existen las imágenes
series_tiempo_path = "series_tiempo.png"
correlacion_path = "correlacion.png"
reg_tc_ti_path = "reg_tc_ti.png"
reg_ti_inf_path = "reg_ti_inf.png"
reg_tc_inf_path = "reg_tc_inf.png"

# Verificar si existen los archivos CSV
indicadores_path = "indicadores_economicos.csv"
resultados_path = "resultados_regresiones.csv"

# Comprobar si existen las imágenes y los archivos CSV
imagenes_existen = all(os.path.exists(path) for path in [series_tiempo_path, correlacion_path, reg_tc_ti_path, reg_ti_inf_path, reg_tc_inf_path])
datos_existen = all(os.path.exists(path) for path in [indicadores_path, resultados_path])

if imagenes_existen and datos_existen:
    # Dividir la página en pestañas
    tab1, tab2, tab3 = st.tabs(["📈 Series de Tiempo", "🔄 Correlaciones", "📊 Regresiones"])
    
    with tab1:
        st.header("Series de Tiempo")
        st.image(series_tiempo_path, use_column_width=True)
        
        # Mostrar datos si están disponibles
        try:
            df = pd.read_csv(indicadores_path)
            with st.expander("📋 Ver datos"):
                st.write(df)
        except Exception as e:
            st.warning(f"No se pudieron cargar los datos: {e}")
    
    with tab2:
        st.header("Correlaciones entre Variables")
        st.image(correlacion_path, use_column_width=True)
        
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
        
        # Mostrar resultados de regresiones
        try:
            resultados = pd.read_csv(resultados_path)
            st.subheader("Resumen de Regresiones")
            st.write(resultados)
        except Exception as e:
            st.warning(f"No se pudieron cargar los resultados de regresiones: {e}")
        
        # Mostrar imágenes de regresiones
        st.subheader("Tipo de Cambio ~ Tasa de Interés")
        st.image(reg_tc_ti_path, use_column_width=True)
        
        st.subheader("Tasa de Interés ~ Inflación")
        st.image(reg_ti_inf_path, use_column_width=True)
        
        st.subheader("Tipo de Cambio ~ Inflación")
        st.image(reg_tc_inf_path, use_column_width=True)
else:
    st.error("No se encontraron las imágenes o archivos de datos. Por favor ejecuta el análisis primero (notebooks/analysis.py).")

# Footer
st.markdown("---")
st.markdown("""
**Tarea 06 - Análisis Económico con ETL/ELT y AWS**  
Desarrollado con Streamlit, Pandas, Scikit-learn y Plotly.
""") 