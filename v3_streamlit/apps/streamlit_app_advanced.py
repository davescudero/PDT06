#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Aplicación Streamlit avanzada para visualizar los resultados del análisis de 
indicadores económicos: Tipo de Cambio, Tasa de Interés e Inflación.
Incluye modelos y análisis avanzados.
"""

import streamlit as st
import pandas as pd
import os
import numpy as np

# Configuración de la página
st.set_page_config(
    page_title="Análisis Avanzado: Indicadores Económicos Mexicanos",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título y descripción
st.title("📊 Análisis Avanzado de Indicadores Económicos Mexicanos")
st.markdown("""
Esta aplicación visualiza el análisis avanzado de tres importantes indicadores económicos de México:
- 💱 **Tipo de cambio PESO/USD**
- 💹 **Tasa de interés**
- 📈 **Inflación**

Los datos provienen del Banco de México y el INEGI, procesados a través de ETL/ELT y analizados con modelos estadísticos avanzados.
""")

# Definir la ruta base para todos los archivos
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# Verificar si existen las imágenes del análisis avanzado
archivos_basicos = [
    os.path.join(BASE_PATH, "series_tiempo.png"), 
    os.path.join(BASE_PATH, "correlacion.png"), 
    os.path.join(BASE_PATH, "reg_tc_ti.png"), 
    os.path.join(BASE_PATH, "reg_ti_inf.png"), 
    os.path.join(BASE_PATH, "reg_tc_inf.png")
]

archivos_avanzados = [
    os.path.join(BASE_PATH, "series_temporales_analisis.png"),
    os.path.join(BASE_PATH, "series_temporales_diferenciadas.png"),
    os.path.join(BASE_PATH, "autocorrelacion.png"),
    os.path.join(BASE_PATH, "arima_prediccion_tipo_de_cambio.png"),
    os.path.join(BASE_PATH, "arima_prediccion_tasa_de_interes.png"),
    os.path.join(BASE_PATH, "arima_prediccion_inflacion.png"),
    os.path.join(BASE_PATH, "regresion_polinomica_tipo_de_cambio_tasa_de_interes.png"),
    os.path.join(BASE_PATH, "regresion_polinomica_tasa_de_interes_inflacion.png"),
    os.path.join(BASE_PATH, "regresion_polinomica_tipo_de_cambio_inflacion.png"),
    os.path.join(BASE_PATH, "causalidad_granger_matrix.png"),
    os.path.join(BASE_PATH, "promedio_movil_tipo_de_cambio.png"),
    os.path.join(BASE_PATH, "promedio_movil_tasa_de_interes.png"),
    os.path.join(BASE_PATH, "promedio_movil_inflacion.png")
]

# Verificar si existen los archivos CSV
datos_csv = [
    os.path.join(BASE_PATH, "indicadores_economicos.csv"), 
    os.path.join(BASE_PATH, "resultados_regresiones.csv"),
    os.path.join(BASE_PATH, "resultados_regresiones_polinomicas.csv")
]

# Verificar existencia de archivos
basicos_existen = any(os.path.exists(path) for path in archivos_basicos)
avanzados_existen = any(os.path.exists(path) for path in archivos_avanzados)
datos_existen = any(os.path.exists(path) for path in datos_csv)

# Sidebar con información del proyecto
st.sidebar.header("Sobre el Proyecto")
st.sidebar.markdown("""
Este proyecto analiza indicadores económicos mexicanos usando:

1. **ETL/ELT con AWS**
   - Extracción de APIs (Banxico, INEGI)
   - Almacenamiento en S3
   - Procesamiento con Athena

2. **Análisis Avanzado**
   - Modelos ARIMA
   - Regresiones polinómicas
   - Pruebas de causalidad
   - Análisis de series temporales
""")

# Mostrar los datos y análisis
if datos_existen or basicos_existen or avanzados_existen:
    # Crear las pestañas principales
    tabs = st.tabs([
        "📈 Series Temporales", 
        "🔄 Correlaciones", 
        "📊 Regresiones", 
        "🔮 Modelos ARIMA",
        "🌊 Causalidad de Granger",
        "📝 Informe"
    ])
    
    # Pestaña 1: Series Temporales
    with tabs[0]:
        st.header("Series Temporales")
        
        # Subtabs para diferentes visualizaciones
        subtabs1 = st.tabs(["Originales", "Diferenciadas", "Promedios Móviles", "Autocorrelación"])
        
        with subtabs1[0]:
            st.subheader("Series Temporales Originales")
            if os.path.exists(os.path.join(BASE_PATH, "series_temporales_analisis.png")):
                st.image(os.path.join(BASE_PATH, "series_temporales_analisis.png"), use_container_width=True)
            elif os.path.exists(os.path.join(BASE_PATH, "series_tiempo.png")):
                st.image(os.path.join(BASE_PATH, "series_tiempo.png"), use_container_width=True)
            else:
                st.warning("No se encontró la visualización de series temporales")
        
        with subtabs1[1]:
            st.subheader("Series Temporales Diferenciadas")
            if os.path.exists(os.path.join(BASE_PATH, "series_temporales_diferenciadas.png")):
                st.image(os.path.join(BASE_PATH, "series_temporales_diferenciadas.png"), use_container_width=True)
                st.markdown("""
                Las series diferenciadas muestran el cambio de una observación a la siguiente. 
                Esto es útil para:
                - Eliminar tendencias y hacer las series estacionarias
                - Visualizar la volatilidad a lo largo del tiempo
                - Identificar patrones cíclicos
                """)
            else:
                st.warning("No se encontró la visualización de series diferenciadas")
        
        with subtabs1[2]:
            st.subheader("Análisis de Promedios Móviles")
            
            # Elegir variable para mostrar promedio móvil
            variable = st.selectbox(
                "Selecciona una variable:",
                ["tipo_de_cambio", "tasa_de_interes", "inflacion"],
                format_func=lambda x: {
                    "tipo_de_cambio": "Tipo de Cambio", 
                    "tasa_de_interes": "Tasa de Interés", 
                    "inflacion": "Inflación"
                }[x]
            )
            
            # Mostrar la imagen correspondiente
            if os.path.exists(os.path.join(BASE_PATH, f"promedio_movil_{variable}.png")):
                st.image(os.path.join(BASE_PATH, f"promedio_movil_{variable}.png"), use_container_width=True)
                st.markdown("""
                El análisis de promedio móvil suaviza fluctuaciones a corto plazo para destacar tendencias a largo plazo.
                - **MA(3)**: Promedio móvil de 3 meses - tendencias a corto plazo
                - **MA(6)**: Promedio móvil de 6 meses - tendencias a medio plazo
                - **MA(12)**: Promedio móvil de 12 meses - tendencias anuales
                """)
            else:
                st.warning(f"No se encontró el análisis de promedio móvil para {variable}")
        
        with subtabs1[3]:
            st.subheader("Funciones de Autocorrelación")
            if os.path.exists(os.path.join(BASE_PATH, "autocorrelacion.png")):
                st.image(os.path.join(BASE_PATH, "autocorrelacion.png"), use_container_width=True)
                st.markdown("""
                **Funciones de autocorrelación (ACF) y autocorrelación parcial (PACF)**
                
                Estas funciones ayudan a identificar:
                - Patrones estacionales en los datos
                - La estructura de dependencia temporal
                - Los parámetros óptimos para modelos ARIMA
                
                Las líneas azules punteadas representan los intervalos de confianza (95%). Las barras que sobrepasan estos límites indican correlaciones significativas.
                """)
            else:
                st.warning("No se encontró el análisis de autocorrelación")
    
    # Pestaña 2: Correlaciones
    with tabs[1]:
        st.header("Correlaciones entre Variables")
        
        if os.path.exists(os.path.join(BASE_PATH, "correlacion.png")):
            st.image(os.path.join(BASE_PATH, "correlacion.png"), use_container_width=True)
            
            # Cargar datos si están disponibles
            try:
                df = pd.read_csv(os.path.join(BASE_PATH, "indicadores_economicos.csv"))
                corr_values = df[['tipo_de_cambio', 'tasa_de_interes', 'inflacion']].corr()
                
                # Mostrar tabla de correlación
                st.subheader("Matriz de Correlación")
                st.dataframe(corr_values.style.format("{:.4f}"))
                
                # Interpretación
                st.subheader("Interpretación")
                st.markdown(f"""
                - **Tipo de Cambio ~ Tasa de Interés**: Correlación de {corr_values.loc['tipo_de_cambio', 'tasa_de_interes']:.4f} (muy débil)
                - **Tipo de Cambio ~ Inflación**: Correlación de {corr_values.loc['tipo_de_cambio', 'inflacion']:.4f} (muy débil)
                - **Tasa de Interés ~ Inflación**: Correlación de {corr_values.loc['tasa_de_interes', 'inflacion']:.4f} (muy débil)
                
                La correlación lineal simple entre estas variables es muy débil, lo que sugiere que:
                1. Las relaciones pueden ser no lineales
                2. Pueden existir retardos temporales importantes
                3. Otros factores externos pueden estar influyendo en estas variables
                """)
            except Exception as e:
                st.warning(f"No se pudieron cargar los datos para análisis de correlación: {e}")
        else:
            st.warning("No se encontró la visualización de correlación")
    
    # Pestaña 3: Regresiones
    with tabs[2]:
        st.header("Análisis de Regresión")
        
        # Subtabs para diferentes tipos de regresión
        subtabs3 = st.tabs(["Regresiones Lineales", "Regresiones Polinómicas"])
        
        with subtabs3[0]:
            st.subheader("Regresiones Lineales")
            
            # Seleccionar relación para mostrar
            relacion_lineal = st.selectbox(
                "Selecciona una relación:",
                ["tc_ti", "ti_inf", "tc_inf"],
                format_func=lambda x: {
                    "tc_ti": "Tipo de Cambio ~ Tasa de Interés", 
                    "ti_inf": "Tasa de Interés ~ Inflación", 
                    "tc_inf": "Tipo de Cambio ~ Inflación"
                }[x]
            )
            
            # Mapeo de relaciones a archivos de imagen
            archivos_regresion = {
                "tc_ti": "reg_tc_ti.png",
                "ti_inf": "reg_ti_inf.png",
                "tc_inf": "reg_tc_inf.png"
            }
            
            # Mostrar la imagen seleccionada
            if os.path.exists(os.path.join(BASE_PATH, archivos_regresion[relacion_lineal])):
                st.image(os.path.join(BASE_PATH, archivos_regresion[relacion_lineal]), use_container_width=True)
                
                # Cargar resultados si están disponibles
                try:
                    resultados = pd.read_csv(os.path.join(BASE_PATH, "resultados_regresiones.csv"))
                    idx = {"tc_ti": 0, "ti_inf": 1, "tc_inf": 2}[relacion_lineal]
                    
                    reg = resultados.iloc[idx]
                    st.markdown(f"""
                    **Detalles de la Regresión**
                    
                    - **Ecuación**: Y = {reg['Coeficiente']:.4f} × X + {reg['Intercepto']:.4f}
                    - **R²**: {reg['R²']:.4f} (Porcentaje de varianza explicada)
                    - **MSE**: {reg['MSE']:.4f} (Error cuadrático medio)
                    
                    El coeficiente de determinación (R²) cercano a cero indica que este modelo lineal simple no captura adecuadamente la relación entre las variables.
                    """)
                except Exception as e:
                    st.warning(f"No se pudieron cargar los resultados de regresión: {e}")
            else:
                st.warning(f"No se encontró la visualización para la regresión seleccionada")
        
        with subtabs3[1]:
            st.subheader("Regresiones Polinómicas")
            
            # Seleccionar relación para mostrar
            relacion_poli = st.selectbox(
                "Selecciona una relación:",
                ["tipo_de_cambio_tasa_de_interes", "tasa_de_interes_inflacion", "tipo_de_cambio_inflacion"],
                format_func=lambda x: {
                    "tipo_de_cambio_tasa_de_interes": "Tipo de Cambio ~ Tasa de Interés", 
                    "tasa_de_interes_inflacion": "Tasa de Interés ~ Inflación", 
                    "tipo_de_cambio_inflacion": "Tipo de Cambio ~ Inflación"
                }[x]
            )
            
            # Mapeo de relaciones a archivos de imagen
            archivos_regresion_poli = {
                "tipo_de_cambio_tasa_de_interes": "regresion_polinomica_tipo_de_cambio_tasa_de_interes.png",
                "tasa_de_interes_inflacion": "regresion_polinomica_tasa_de_interes_inflacion.png",
                "tipo_de_cambio_inflacion": "regresion_polinomica_tipo_de_cambio_inflacion.png"
            }
            
            # Mostrar la imagen seleccionada
            if os.path.exists(os.path.join(BASE_PATH, archivos_regresion_poli[relacion_poli])):
                st.image(os.path.join(BASE_PATH, archivos_regresion_poli[relacion_poli]), use_container_width=True)
                
                # Cargar resultados si están disponibles
                try:
                    if os.path.exists(os.path.join(BASE_PATH, "resultados_regresiones_polinomicas.csv")):
                        resultados_poli = pd.read_csv(os.path.join(BASE_PATH, "resultados_regresiones_polinomicas.csv"))
                        idx = {
                            "tipo_de_cambio_tasa_de_interes": 0, 
                            "tasa_de_interes_inflacion": 1, 
                            "tipo_de_cambio_inflacion": 2
                        }[relacion_poli]
                        
                        reg = resultados_poli.iloc[idx]
                        st.markdown(f"""
                        **Detalles de la Regresión Polinómica**
                        
                        - **Grado óptimo**: {int(reg['grado'])}
                        - **R²**: {reg['r2']:.4f} (Porcentaje de varianza explicada)
                        
                        Aunque la regresión polinómica mejora ligeramente sobre la regresión lineal, el R² sigue siendo bajo, 
                        lo que sugiere que incluso modelos no lineales simples no capturan toda la complejidad de la relación.
                        """)
                    else:
                        st.markdown("""
                        **Regresión Polinómica**
                        
                        La regresión polinómica permite capturar relaciones no lineales entre variables, 
                        ajustando curvas de mayor orden en lugar de líneas rectas.
                        
                        En este caso, se probaron polinomios de grado 1, 2 y 3, seleccionando el que proporciona mejor ajuste.
                        """)
                except Exception as e:
                    st.warning(f"No se pudieron cargar los resultados de regresión polinómica: {e}")
            else:
                st.warning(f"No se encontró la visualización para la regresión polinómica seleccionada")
    
    # Pestaña 4: Modelos ARIMA
    with tabs[3]:
        st.header("Modelos ARIMA de Series Temporales")
        
        # Seleccionar variable para mostrar predicción ARIMA
        variable_arima = st.selectbox(
            "Selecciona una variable:",
            ["tipo_de_cambio", "tasa_de_interes", "inflacion"],
            format_func=lambda x: {
                "tipo_de_cambio": "Tipo de Cambio", 
                "tasa_de_interes": "Tasa de Interés", 
                "inflacion": "Inflación"
            }[x],
            key="arima_select"
        )
        
        # Mostrar la imagen correspondiente
        if os.path.exists(os.path.join(BASE_PATH, f"arima_prediccion_{variable_arima}.png")):
            st.image(os.path.join(BASE_PATH, f"arima_prediccion_{variable_arima}.png"), use_container_width=True)
            
            # Información sobre el modelo
            st.markdown(f"""
            **Modelo ARIMA para {variable_arima.replace('_', ' ').title()}**
            
            Los modelos ARIMA (AutoRegressive Integrated Moving Average) son ampliamente utilizados para modelar y 
            predecir series temporales. Para este análisis, se utilizó un modelo ARIMA(1,1,1), donde:
            
            - **AR(1)**: Componente autorregresivo de orden 1
            - **I(1)**: Diferenciación de primer orden para lograr estacionariedad
            - **MA(1)**: Componente de media móvil de orden 1
            
            La línea azul muestra los datos de entrenamiento, la línea verde los valores reales, y la línea roja 
            las predicciones del modelo para los últimos 12 meses de datos.
            """)
            
            # MSE según la variable
            mse_values = {
                "tipo_de_cambio": 0.9571,
                "tasa_de_interes": 19.3385,
                "inflacion": 8.3385
            }
            
            st.metric(
                label="Error Cuadrático Medio (MSE)",
                value=f"{mse_values[variable_arima]:.4f}"
            )
            
            st.markdown("""
            **Interpretación del MSE:**
            - Valores más bajos indican mejores predicciones
            - El MSE está en las mismas unidades que los datos al cuadrado
            - Comparativamente, el tipo de cambio tiene menor error de predicción
            """)
        else:
            st.warning(f"No se encontró la predicción ARIMA para {variable_arima}")
    
    # Pestaña 5: Causalidad de Granger
    with tabs[4]:
        st.header("Análisis de Causalidad de Granger")
        
        if os.path.exists(os.path.join(BASE_PATH, "causalidad_granger_matrix.png")):
            st.image(os.path.join(BASE_PATH, "causalidad_granger_matrix.png"), use_container_width=True)
            
            st.markdown("""
            **Prueba de Causalidad de Granger**
            
            La prueba de causalidad de Granger determina si una serie temporal es útil para predecir otra. En otras palabras, 
            evalúa si los cambios en una variable "causan" (en el sentido de Granger) cambios en otra variable.
            
            **Resultados principales:**
            """)
            
            # Tabla de resultados de causalidad
            causality_data = {
                "Causa": ["Tasa de Interés", "Tasa de Interés", "Inflación", "Inflación"],
                "Efecto": ["Tipo de Cambio", "Inflación", "Tipo de Cambio", "Tasa de Interés"],
                "Retardo (meses)": [9, 9, 4, 11],
                "p-value": [0.0216, 0.0000, 0.0000, 0.0000]
            }
            
            causality_df = pd.DataFrame(causality_data)
            st.dataframe(causality_df)
            
            st.markdown("""
            **Interpretación:**
            
            1. **Bidireccionalidad Inflación-Tasa de Interés:** Existe una relación causal en ambas direcciones, lo que refleja el complejo mecanismo de transmisión de la política monetaria.
            
            2. **Efectos Temporales:** Los cambios en una variable afectan a otra después de cierto tiempo:
               - La inflación afecta al tipo de cambio después de 4 meses
               - La tasa de interés afecta al tipo de cambio y a la inflación después de 9 meses
               - La inflación afecta a la tasa de interés después de 11 meses
            
            3. **Implicaciones para Políticas Monetarias:** Las decisiones del banco central sobre tasas de interés tienen efectos retardados tanto en la inflación como en el tipo de cambio.
            """)
        else:
            st.warning("No se encontró el análisis de causalidad de Granger")
    
    # Pestaña 6: Informe completo
    with tabs[5]:
        st.header("Informe Completo de Análisis")
        
        if os.path.exists(os.path.join(BASE_PATH, "informe_analisis_avanzado.md")):
            with open(os.path.join(BASE_PATH, "informe_analisis_avanzado.md"), "r") as f:
                informe = f.read()
            
            st.markdown(informe)
        else:
            st.warning("No se encontró el informe de análisis avanzado")
            
            # Mostrar conclusiones generales si no hay informe
            st.subheader("Conclusiones Generales")
            st.markdown("""
            1. **Relaciones Complejas**: Las relaciones entre los indicadores económicos son más complejas que simples correlaciones lineales.

            2. **Efectos Temporales**: Existen efectos retardados entre las variables, donde los cambios en una variable afectan a otra después de varios meses.

            3. **Bidireccionalidad**: La relación entre la inflación y la tasa de interés es bidireccional, lo que refleja la complejidad de la economía y la política monetaria.

            4. **Predictibilidad Limitada**: A pesar de utilizar modelos avanzados, la predictibilidad sigue siendo moderada, lo que refleja la influencia de factores externos no incluidos en el análisis.

            5. **Implicaciones para Políticas**: Las decisiones de política monetaria (tasa de interés) tienen impactos en el tipo de cambio e inflación que se manifiestan con retardos temporales significativos.
            """)
else:
    st.error("No se encontraron las imágenes o archivos de datos. Por favor ejecuta los análisis primero.")

# Footer
st.markdown("---")
st.markdown("""
**Tarea 06 - Análisis Económico con ETL/ELT y AWS**  
Desarrollado con Python, Pandas, Statsmodels, Scikit-learn, Streamlit y AWS.
""")

# Mostrar información sobre las herramientas en la sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Herramientas Utilizadas")
st.sidebar.markdown("""
- **ETL/ELT**: Python, Pandas, boto3
- **Análisis**: Statsmodels, Scikit-learn, NumPy
- **Visualización**: Matplotlib, Seaborn, Plotly
- **AWS**: S3, Glue, Athena
- **App**: Streamlit
""")

# Añadir información de contacto
st.sidebar.markdown("---")
st.sidebar.info("Desarrollado por David Escudero") 