#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Aplicación Streamlit mejorada para visualizar los resultados del análisis de 
indicadores económicos: Tipo de Cambio, Tasa de Interés e Inflación.
Utiliza los datos limpios y las regresiones mejoradas.
"""

import streamlit as st
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(
    page_title="Análisis Mejorado: Indicadores Económicos Mexicanos",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título y descripción
st.title("📊 Análisis Mejorado de Indicadores Económicos Mexicanos")
st.markdown("""
Esta aplicación visualiza el análisis mejorado de tres importantes indicadores económicos de México:
- 💱 **Tipo de cambio PESO/USD**
- 💹 **Tasa de interés**
- 📈 **Inflación**

Los datos provienen del Banco de México y el INEGI, procesados y limpiados para un análisis más preciso.
""")

# Definir la ruta base para todos los archivos
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# Verificar que existan los archivos necesarios
datos_existen = os.path.exists(os.path.join(BASE_PATH, "indicadores_economicos_clean.csv"))
resultados_existen = os.path.exists(os.path.join(BASE_PATH, "resultados_regresiones_polinomicas_mejoradas.csv"))
informe_existe = os.path.exists(os.path.join(BASE_PATH, "informe_analisis_mejorado.md"))

# Sidebar con información del proyecto
st.sidebar.header("Sobre el Proyecto")
st.sidebar.markdown("""
Este proyecto analiza indicadores económicos mexicanos usando:

1. **ETL/ELT con AWS**
   - Extracción de APIs (Banxico, INEGI)
   - Almacenamiento en S3
   - Procesamiento con Athena

2. **Análisis Mejorado**
   - Limpieza y corrección de datos
   - Modelos con retardos temporales
   - Regresiones polinómicas avanzadas
   - Modelos ARIMA
""")

# Mostrar los datos y análisis
if datos_existen:
    # Cargar datos
    df = pd.read_csv(os.path.join(BASE_PATH, "indicadores_economicos_clean.csv"))
    df['date'] = pd.to_datetime(df['date'])
    
    # Crear las pestañas principales
    tabs = st.tabs([
        "📈 Datos y Series Temporales", 
        "🔄 Correlaciones", 
        "📊 Regresiones Mejoradas", 
        "🔮 Modelos ARIMA",
        "📝 Informe Completo"
    ])
    
    # Pestaña 1: Datos y Series Temporales
    with tabs[0]:
        st.header("Datos y Series Temporales")
        
        # Mostrar datos limpios
        st.subheader("Datos Limpios")
        st.dataframe(df.set_index('date'))
        
        # Estadísticas descriptivas
        st.subheader("Estadísticas Descriptivas")
        st.dataframe(df.describe())
        
        # Series temporales
        st.subheader("Series Temporales")
        if os.path.exists(os.path.join(BASE_PATH, "series_temporales_analisis_mejorado.png")):
            st.image(os.path.join(BASE_PATH, "series_temporales_analisis_mejorado.png"), use_container_width=True)
        else:
            # Si no existe la imagen, graficar en tiempo real
            fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
            df.set_index('date')['tipo_de_cambio'].plot(ax=axes[0], title='Tipo de Cambio (MXN/USD)')
            df.set_index('date')['tasa_de_interes'].plot(ax=axes[1], title='Tasa de Interés (%)')
            df.set_index('date')['inflacion'].plot(ax=axes[2], title='Inflación (%)')
            plt.tight_layout()
            st.pyplot(fig)
    
    # Pestaña 2: Correlaciones
    with tabs[1]:
        st.header("Análisis de Correlación")
        
        # Correlación simple
        st.subheader("Matriz de Correlación")
        corr = df.set_index('date')[['tipo_de_cambio', 'tasa_de_interes', 'inflacion']].corr()
        st.dataframe(corr)
        
        # Visualizar matriz de correlación
        if os.path.exists(os.path.join(BASE_PATH, "correlacion_mejorada.png")):
            st.image(os.path.join(BASE_PATH, "correlacion_mejorada.png"), use_container_width=True)
        
        # Correlación con retardos
        st.subheader("Correlación con Retardos Temporales")
        if os.path.exists(os.path.join(BASE_PATH, "correlacion_retardos_mejorada.png")):
            st.image(os.path.join(BASE_PATH, "correlacion_retardos_mejorada.png"), use_container_width=True)
            
            st.markdown("""
            El análisis de correlación con retardos temporales muestra cómo la relación entre los indicadores
            económicos varía según el tiempo transcurrido. Este análisis es crucial para entender:
            
            1. **Efectos Retardados**: Las políticas monetarias (tasa de interés) pueden tomar tiempo en afectar
               al tipo de cambio o la inflación.
            
            2. **Relaciones Causales**: Correlaciones más fuertes en ciertos retardos pueden indicar relaciones
               causales, donde los cambios en una variable preceden y potencialmente causan cambios en otra.
            
            3. **Horizontes de Predicción**: Conocer los retardos con mayor correlación ayuda a determinar el
               horizonte temporal óptimo para los modelos predictivos.
            """)
    
    # Pestaña 3: Regresiones Mejoradas
    with tabs[2]:
        st.header("Regresiones Polinómicas Mejoradas")
        
        if resultados_existen:
            # Cargar resultados de regresiones
            resultados_poli = pd.read_csv(os.path.join(BASE_PATH, "resultados_regresiones_polinomicas_mejoradas.csv"))
            
            # Selector para mostrar diferentes regresiones
            opciones = []
            for _, row in resultados_poli.iterrows():
                lag_text = f" (Retardo: {int(row['lag'])} meses)" if row['lag'] > 0 else ""
                opciones.append(f"{row['y_label']} ~ {row['x_label']}{lag_text}")
            
            seleccion = st.selectbox("Selecciona una regresión:", opciones)
            
            # Encontrar el índice de la regresión seleccionada
            indice = opciones.index(seleccion)
            regresion = resultados_poli.iloc[indice]
            
            # Mostrar detalles de la regresión
            st.subheader(f"Detalles de la Regresión: {seleccion}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("R²", f"{regresion['r2']:.4f}")
                st.metric("Grado del Polinomio", f"{int(regresion['grado'])}")
                
                # Mostrar la ecuación
                st.subheader("Ecuación Polinómica")
                coeficientes = eval(regresion['coeficientes'])
                ecuacion = f"{regresion['y_label']} = {regresion['intercepto']:.4f}"
                for i, coef in enumerate(coeficientes[1:], 1):
                    if i == 1:
                        ecuacion += f" + {coef:.4f} × {regresion['x_label']}"
                    else:
                        ecuacion += f" + {coef:.4f} × {regresion['x_label']}^{i}"
                
                st.markdown(f"```{ecuacion}```")
                
                # Mostrar interpretación
                st.subheader("Interpretación")
                
                if regresion['lag'] > 0:
                    st.markdown(f"""
                    Esta regresión muestra el efecto de los cambios en {regresion['x_label']} sobre {regresion['y_label']} 
                    con un retardo de {int(regresion['lag'])} meses. 
                    
                    El valor R² de {regresion['r2']:.4f} indica que aproximadamente el {regresion['r2']*100:.1f}% de la 
                    variabilidad en {regresion['y_label']} puede ser explicada por los cambios en {regresion['x_label']} 
                    ocurridos {int(regresion['lag'])} meses antes.
                    """)
                else:
                    st.markdown(f"""
                    Esta regresión muestra la relación contemporánea entre {regresion['x_label']} y {regresion['y_label']}.
                    
                    El valor R² de {regresion['r2']:.4f} indica que aproximadamente el {regresion['r2']*100:.1f}% de la 
                    variabilidad en {regresion['y_label']} puede ser explicada por los cambios en {regresion['x_label']}.
                    """)
            
            with col2:
                # Mostrar la imagen de la regresión
                if os.path.exists(os.path.join(BASE_PATH, regresion['imagen'])):
                    st.image(os.path.join(BASE_PATH, regresion['imagen']), use_container_width=True)
            
            # Comparación de regresiones
            st.subheader("Comparación de Todas las Regresiones")
            
            # Preparar datos para la tabla de comparación
            tabla_comparacion = []
            for _, row in resultados_poli.iterrows():
                lag_text = f" (t-{int(row['lag'])})" if row['lag'] > 0 else ""
                tabla_comparacion.append({
                    "Relación": f"{row['y_label']} ~ {row['x_label']}{lag_text}",
                    "R²": f"{row['r2']:.4f}",
                    "Grado": int(row['grado']),
                    "Mejora con Retardo": "N/A" if row['lag'] == 0 else "Sí" if row['r2'] > resultados_poli[(resultados_poli['x_var'] == row['x_var']) & (resultados_poli['y_var'] == row['y_var']) & (resultados_poli['lag'] == 0)]['r2'].values[0] else "No"
                })
            
            st.dataframe(pd.DataFrame(tabla_comparacion))
            
            st.markdown("""
            ### Conclusiones de las Regresiones
            
            1. **Importancia de los Retardos**: Los modelos que incorporan retardos temporales muestran una mejora 
            significativa en su capacidad predictiva (R²) comparados con los modelos sin retardo.
            
            2. **No-Linealidad**: El hecho de que los modelos polinómicos de grado 3 tengan mejor ajuste que los 
            lineales indica relaciones no lineales entre las variables económicas.
            
            3. **Causalidad Temporal**: Los retardos específicos con mejor ajuste sugieren posibles relaciones 
            causales con un intervalo temporal definido.
            """)
        else:
            st.warning("No se encontraron resultados de regresiones mejoradas.")
    
    # Pestaña 4: Modelos ARIMA
    with tabs[3]:
        st.header("Modelos ARIMA de Series Temporales")
        
        # Selector de variable
        variable = st.selectbox(
            "Selecciona una variable:",
            ["tipo_de_cambio", "tasa_de_interes", "inflacion"],
            format_func=lambda x: {
                "tipo_de_cambio": "Tipo de Cambio (MXN/USD)", 
                "tasa_de_interes": "Tasa de Interés (%)", 
                "inflacion": "Inflación (%)"
            }[x]
        )
        
        # Mostrar resultado de ARIMA
        if os.path.exists(os.path.join(BASE_PATH, f"arima_prediccion_mejorado_{variable}.png")):
            st.image(os.path.join(BASE_PATH, f"arima_prediccion_mejorado_{variable}.png"), use_container_width=True)
            
            # Mostrar métricas según la variable
            mse_values = {
                "tipo_de_cambio": 17.1166,
                "tasa_de_interes": 2.5041,
                "inflacion": 3.6692
            }
            
            st.metric("Error Cuadrático Medio (MSE)", f"{mse_values[variable]:.4f}")
            
            st.markdown(f"""
            ### Modelo ARIMA para {variable.replace('_', ' ').title()}
            
            El modelo ARIMA (AutoRegressive Integrated Moving Average) se utilizó para modelar y predecir 
            la serie temporal. Se aplicó un modelo ARIMA(1,1,1), donde:
            
            - **AR(1)**: Componente autorregresivo de orden 1
            - **I(1)**: Diferenciación de primer orden para lograr estacionariedad
            - **MA(1)**: Componente de media móvil de orden 1
            
            En el gráfico:
            - La línea azul muestra los datos de entrenamiento
            - La línea verde muestra los valores reales
            - La línea roja muestra las predicciones del modelo
            
            **Interpretación del MSE:** 
            - Valores más bajos indican mejores predicciones
            - El MSE está en las mismas unidades que los datos al cuadrado
            """)
        else:
            st.warning(f"No se encontró la predicción ARIMA para {variable}")
    
    # Pestaña 5: Informe Completo
    with tabs[4]:
        st.header("Informe de Análisis Completo")
        
        if informe_existe:
            with open(os.path.join(BASE_PATH, "informe_analisis_mejorado.md"), "r") as f:
                informe = f.read()
            
            st.markdown(informe)
        else:
            st.warning("No se encontró el informe de análisis completo.")
            
            # Mostrar conclusiones generales si no hay informe
            st.subheader("Conclusiones Generales")
            st.markdown("""
            1. **Correlaciones Mejoradas**: El análisis con retardos temporales muestra relaciones más fuertes que las correlaciones simples sin retardo.

            2. **Regresiones Polinómicas**: Los modelos con retardos temporales proporcionan un mejor ajuste, demostrando que las relaciones entre los indicadores económicos tienen componentes temporales importantes.

            3. **Modelos ARIMA**: Proporcionan una capacidad predictiva moderada para las series temporales, capturando la dinámica a corto plazo.

            4. **Implicaciones para Políticas**: Los resultados sugieren que las políticas monetarias (tasa de interés) tienen impactos en tipo de cambio e inflación que se manifiestan con ciertos retardos temporales específicos.

            5. **Recomendaciones**: Para futuras investigaciones, se recomienda considerar modelos más complejos como VAR (Vector Autoregression) o VEC (Vector Error Correction) que puedan capturar mejor la dinámica multivariable de estos indicadores económicos.
            """)
else:
    st.error("""
    No se encontraron los datos limpios. Por favor ejecuta primero los siguientes scripts:
    1. `python limpiar_datos.py` - Para limpiar los datos originales
    2. `python analysis_improved.py` - Para realizar el análisis mejorado
    """)

# Footer
st.markdown("---")
st.markdown("""
**Tarea 06 - Análisis Económico Mejorado con ETL/ELT y AWS**  
Desarrollado con Python, Pandas, Statsmodels, Scikit-learn, Streamlit y AWS.
""")

# Mostrar información sobre las herramientas en la sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Herramientas Utilizadas")
st.sidebar.markdown("""
- **Limpieza de Datos**: Python, Pandas, NumPy
- **Análisis**: Statsmodels, Scikit-learn
- **Visualización**: Matplotlib, Seaborn, Streamlit
- **ETL/ELT**: AWS S3, Glue, Athena
""")

# Añadir información de contacto
st.sidebar.markdown("---")
st.sidebar.info("Desarrollado por David Escudero") 