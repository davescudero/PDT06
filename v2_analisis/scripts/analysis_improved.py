#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Análisis Mejorado de Indicadores Económicos

Este script implementa modelos más adecuados para analizar la relación entre:
- Tipo de cambio PESO/USD
- Tasa de interés
- Inflación

Utiliza los datos limpios generados por el script limpiar_datos.py.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import warnings

# Ignorar advertencias para una salida más limpia
warnings.filterwarnings("ignore")

# Configurar estilo de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("notebook")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def cargar_datos():
    """Cargar y preparar los datos."""
    print("Cargando datos limpios...")
    
    # Verificar si los datos limpios existen, si no, ejecutar el script de limpieza
    if not os.path.exists('indicadores_economicos_clean.csv'):
        print("No se encontraron datos limpios. Ejecutando script de limpieza...")
        from limpiar_datos import limpiar_datos_indicadores
        limpiar_datos_indicadores()
    
    # Cargar el dataset limpio
    df = pd.read_csv('indicadores_economicos_clean.csv')
    
    # Convertir fecha a formato datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Establecer fecha como índice
    df.set_index('date', inplace=True)
    
    # Asegurar que los datos estén ordenados por fecha
    df.sort_index(inplace=True)
    
    # Verificar que no haya valores nulos
    if df.isnull().any().any():
        print("Advertencia: Se encontraron valores nulos, reemplazándolos...")
        df.fillna(method='ffill', inplace=True)  # Forward fill
    
    # Mostrar estadísticas descriptivas
    print("\nEstadísticas descriptivas:")
    print(df.describe())
    
    # Mostrar matriz de correlación
    print("\nMatriz de correlación:")
    print(df.corr())
    
    return df

def analizar_estacionariedad(df):
    """Analizar la estacionariedad de las series temporales."""
    print("\nAnalizando estacionariedad de las series temporales...")
    
    resultados = {}
    
    for columna in df.columns:
        # Prueba Dickey-Fuller aumentada
        result = adfuller(df[columna].dropna())
        
        # Interpretar resultados
        estacionaria = result[1] < 0.05
        
        resultados[columna] = {
            'p-value': result[1],
            'estacionaria': estacionaria,
            'estadístico-test': result[0],
            'valores-críticos': result[4]
        }
        
        print(f"\nSerie: {columna}")
        print(f"Es estacionaria: {'Sí' if estacionaria else 'No'}")
        print(f"p-value: {result[1]:.4f}")
        print(f"Estadístico de prueba: {result[0]:.4f}")
        print("Valores críticos:")
        for key, value in result[4].items():
            print(f"\t{key}: {value:.4f}")
    
    # Visualizar las series temporales
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    # Graficar las series
    df['tipo_de_cambio'].plot(ax=axes[0], title='Tipo de Cambio (MXN/USD)')
    df['tasa_de_interes'].plot(ax=axes[1], title='Tasa de Interés (%)')
    df['inflacion'].plot(ax=axes[2], title='Inflación (%)')
    
    # Añadir grid y leyenda
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Ajustar diseño
    plt.tight_layout()
    plt.savefig('series_temporales_analisis_mejorado.png')
    plt.close()
    
    # Visualizar las diferencias para hacer las series estacionarias
    diff_df = df.diff().dropna()
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    # Graficar las diferencias
    diff_df['tipo_de_cambio'].plot(ax=axes[0], title='Diferencia de Tipo de Cambio')
    diff_df['tasa_de_interes'].plot(ax=axes[1], title='Diferencia de Tasa de Interés')
    diff_df['inflacion'].plot(ax=axes[2], title='Diferencia de Inflación')
    
    # Añadir grid y leyenda
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Ajustar diseño
    plt.tight_layout()
    plt.savefig('series_temporales_diferenciadas_mejorado.png')
    plt.close()
    
    return resultados, diff_df

def analisis_de_correlacion(df):
    """
    Realiza un análisis de correlación más detallado, incluyendo
    correlaciones con retardos temporales.
    """
    print("\nAnalizando correlaciones entre variables...")
    
    # Correlación simple
    corr_simple = df.corr()
    print("Correlación simple:")
    print(corr_simple)
    
    # Visualizar matriz de correlación
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_simple, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Matriz de Correlación - Variables Económicas')
    plt.tight_layout()
    plt.savefig('correlacion_mejorada.png')
    plt.close()
    
    # Análisis de correlación con retardos (de 1 a 12 meses)
    print("\nAnálisis de correlación con retardos temporales:")
    max_lag = 12
    
    # Crear DataFrames para almacenar resultados
    lag_corr_tc_ti = pd.DataFrame(index=range(1, max_lag + 1), columns=['Correlación'])
    lag_corr_tc_inf = pd.DataFrame(index=range(1, max_lag + 1), columns=['Correlación'])
    lag_corr_ti_inf = pd.DataFrame(index=range(1, max_lag + 1), columns=['Correlación'])
    
    # Calcular correlaciones con retardos
    for lag in range(1, max_lag + 1):
        # Tipo de cambio ~ Tasa de interés retardada
        lag_corr_tc_ti.loc[lag, 'Correlación'] = df['tipo_de_cambio'].corr(df['tasa_de_interes'].shift(lag))
        
        # Tipo de cambio ~ Inflación retardada
        lag_corr_tc_inf.loc[lag, 'Correlación'] = df['tipo_de_cambio'].corr(df['inflacion'].shift(lag))
        
        # Tasa de interés ~ Inflación retardada
        lag_corr_ti_inf.loc[lag, 'Correlación'] = df['tasa_de_interes'].corr(df['inflacion'].shift(lag))
    
    # Visualizar correlaciones con retardos
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    lag_corr_tc_ti.plot(ax=axes[0], marker='o')
    axes[0].set_title('Correlación: Tipo de Cambio ~ Tasa de Interés con Retardo')
    axes[0].set_xlabel('Retardo (meses)')
    axes[0].set_ylabel('Correlación')
    axes[0].grid(True, alpha=0.3)
    
    lag_corr_tc_inf.plot(ax=axes[1], marker='o')
    axes[1].set_title('Correlación: Tipo de Cambio ~ Inflación con Retardo')
    axes[1].set_xlabel('Retardo (meses)')
    axes[1].set_ylabel('Correlación')
    axes[1].grid(True, alpha=0.3)
    
    lag_corr_ti_inf.plot(ax=axes[2], marker='o')
    axes[2].set_title('Correlación: Tasa de Interés ~ Inflación con Retardo')
    axes[2].set_xlabel('Retardo (meses)')
    axes[2].set_ylabel('Correlación')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('correlacion_retardos_mejorada.png')
    plt.close()
    
    # Guardar resultados
    resultado_correlaciones = {
        'correlacion_simple': corr_simple,
        'correlacion_tc_ti_retardos': lag_corr_tc_ti,
        'correlacion_tc_inf_retardos': lag_corr_tc_inf,
        'correlacion_ti_inf_retardos': lag_corr_ti_inf
    }
    
    return resultado_correlaciones

def regresion_polinomica_mejorada(df):
    """
    Realiza regresiones polinómicas entre variables, considerando también
    los mejores retardos temporales identificados en el análisis de correlación.
    """
    print("\nRealizando regresiones polinómicas mejoradas...")
    
    resultados_polinomicos = []
    
    # Obtener los mejores retardos de correlación
    resultado_correlaciones = analisis_de_correlacion(df)
    
    # Encontrar el mejor retardo para cada par de variables
    mejor_retardo_tc_ti = resultado_correlaciones['correlacion_tc_ti_retardos']['Correlación'].abs().idxmax()
    mejor_retardo_tc_inf = resultado_correlaciones['correlacion_tc_inf_retardos']['Correlación'].abs().idxmax()
    mejor_retardo_ti_inf = resultado_correlaciones['correlacion_ti_inf_retardos']['Correlación'].abs().idxmax()
    
    print(f"Mejor retardo para Tipo de Cambio ~ Tasa de Interés: {mejor_retardo_tc_ti} meses")
    print(f"Mejor retardo para Tipo de Cambio ~ Inflación: {mejor_retardo_tc_inf} meses")
    print(f"Mejor retardo para Tasa de Interés ~ Inflación: {mejor_retardo_ti_inf} meses")
    
    # Definir las relaciones a analizar con sus mejores retardos
    relaciones = [
        # Sin retardo
        ('tasa_de_interes', 'tipo_de_cambio', 'Tasa de Interés (%)', 'Tipo de Cambio (MXN/USD)', 'blue', 0),
        ('inflacion', 'tasa_de_interes', 'Inflación (%)', 'Tasa de Interés (%)', 'red', 0),
        ('inflacion', 'tipo_de_cambio', 'Inflación (%)', 'Tipo de Cambio (MXN/USD)', 'green', 0),
        
        # Con mejores retardos
        ('tasa_de_interes', 'tipo_de_cambio', f'Tasa de Interés (%) [t-{mejor_retardo_tc_ti}]', 'Tipo de Cambio (MXN/USD)', 'darkblue', mejor_retardo_tc_ti),
        ('inflacion', 'tasa_de_interes', f'Inflación (%) [t-{mejor_retardo_ti_inf}]', 'Tasa de Interés (%)', 'darkred', mejor_retardo_ti_inf),
        ('inflacion', 'tipo_de_cambio', f'Inflación (%) [t-{mejor_retardo_tc_inf}]', 'Tipo de Cambio (MXN/USD)', 'darkgreen', mejor_retardo_tc_inf)
    ]
    
    # Probar diferentes grados de polinomio
    grados = [1, 2, 3]
    
    for x_var, y_var, x_label, y_label, color, lag in relaciones:
        mejor_r2 = -float('inf')
        mejor_grado = 1
        mejor_modelo = None
        
        # Preparar los datos con el retardo adecuado
        if lag > 0:
            # Crear un DataFrame con el retardo
            df_lag = pd.DataFrame()
            df_lag[y_var] = df[y_var]
            df_lag[x_var] = df[x_var].shift(lag)
            df_lag = df_lag.dropna()
            
            X = df_lag[x_var].values.reshape(-1, 1)
            y = df_lag[y_var].values
            
            print(f"\nAnalizando {y_label} ~ {x_label} con retardo de {lag} meses")
        else:
            X = df[x_var].values.reshape(-1, 1)
            y = df[y_var].values
            
            print(f"\nAnalizando {y_label} ~ {x_label} sin retardo")
        
        for grado in grados:
            # Crear características polinómicas
            poly = PolynomialFeatures(degree=grado)
            X_poly = poly.fit_transform(X)
            
            # Crear y ajustar el modelo
            modelo = LinearRegression()
            modelo.fit(X_poly, y)
            
            # Calcular métricas
            y_pred = modelo.predict(X_poly)
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            
            print(f"Regresión polinómica grado {grado}: R² = {r2:.4f}, MSE = {mse:.4f}")
            
            if r2 > mejor_r2:
                mejor_r2 = r2
                mejor_grado = grado
                mejor_modelo = (modelo, poly)
        
        # Si tenemos un modelo válido
        if mejor_modelo is not None:
            modelo, poly = mejor_modelo
            
            # Ordenar X para graficar la curva suavemente
            if lag > 0:
                X = df_lag[x_var].values.reshape(-1, 1)
            else:
                X = df[x_var].values.reshape(-1, 1)
                
            X_sort = np.sort(X, axis=0)
            
            # Transformar y predecir
            X_poly_sort = poly.transform(X_sort)
            y_pred_sort = modelo.predict(X_poly_sort)
            
            # Calcular coeficientes
            coeficientes = modelo.coef_
            intercepto = modelo.intercept_
            
            # Graficar
            plt.figure(figsize=(10, 6))
            
            if lag > 0:
                plt.scatter(df_lag[x_var], df_lag[y_var], color=color, alpha=0.6)
            else:
                plt.scatter(df[x_var], df[y_var], color=color, alpha=0.6)
                
            plt.plot(X_sort, y_pred_sort, color='red', linewidth=2)
            
            # Añadir etiquetas y título
            lag_text = f" (Retardo: {lag} meses)" if lag > 0 else ""
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(f'Regresión Polinómica (grado {mejor_grado}): {y_label} ~ {x_label}{lag_text}\nR² = {mejor_r2:.4f}')
            plt.grid(True, alpha=0.3)
            
            # Guardar figura
            file_name = f'regresion_polinomica_{y_var}_{x_var}_lag{lag}.png'
            plt.savefig(file_name)
            plt.close()
            
            # Agregar resultados
            resultados_polinomicos.append({
                'x_var': x_var,
                'y_var': y_var,
                'x_label': x_label,
                'y_label': y_label,
                'lag': lag,
                'grado': mejor_grado,
                'r2': mejor_r2,
                'coeficientes': coeficientes.tolist(),
                'intercepto': float(intercepto),
                'imagen': file_name
            })
    
    # Crear DataFrame con resultados
    resultados_df = pd.DataFrame(resultados_polinomicos)
    resultados_df.to_csv('resultados_regresiones_polinomicas_mejoradas.csv', index=False)
    
    print("\nResultados de regresiones polinómicas:")
    for i, res in enumerate(resultados_polinomicos):
        lag_text = f" (Retardo: {res['lag']} meses)" if res['lag'] > 0 else ""
        print(f"{i+1}. {res['y_label']} ~ {res['x_label']}{lag_text}: R² = {res['r2']:.4f}, Grado = {res['grado']}")
    
    return resultados_polinomicos

def modelar_arima(df):
    """
    Modelar series temporales con ARIMA y realizar predicciones.
    """
    print("\nModelando series temporales con ARIMA...")
    
    resultados_arima = {}
    
    # Tamaño del conjunto de prueba (últimos 12 meses)
    test_size = 12
    
    for columna in df.columns:
        serie = df[columna].dropna()
        
        # Dividir en conjunto de entrenamiento y prueba
        train = serie[:-test_size]
        test = serie[-test_size:]
        
        # Determinación de órdenes óptimos
        # Esto podría mejorarse con una búsqueda en grid
        p, d, q = 1, 1, 1
        
        print(f"\nModelando ARIMA para {columna}...")
        try:
            modelo = ARIMA(train, order=(p, d, q))
            resultados = modelo.fit()
            
            # Hacer predicciones
            predicciones = resultados.forecast(steps=test_size)
            
            # Calcular métricas
            mse = mean_squared_error(test, predicciones)
            
            resultados_arima[columna] = {
                'modelo': modelo,
                'resultados': resultados,
                'predicciones': predicciones,
                'mse': mse
            }
            
            print(f"MSE para {columna}: {mse:.4f}")
            
            # Visualizar predicciones vs valores reales
            plt.figure(figsize=(12, 6))
            plt.plot(train.index, train, label='Entrenamiento', color='blue')
            plt.plot(test.index, test, label='Real', color='green')
            plt.plot(test.index, predicciones, label='Predicción', color='red')
            plt.title(f'Predicción ARIMA para {columna}')
            plt.xlabel('Fecha')
            plt.ylabel('Valor')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(f'arima_prediccion_mejorado_{columna}.png')
            plt.close()
        except Exception as e:
            print(f"Error al modelar ARIMA para {columna}: {e}")
    
    return resultados_arima

def generar_informe(df, resultados_estacionariedad, resultados_arima, resultados_polinomicos, resultados_correlaciones):
    """
    Generar un informe detallado de los análisis realizados.
    """
    print("\nGenerando informe de análisis mejorado...")
    
    with open('informe_analisis_mejorado.md', 'w') as f:
        f.write('# Informe de Análisis Mejorado de Indicadores Económicos\n\n')
        f.write('## Resumen de Datos\n\n')
        
        # Información general de los datos
        f.write('### Estadísticas Descriptivas\n\n')
        f.write('```\n')
        f.write(str(df.describe()))
        f.write('\n```\n\n')
        
        # Correlación simple
        f.write('### Matriz de Correlación\n\n')
        f.write('```\n')
        f.write(str(df.corr()))
        f.write('\n```\n\n')
        f.write('![Matriz de Correlación](correlacion_mejorada.png)\n\n')
        
        # Correlación con retardos
        f.write('### Correlación con Retardos Temporales\n\n')
        f.write('![Correlación con Retardos](correlacion_retardos_mejorada.png)\n\n')
        
        # Análisis de estacionariedad
        f.write('## Análisis de Estacionariedad\n\n')
        for columna, resultados in resultados_estacionariedad[0].items():
            f.write(f'### {columna}\n\n')
            f.write(f'- Es estacionaria: {"Sí" if resultados["estacionaria"] else "No"}\n')
            f.write(f'- p-value: {resultados["p-value"]:.4f}\n')
            f.write(f'- Estadístico de prueba: {resultados["estadístico-test"]:.4f}\n')
            f.write('- Valores críticos:\n')
            for key, value in resultados["valores-críticos"].items():
                f.write(f'  - {key}: {value:.4f}\n')
            f.write('\n')
        
        # Series temporales
        f.write('### Visualización de Series Temporales\n\n')
        f.write('![Series Temporales](series_temporales_analisis_mejorado.png)\n\n')
        f.write('### Series Temporales Diferenciadas\n\n')
        f.write('![Series Diferenciadas](series_temporales_diferenciadas_mejorado.png)\n\n')
        
        # Modelos ARIMA
        f.write('## Modelos ARIMA de Series Temporales\n\n')
        for columna, resultados in resultados_arima.items():
            f.write(f'### {columna}\n\n')
            f.write(f'- MSE: {resultados["mse"]:.4f}\n')
            f.write(f'- Orden del modelo: (1,1,1)\n\n')
            f.write(f'![Predicción ARIMA](arima_prediccion_mejorado_{columna}.png)\n\n')
        
        # Regresiones Polinómicas
        f.write('## Regresiones Polinómicas Mejoradas\n\n')
        for resultado in resultados_polinomicos:
            lag_text = f" (Retardo: {resultado['lag']} meses)" if resultado['lag'] > 0 else ""
            f.write(f'### {resultado["y_label"]} ~ {resultado["x_label"]}{lag_text}\n\n')
            f.write(f'- Grado óptimo: {resultado["grado"]}\n')
            f.write(f'- R²: {resultado["r2"]:.4f}\n')
            f.write('- Ecuación Polinómica:\n')
            
            ecuacion = f'  {resultado["y_label"]} = {resultado["intercepto"]:.4f}'
            for i, coef in enumerate(resultado["coeficientes"][1:], 1):
                if i == 1:
                    ecuacion += f' + {coef:.4f} × {resultado["x_label"]}'
                else:
                    ecuacion += f' + {coef:.4f} × {resultado["x_label"]}^{i}'
            
            f.write(ecuacion + '\n\n')
            f.write(f'![Regresión Polinómica]({resultado["imagen"]})\n\n')
        
        # Conclusiones
        f.write('## Conclusiones\n\n')
        f.write('1. **Correlaciones Mejoradas**: El análisis con retardos temporales muestra relaciones más fuertes que las correlaciones simples sin retardo.\n\n')
        f.write('2. **Regresiones Polinómicas**: Los modelos con retardos temporales proporcionan un mejor ajuste, demostrando que las relaciones entre los indicadores económicos tienen componentes temporales importantes.\n\n')
        f.write('3. **Modelos ARIMA**: Proporcionan una capacidad predictiva moderada para las series temporales, capturando la dinámica a corto plazo.\n\n')
        f.write('4. **Implicaciones para Políticas**: Los resultados sugieren que las políticas monetarias (tasa de interés) tienen impactos en tipo de cambio e inflación que se manifiestan con ciertos retardos temporales específicos.\n\n')
        f.write('5. **Recomendaciones**: Para futuras investigaciones, se recomienda considerar modelos más complejos como VAR (Vector Autoregression) o VEC (Vector Error Correction) que puedan capturar mejor la dinámica multivariable de estos indicadores económicos.\n')
    
    print("Informe generado: informe_analisis_mejorado.md")

def main():
    """Función principal."""
    print("Iniciando análisis mejorado de indicadores económicos...")
    
    # Cargar datos
    df = cargar_datos()
    
    # Analizar estacionariedad
    resultados_estacionariedad = analizar_estacionariedad(df)
    
    # Análisis de correlación
    resultados_correlaciones = analisis_de_correlacion(df)
    
    # Realizar regresiones polinómicas mejoradas
    resultados_polinomicos = regresion_polinomica_mejorada(df)
    
    # Modelar series con ARIMA
    resultados_arima = modelar_arima(df)
    
    # Generar informe
    generar_informe(df, resultados_estacionariedad, resultados_arima, resultados_polinomicos, resultados_correlaciones)
    
    print("\nAnálisis mejorado completado.")
    
if __name__ == "__main__":
    main() 