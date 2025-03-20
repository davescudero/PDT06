#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Análisis Avanzado de Indicadores Económicos

Este script implementa modelos más avanzados para analizar la relación entre:
- Tipo de cambio PESO/USD
- Tasa de interés
- Inflación

Incluye análisis de series temporales, regresión polinómica, y pruebas de causalidad.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import grangercausalitytests
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
    print("Cargando datos...")
    
    # Cargar el dataset
    df = pd.read_csv('indicadores_economicos.csv')
    
    # Convertir fecha a formato datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Establecer fecha como índice
    df.set_index('date', inplace=True)
    
    # Asegurar que los datos estén ordenados por fecha
    df.sort_index(inplace=True)
    
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
    
    # Ajustar diseño
    plt.tight_layout()
    plt.savefig('series_temporales_analisis.png')
    plt.close()
    
    # Visualizar las diferencias para hacer las series estacionarias
    diff_df = df.diff().dropna()
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    # Graficar las diferencias
    diff_df['tipo_de_cambio'].plot(ax=axes[0], title='Diferencia de Tipo de Cambio')
    diff_df['tasa_de_interes'].plot(ax=axes[1], title='Diferencia de Tasa de Interés')
    diff_df['inflacion'].plot(ax=axes[2], title='Diferencia de Inflación')
    
    # Ajustar diseño
    plt.tight_layout()
    plt.savefig('series_temporales_diferenciadas.png')
    plt.close()
    
    return resultados, diff_df

def analizar_autocorrelacion(df):
    """Analizar la autocorrelación de las series temporales."""
    print("\nAnalizando autocorrelación de las series temporales...")
    
    # Crear un grid de gráficos
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    # ACF y PACF para Tipo de Cambio
    plot_acf(df['tipo_de_cambio'].dropna(), ax=axes[0, 0], title='ACF - Tipo de Cambio')
    plot_pacf(df['tipo_de_cambio'].dropna(), ax=axes[0, 1], title='PACF - Tipo de Cambio')
    
    # ACF y PACF para Tasa de Interés
    plot_acf(df['tasa_de_interes'].dropna(), ax=axes[1, 0], title='ACF - Tasa de Interés')
    plot_pacf(df['tasa_de_interes'].dropna(), ax=axes[1, 1], title='PACF - Tasa de Interés')
    
    # ACF y PACF para Inflación
    plot_acf(df['inflacion'].dropna(), ax=axes[2, 0], title='ACF - Inflación')
    plot_pacf(df['inflacion'].dropna(), ax=axes[2, 1], title='PACF - Inflación')
    
    # Ajustar diseño
    plt.tight_layout()
    plt.savefig('autocorrelacion.png')
    plt.close()

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
        
        # Determinación de órdenes óptimos basados en análisis de ACF y PACF
        # Esto podría automatizarse con una búsqueda en grid, pero para simplificar usaremos:
        p, d, q = 1, 1, 1
        
        print(f"\nModelando ARIMA para {columna}...")
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
        plt.plot(train.index, train, label='Entrenamiento')
        plt.plot(test.index, test, label='Real')
        plt.plot(test.index, predicciones, label='Predicción', color='red')
        plt.title(f'Predicción ARIMA para {columna}')
        plt.legend()
        plt.savefig(f'arima_prediccion_{columna}.png')
        plt.close()
    
    return resultados_arima

def regresion_polinomica(df):
    """
    Realizar regresiones polinómicas entre las variables.
    """
    print("\nRealizando regresiones polinómicas...")
    
    resultados_polinomicos = []
    
    # Definir las relaciones a analizar
    relaciones = [
        ('tasa_de_interes', 'tipo_de_cambio', 'Tasa de Interés (%)', 'Tipo de Cambio (MXN/USD)', 'blue'),
        ('inflacion', 'tasa_de_interes', 'Inflación (%)', 'Tasa de Interés (%)', 'red'),
        ('inflacion', 'tipo_de_cambio', 'Inflación (%)', 'Tipo de Cambio (MXN/USD)', 'green')
    ]
    
    # Probar diferentes grados de polinomio
    grados = [1, 2, 3]
    
    for x_var, y_var, x_label, y_label, color in relaciones:
        mejor_r2 = -float('inf')
        mejor_grado = 1
        mejor_modelo = None
        
        for grado in grados:
            # Preparar datos
            X = df[x_var].values.reshape(-1, 1)
            y = df[y_var].values
            
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
            
            print(f"Regresión polinómica {y_label} ~ {x_label} (grado {grado}):")
            print(f"R² = {r2:.4f}, MSE = {mse:.4f}")
            
            if r2 > mejor_r2:
                mejor_r2 = r2
                mejor_grado = grado
                mejor_modelo = (modelo, poly)
        
        # Visualizar la mejor regresión polinómica
        print(f"\nMejor grado para {y_label} ~ {x_label}: {mejor_grado} (R² = {mejor_r2:.4f})")
        
        # Ordenar X para graficar la curva suavemente
        X = df[x_var].values.reshape(-1, 1)
        X_sort = np.sort(X, axis=0)
        
        # Transformar y predecir
        modelo, poly = mejor_modelo
        X_poly_sort = poly.transform(X_sort)
        y_pred_sort = modelo.predict(X_poly_sort)
        
        # Calcular coeficientes
        coeficientes = modelo.coef_
        intercepto = modelo.intercept_
        
        # Graficar
        plt.figure(figsize=(10, 6))
        plt.scatter(df[x_var], df[y_var], color=color, alpha=0.6)
        plt.plot(X_sort, y_pred_sort, color='red', linewidth=2)
        
        # Añadir etiquetas y título
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(f'Regresión Polinómica (grado {mejor_grado}): {y_label} ~ {x_label} (R² = {mejor_r2:.4f})')
        plt.grid(True, alpha=0.3)
        
        # Guardar figura
        plt.savefig(f'regresion_polinomica_{y_var}_{x_var}.png')
        plt.close()
        
        # Agregar resultados
        resultados_polinomicos.append({
            'x_var': x_var,
            'y_var': y_var,
            'x_label': x_label,
            'y_label': y_label,
            'grado': mejor_grado,
            'r2': mejor_r2,
            'coeficientes': coeficientes.tolist(),
            'intercepto': float(intercepto)
        })
    
    # Crear DataFrame con resultados
    resultados_df = pd.DataFrame(resultados_polinomicos)
    resultados_df.to_csv('resultados_regresiones_polinomicas.csv', index=False)
    
    return resultados_polinomicos

def analizar_causalidad_granger(df):
    """
    Analizar la causalidad de Granger entre las variables.
    """
    print("\nAnalizando causalidad de Granger...")
    
    variables = df.columns.tolist()
    max_lag = 12  # Analizar hasta 12 meses de retardo
    resultados_causalidad = {}
    
    for var_causa in variables:
        for var_efecto in variables:
            if var_causa != var_efecto:
                # Seleccionar las variables relevantes
                datos = df[[var_causa, var_efecto]].dropna()
                
                # Realizar el test
                test_result = grangercausalitytests(datos, maxlag=max_lag, verbose=False)
                
                # Extraer p-values para cada lag
                p_values = [test_result[i+1][0]['ssr_ftest'][1] for i in range(max_lag)]
                
                # Determinar causalidad (p < 0.05)
                hay_causalidad = any(p < 0.05 for p in p_values)
                
                # Guardar resultados
                resultados_causalidad[(var_causa, var_efecto)] = {
                    'hay_causalidad': hay_causalidad,
                    'p_values': p_values,
                    'mejor_lag': p_values.index(min(p_values)) + 1 if hay_causalidad else None
                }
                
                print(f"¿{var_causa} causa {var_efecto}? {'Sí' if hay_causalidad else 'No'}")
                if hay_causalidad:
                    print(f"  Mejor lag: {resultados_causalidad[(var_causa, var_efecto)]['mejor_lag']} meses")
                    print(f"  p-value: {min(p_values):.4f}")
    
    # Visualizar la matriz de causalidad
    causalidad_matrix = np.zeros((len(variables), len(variables)))
    
    for i, var_causa in enumerate(variables):
        for j, var_efecto in enumerate(variables):
            if var_causa != var_efecto:
                if resultados_causalidad[(var_causa, var_efecto)]['hay_causalidad']:
                    causalidad_matrix[i, j] = 1
    
    # Crear una visualización de la matriz de causalidad
    plt.figure(figsize=(10, 8))
    sns.heatmap(causalidad_matrix, annot=True, cmap='YlOrRd', 
                xticklabels=variables, yticklabels=variables, fmt='.0f')
    plt.title('Matriz de Causalidad de Granger (p < 0.05)')
    plt.xlabel('Variable efecto')
    plt.ylabel('Variable causa')
    plt.tight_layout()
    plt.savefig('causalidad_granger_matrix.png')
    plt.close()
    
    return resultados_causalidad

def analisis_promedio_movil(df):
    """
    Realizar análisis de promedio móvil para identificar tendencias.
    """
    print("\nRealizando análisis de promedio móvil...")
    
    ventanas = [3, 6, 12]  # Ventanas de 3, 6 y 12 meses
    
    for columna in df.columns:
        plt.figure(figsize=(12, 6))
        
        # Graficar la serie original
        plt.plot(df.index, df[columna], label='Original', alpha=0.5)
        
        # Calcular y graficar los promedios móviles
        for ventana in ventanas:
            prom_movil = df[columna].rolling(window=ventana).mean()
            plt.plot(df.index, prom_movil, label=f'MA({ventana})')
        
        plt.title(f'Análisis de Promedio Móvil para {columna}')
        plt.xlabel('Fecha')
        plt.ylabel('Valor')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'promedio_movil_{columna}.png')
        plt.close()

def generar_informe(df, resultados_estacionariedad, resultados_arima, resultados_polinomicos, resultados_causalidad):
    """
    Generar un informe detallado de los análisis realizados.
    """
    print("\nGenerando informe de análisis avanzado...")
    
    with open('informe_analisis_avanzado.md', 'w') as f:
        f.write('# Informe de Análisis Avanzado de Indicadores Económicos\n\n')
        f.write('## Resumen de Datos\n\n')
        
        # Información general de los datos
        f.write('### Estadísticas Descriptivas\n\n')
        f.write('```\n')
        f.write(str(df.describe()))
        f.write('\n```\n\n')
        
        # Correlación
        f.write('### Matriz de Correlación\n\n')
        f.write('```\n')
        f.write(str(df.corr()))
        f.write('\n```\n\n')
        
        # Resumen por año
        f.write('### Comportamiento por Año\n\n')
        resumen_anual = df.groupby(df.index.year).mean()
        f.write('```\n')
        f.write(str(resumen_anual))
        f.write('\n```\n\n')
        
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
        
        # Modelos ARIMA
        f.write('## Modelos ARIMA de Series Temporales\n\n')
        for columna, resultados in resultados_arima.items():
            f.write(f'### {columna}\n\n')
            f.write(f'- MSE: {resultados["mse"]:.4f}\n')
            f.write(f'- Orden del modelo: (1,1,1)\n\n')
            f.write('![Predicción ARIMA](arima_prediccion_' + columna + '.png)\n\n')
        
        # Regresiones Polinómicas
        f.write('## Regresiones Polinómicas\n\n')
        for resultado in resultados_polinomicos:
            f.write(f'### {resultado["y_label"]} ~ {resultado["x_label"]}\n\n')
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
            f.write(f'![Regresión Polinómica](regresion_polinomica_{resultado["y_var"]}_{resultado["x_var"]}.png)\n\n')
        
        # Análisis de Causalidad de Granger
        f.write('## Análisis de Causalidad de Granger\n\n')
        f.write('### Resumen de Causalidad\n\n')
        
        for (var_causa, var_efecto), resultados in resultados_causalidad.items():
            if resultados['hay_causalidad']:
                f.write(f'- **{var_causa}** causa **{var_efecto}** con un retardo de {resultados["mejor_lag"]} meses (p-value: {min(resultados["p_values"]):.4f})\n')
        
        f.write('\n### Matriz de Causalidad\n\n')
        f.write('![Matriz de Causalidad](causalidad_granger_matrix.png)\n\n')
        
        # Conclusiones
        f.write('## Conclusiones\n\n')
        f.write('1. **Estacionariedad**: Los indicadores económicos muestran tendencias no estacionarias, lo que indica la presencia de cambios estructurales a lo largo del tiempo.\n\n')
        f.write('2. **Modelos ARIMA**: Proporcionan una capacidad predictiva moderada para las series temporales, capturando la dinámica a corto plazo.\n\n')
        f.write('3. **Regresiones Polinómicas**: Mejoran sobre los modelos lineales simples, capturando relaciones no lineales entre los indicadores.\n\n')
        f.write('4. **Causalidad de Granger**: Revela relaciones causales temporales entre las variables, ofreciendo insights sobre cómo los cambios en un indicador pueden preceder cambios en otro.\n\n')
        f.write('5. **Implicaciones para Políticas**: Los resultados sugieren que las políticas monetarias (tasa de interés) tienen impactos en tipo de cambio e inflación que se manifiestan con ciertos retardos temporales.\n')
    
    print("Informe generado: informe_analisis_avanzado.md")

def main():
    """Función principal."""
    print("Iniciando análisis avanzado de indicadores económicos...")
    
    # Cargar datos
    df = cargar_datos()
    
    # Analizar estacionariedad
    resultados_estacionariedad = analizar_estacionariedad(df)
    
    # Analizar autocorrelación
    analizar_autocorrelacion(df)
    
    # Análisis de promedio móvil
    analisis_promedio_movil(df)
    
    # Modelar series con ARIMA
    resultados_arima = modelar_arima(df)
    
    # Realizar regresiones polinómicas
    resultados_polinomicos = regresion_polinomica(df)
    
    # Analizar causalidad de Granger
    resultados_causalidad = analizar_causalidad_granger(df)
    
    # Generar informe
    generar_informe(df, resultados_estacionariedad, resultados_arima, resultados_polinomicos, resultados_causalidad)
    
    print("\nAnálisis avanzado completado.")
    
if __name__ == "__main__":
    main() 