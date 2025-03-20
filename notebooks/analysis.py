#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Análisis de Indicadores Económicos: Tipo de Cambio, Tasa de Interés e Inflación

Este script realiza un análisis de la relación entre los indicadores económicos de México, específicamente:
- Tipo de cambio PESO/USD
- Tasa de interés
- Inflación

Utilizaremos los datos almacenados en Amazon Athena para realizar regresiones lineales y 
visualizar las relaciones entre estas variables.
"""

# Importar librerías necesarias
import os
import boto3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
import configparser
import time

# Configurar estilo de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("notebook")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Leer configuración usando rutas absolutas
config = configparser.ConfigParser()
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(os.path.dirname(script_dir), 'config.ini')
config.read(config_path)

# Configurar AWS
bucket_name = config['aws']['bucket_name']
region_name = config['aws']['region_name']

# Configurar Athena
database_name = 'econ'
s3_output = f's3://{bucket_name}/athena-results/'

# Función para ejecutar consultas en Athena
def run_query(query):
    """Ejecutar una consulta en Athena y devolver los resultados como DataFrame."""
    athena_client = boto3.client('athena', region_name=region_name)
    s3_client = boto3.client('s3', region_name=region_name)
    
    # Ejecutar consulta
    response = athena_client.start_query_execution(
        QueryString=query,
        QueryExecutionContext={
            'Database': database_name
        },
        ResultConfiguration={
            'OutputLocation': s3_output,
        }
    )
    
    # Obtener ID de ejecución
    query_execution_id = response['QueryExecutionId']
    
    # Esperar a que se complete la consulta
    while True:
        response = athena_client.get_query_execution(QueryExecutionId=query_execution_id)
        state = response['QueryExecution']['Status']['State']
        
        if state in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
            break
            
        time.sleep(1)
    
    if state == 'SUCCEEDED':
        # Obtener resultados
        result_response = athena_client.get_query_results(QueryExecutionId=query_execution_id)
        
        # Extraer columnas
        columns = [col['Label'] for col in result_response['ResultSet']['ResultSetMetadata']['ColumnInfo']]
        
        # Extraer datos
        rows = []
        for row in result_response['ResultSet']['Rows'][1:]:
            data = []
            for i, col in enumerate(row['Data']):
                if 'VarCharValue' in col:
                    data.append(col['VarCharValue'])
                else:
                    data.append(None)
            rows.append(data)
        
        # Crear DataFrame
        return pd.DataFrame(rows, columns=columns)
    else:
        print(f"Error en la consulta: {state}")
        return None

# Obtener datos de Athena
def obtener_datos():
    print("Obteniendo datos de Athena...")
    
    # Consulta para obtener los datos combinados
    query = """
    SELECT
        date,
        tipo_de_cambio,
        tasa_de_interes,
        inflacion
    FROM
        indicadores_economicos
    ORDER BY
        date
    """
    
    # Ejecutar consulta
    df = run_query(query)
    
    # Convertir tipos de datos
    df['date'] = pd.to_datetime(df['date'])
    df['tipo_de_cambio'] = pd.to_numeric(df['tipo_de_cambio'])
    df['tasa_de_interes'] = pd.to_numeric(df['tasa_de_interes'])
    df['inflacion'] = pd.to_numeric(df['inflacion'])
    
    return df

# Visualizar series de tiempo
def visualizar_series_tiempo(df):
    print("Visualizando series de tiempo...")
    
    # Crear subplots para visualizar las series de tiempo
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    # Tipo de cambio
    axes[0].plot(df['date'], df['tipo_de_cambio'], color='blue')
    axes[0].set_ylabel('Tipo de Cambio (MXN/USD)')
    axes[0].set_title('Tipo de Cambio PESO/USD')
    axes[0].grid(True)
    
    # Tasa de interés
    axes[1].plot(df['date'], df['tasa_de_interes'], color='red')
    axes[1].set_ylabel('Tasa de Interés (%)')
    axes[1].set_title('Tasa de Interés')
    axes[1].grid(True)
    
    # Inflación
    axes[2].plot(df['date'], df['inflacion'], color='green')
    axes[2].set_ylabel('Inflación (%)')
    axes[2].set_xlabel('Fecha')
    axes[2].set_title('Inflación')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('series_tiempo.png')
    plt.close()

# Función para realizar regresión lineal y visualizar
def realizar_regresion(x, y, x_label, y_label, color='blue', filename=None):
    """Realizar regresión lineal y visualizar los resultados."""
    print(f"Realizando regresión: {y_label} ~ {x_label}...")
    
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
    
    # Crear gráfico
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color=color, alpha=0.6)
    plt.plot(x, y_pred, color='red', linewidth=2)
    
    # Añadir etiquetas y título
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'{y_label} ~ {x_label} (R² = {r2:.4f}, MSE = {mse:.4f})')
    plt.grid(True, alpha=0.3)
    
    # Añadir ecuación de regresión
    equation = f'{y_label} = {model.coef_[0]:.4f} * {x_label} + {model.intercept_:.4f}'
    plt.annotate(equation, xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=12, ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename)
    plt.close()
    
    # Devolver resultados del modelo
    return {
        'coef': model.coef_[0],
        'intercept': model.intercept_,
        'r2': r2,
        'mse': mse
    }

# Analizar correlación
def analizar_correlacion(df):
    print("Analizando correlación entre variables...")
    
    # Matriz de correlación
    corr = df[['tipo_de_cambio', 'tasa_de_interes', 'inflacion']].corr()
    
    # Visualizar matriz de correlación
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".4f", linewidths=0.5)
    plt.title('Matriz de Correlación entre Variables Económicas')
    plt.tight_layout()
    plt.savefig('correlacion.png')
    plt.close()
    
    return corr

# Función principal
def main():
    # Obtener datos
    df = obtener_datos()
    
    if df is not None and not df.empty:
        print(f"Datos obtenidos: {len(df)} registros")
        
        # Visualizar series de tiempo
        visualizar_series_tiempo(df)
        
        # Realizar regresiones
        resultados_tc_ti = realizar_regresion(
            df['tasa_de_interes'], 
            df['tipo_de_cambio'], 
            'Tasa de Interés (%)', 
            'Tipo de Cambio (MXN/USD)',
            color='blue',
            filename='reg_tc_ti.png'
        )
        
        resultados_ti_inf = realizar_regresion(
            df['inflacion'], 
            df['tasa_de_interes'], 
            'Inflación (%)', 
            'Tasa de Interés (%)',
            color='red',
            filename='reg_ti_inf.png'
        )
        
        resultados_tc_inf = realizar_regresion(
            df['inflacion'], 
            df['tipo_de_cambio'], 
            'Inflación (%)', 
            'Tipo de Cambio (MXN/USD)',
            color='green',
            filename='reg_tc_inf.png'
        )
        
        # Analizar correlación
        corr = analizar_correlacion(df)
        
        # Crear tabla de resumen
        resultados = pd.DataFrame({
            'Regresión': [
                'Tipo de Cambio ~ Tasa de Interés',
                'Tasa de Interés ~ Inflación',
                'Tipo de Cambio ~ Inflación'
            ],
            'Coeficiente': [
                resultados_tc_ti['coef'],
                resultados_ti_inf['coef'],
                resultados_tc_inf['coef']
            ],
            'Intercepto': [
                resultados_tc_ti['intercept'],
                resultados_ti_inf['intercept'],
                resultados_tc_inf['intercept']
            ],
            'R²': [
                resultados_tc_ti['r2'],
                resultados_ti_inf['r2'],
                resultados_tc_inf['r2']
            ],
            'MSE': [
                resultados_tc_ti['mse'],
                resultados_ti_inf['mse'],
                resultados_tc_inf['mse']
            ]
        })
        
        print("\nResumen de resultados:")
        print(resultados)
        
        # Guardar resultados
        resultados.to_csv('resultados_regresiones.csv', index=False)
        df.to_csv('indicadores_economicos.csv', index=False)
        
        print("\nAnálisis completado. Archivos generados:")
        print("- indicadores_economicos.csv: Datos combinados")
        print("- resultados_regresiones.csv: Resumen de regresiones")
        print("- series_tiempo.png: Visualización de series de tiempo")
        print("- reg_tc_ti.png: Regresión de Tipo de Cambio ~ Tasa de Interés")
        print("- reg_ti_inf.png: Regresión de Tasa de Interés ~ Inflación")
        print("- reg_tc_inf.png: Regresión de Tipo de Cambio ~ Inflación")
        print("- correlacion.png: Matriz de correlación")
        
    else:
        print("No se pudieron obtener datos de Athena.")

if __name__ == "__main__":
    main() 