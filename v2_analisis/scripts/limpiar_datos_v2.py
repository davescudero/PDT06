#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para la limpieza y preparación de datos económicos (versión 2)

Este script:
1. Carga los datos del archivo indicadores_economicos.csv
2. Elimina duplicados y valores atípicos
3. Maneja el problema de fechas no coincidentes entre datos de distintas fuentes
4. Guarda los datos limpios en indicadores_economicos_clean_v2.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

# Ignorar advertencias
warnings.filterwarnings('ignore')

def limpiar_datos_indicadores_v2():
    """
    Función principal para limpiar los datos económicos reales,
    manejando específicamente el problema de fechas no coincidentes.
    """
    print("Iniciando proceso de limpieza de datos (versión 2)...")
    
    # Cargar el archivo original
    archivo_original = 'indicadores_economicos.csv'
    
    try:
        df = pd.read_csv(archivo_original)
        print(f"Archivo original cargado: {len(df)} filas")
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {archivo_original}")
        print("Ejecutando el script ETL para generar el archivo...")
        try:
            from etl.extract_transform_load_improved import ejecutar_etl
            ejecutar_etl()
            df = pd.read_csv(archivo_original)
            print(f"Archivo generado y cargado: {len(df)} filas")
        except Exception as e:
            print(f"Error al ejecutar ETL: {e}")
            return None
    
    # Convertir fecha a datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Eliminar duplicados basados en la fecha (manteniendo la primera entrada)
    duplicados = df.duplicated(subset=['date'], keep='first')
    if duplicados.any():
        print(f"Eliminando {duplicados.sum()} entradas duplicadas...")
        df = df.drop_duplicates(subset=['date'], keep='first')
    
    # Ordenar por fecha
    df = df.sort_values('date')
    
    # Manejar valores nulos
    if df.isnull().any().any():
        print("Detectando valores nulos...")
        print(df.isnull().sum())
        
        # Identificar filas con valores nulos
        filas_nulas = df[df.isnull().any(axis=1)]
        print(f"Filas con valores nulos: {len(filas_nulas)}")
        
        # Estrategia: Alinear las fechas para cada variable
        
        # Paso 1: Separar los datos por variable
        df_tc = df[['date', 'tipo_de_cambio']].dropna()
        df_ti = df[['date', 'tasa_de_interes']].dropna()
        df_inf = df[['date', 'inflacion']].dropna()
        
        print(f"Registros de tipo de cambio: {len(df_tc)}")
        print(f"Registros de tasa de interés: {len(df_ti)}")
        print(f"Registros de inflación: {len(df_inf)}")
        
        # Paso 2: Encontrar fechas comunes (inner join)
        # Unir primero tipo de cambio y tasa de interés
        df_merged = pd.merge(df_tc, df_ti, on='date', how='inner')
        
        # Luego unir con inflación
        df_final = pd.merge(df_merged, df_inf, on='date', how='inner')
        
        print(f"Registros con todas las variables (inner join): {len(df_final)}")
        
        # Si hay muy pocos registros comunes, usar outer join y aplicar interpolación
        if len(df_final) < 20:
            print("Pocos registros comunes. Usando outer join con interpolación...")
            
            # Unir con outer join
            df_merged = pd.merge(df_tc, df_ti, on='date', how='outer')
            df_final = pd.merge(df_merged, df_inf, on='date', how='outer')
            
            # Ordenar por fecha
            df_final = df_final.sort_values('date')
            
            # Identificar valores extremos antes de interpolar
            for columna in ['tipo_de_cambio', 'tasa_de_interes', 'inflacion']:
                if df_final[columna].notna().sum() > 0:  # Si hay valores no nulos
                    # Calcular IQR
                    Q1 = df_final[columna].quantile(0.25)
                    Q3 = df_final[columna].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    # Definir límites
                    limite_inferior = Q1 - 1.5 * IQR
                    limite_superior = Q3 + 1.5 * IQR
                    
                    # Reemplazar valores extremos con NaN antes de interpolar
                    mask_extremos = (df_final[columna] < limite_inferior) | (df_final[columna] > limite_superior)
                    if mask_extremos.any():
                        print(f"Reemplazando {mask_extremos.sum()} valores extremos en {columna}")
                        df_final.loc[mask_extremos, columna] = np.nan
            
            # Aplicar interpolación
            df_final = df_final.interpolate(method='linear')
            
            # Verificar si aún quedan NaN (en extremos)
            if df_final.isnull().any().any():
                # Llenar los extremos con el método forward y backward fill
                df_final = df_final.fillna(method='ffill').fillna(method='bfill')
                print("Valores nulos restantes después de interpolación:", df_final.isnull().sum())
        
        df = df_final
    
    # Detección de valores atípicos (outliers) usando IQR
    print("\nDetectando valores atípicos...")
    
    for columna in ['tipo_de_cambio', 'tasa_de_interes', 'inflacion']:
        # Calcular IQR
        Q1 = df[columna].quantile(0.25)
        Q3 = df[columna].quantile(0.75)
        IQR = Q3 - Q1
        
        # Definir límites
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        
        # Identificar outliers
        outliers = df[(df[columna] < limite_inferior) | (df[columna] > limite_superior)]
        
        if not outliers.empty:
            print(f"Detectados {len(outliers)} valores atípicos en {columna}")
            print(f"Rango normal: {limite_inferior:.4f} - {limite_superior:.4f}")
            
            # Reemplazar outliers con la mediana de la columna
            mediana = df[columna].median()
            mask_outliers = (df[columna] < limite_inferior) | (df[columna] > limite_superior)
            df.loc[mask_outliers, columna] = mediana
            print(f"Valores atípicos en {columna} reemplazados con la mediana: {mediana:.4f}")
    
    # Visualizar las series temporales antes de guardar
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    # Graficar las series
    df.plot(x='date', y='tipo_de_cambio', ax=axes[0], title='Tipo de Cambio (MXN/USD)', legend=False)
    df.plot(x='date', y='tasa_de_interes', ax=axes[1], title='Tasa de Interés (%)', legend=False)
    df.plot(x='date', y='inflacion', ax=axes[2], title='Inflación (%)', legend=False)
    
    # Añadir grid
    for ax in axes:
        ax.grid(True, alpha=0.3)
    
    # Ajustar diseño
    plt.tight_layout()
    plt.savefig('series_temporales_limpias_v2.png')
    plt.close()
    
    # Calcular y mostrar matriz de correlación
    print("\nMatriz de correlación antes de guardar:")
    matriz_corr = df.drop('date', axis=1).corr()
    print(matriz_corr)
    
    # Visualizar matriz de correlación
    plt.figure(figsize=(8, 6))
    sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Matriz de Correlación - Variables Económicas')
    plt.tight_layout()
    plt.savefig('correlacion_limpia_v2.png')
    plt.close()
    
    # Guardar datos limpios
    archivo_limpio = 'indicadores_economicos_clean_v2.csv'
    df.to_csv(archivo_limpio, index=False)
    print(f"\nDatos limpios guardados en {archivo_limpio}: {len(df)} filas")
    
    return df

if __name__ == "__main__":
    df_limpio = limpiar_datos_indicadores_v2()
    if df_limpio is not None:
        print("\nProceso de limpieza completado exitosamente.")
    else:
        print("\nEl proceso de limpieza falló.") 