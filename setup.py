#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de configuración inicial para el proyecto de Análisis de Indicadores Económicos.

Este script crea la estructura de directorios necesaria y copia los archivos de ejemplo
para que el proyecto pueda ejecutarse correctamente incluso sin credenciales reales.
"""

import os
import shutil
import sys

def main():
    """Función principal de configuración"""
    print("Configurando el proyecto de Análisis de Indicadores Económicos...")
    
    # Obtener directorio base
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Crear directorios necesarios
    directories = [
        os.path.join(base_dir, 'v4_final', 'data'),
        os.path.join(base_dir, 'v4_final', 'img'),
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Creado directorio: {directory}")
    
    # Crear archivo config.ini si no existe
    config_example = os.path.join(base_dir, 'config.example.ini')
    config_file = os.path.join(base_dir, 'config.ini')
    
    if not os.path.exists(config_file) and os.path.exists(config_example):
        shutil.copy(config_example, config_file)
        print(f"Creado archivo de configuración: {config_file}")
        print("NOTA: Edite este archivo con sus propias credenciales si desea utilizar datos reales.")
    
    # Verificar la existencia del generador de datos de muestra
    sample_generator = os.path.join(base_dir, 'v4_final', 'data', 'sample_data_generator.py')
    if not os.path.exists(sample_generator):
        print("ADVERTENCIA: No se encontró el generador de datos de muestra.")
        print("La aplicación generará datos sintéticos automáticamente al ejecutarse.")
    
    print("\nConfiguración completada con éxito.")
    print("\nPara ejecutar la aplicación Streamlit:")
    print("  cd v4_final/app")
    print("  streamlit run streamlit_app_final.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 