#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script ELT para crear tablas en AWS Glue y Athena.
"""

import os
import configparser
import boto3
import time

# Leer configuración usando rutas absolutas
config = configparser.ConfigParser()
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(os.path.dirname(script_dir), 'config.ini')
config.read(config_path)

# Configurar AWS
bucket_name = config['aws']['bucket_name']
region_name = config['aws']['region_name']

# Crear clientes de AWS
athena_client = boto3.client('athena', region_name=region_name)
glue_client = boto3.client('glue', region_name=region_name)

# Configurar Athena
s3_output = f's3://{bucket_name}/athena-results/'
database_name = 'econ'

def wait_for_query_completion(query_execution_id):
    """
    Esperar a que se complete una consulta en Athena.
    """
    while True:
        response = athena_client.get_query_execution(QueryExecutionId=query_execution_id)
        state = response['QueryExecution']['Status']['State']
        
        if state in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
            return state
        
        time.sleep(1)

def execute_athena_query(query):
    """
    Ejecutar una consulta en Athena.
    """
    response = athena_client.start_query_execution(
        QueryString=query,
        QueryExecutionContext={
            'Database': database_name
        },
        ResultConfiguration={
            'OutputLocation': s3_output,
        }
    )
    
    query_execution_id = response['QueryExecutionId']
    state = wait_for_query_completion(query_execution_id)
    
    if state == 'SUCCEEDED':
        print(f"Consulta completada exitosamente: {query_execution_id}")
        return query_execution_id
    else:
        print(f"Consulta fallida: {state}")
        return None

def create_database():
    """
    Crear la base de datos en AWS Glue si no existe.
    """
    try:
        glue_client.get_database(Name=database_name)
        print(f"La base de datos '{database_name}' ya existe.")
    except glue_client.exceptions.EntityNotFoundException:
        print(f"Creando base de datos '{database_name}'...")
        glue_client.create_database(
            DatabaseInput={
                'Name': database_name,
                'Description': 'Base de datos para análisis económicos',
            }
        )
        print(f"Base de datos '{database_name}' creada exitosamente.")

def create_tables():
    """
    Crear tablas en Athena.
    """
    print("Creando tablas en Athena...")
    
    # Tabla de tipo de cambio
    tipo_cambio_query = f"""
    CREATE EXTERNAL TABLE IF NOT EXISTS tipo_de_cambio (
        date DATE,
        tipo_de_cambio DOUBLE
    )
    ROW FORMAT DELIMITED
    FIELDS TERMINATED BY ','
    STORED AS TEXTFILE
    LOCATION 's3://{bucket_name}/raw/'
    TBLPROPERTIES ('skip.header.line.count'='1', 'serialization.null.format'='')
    """
    execute_athena_query(tipo_cambio_query)
    
    # Tabla de tasa de interés
    tasa_interes_query = f"""
    CREATE EXTERNAL TABLE IF NOT EXISTS tasa_de_interes (
        date DATE,
        tasa_de_interes DOUBLE
    )
    ROW FORMAT DELIMITED
    FIELDS TERMINATED BY ','
    STORED AS TEXTFILE
    LOCATION 's3://{bucket_name}/raw/'
    TBLPROPERTIES ('skip.header.line.count'='1', 'serialization.null.format'='')
    """
    execute_athena_query(tasa_interes_query)
    
    # Tabla de inflación
    inflacion_query = f"""
    CREATE EXTERNAL TABLE IF NOT EXISTS inflacion (
        date DATE,
        inflacion DOUBLE
    )
    ROW FORMAT DELIMITED
    FIELDS TERMINATED BY ','
    STORED AS TEXTFILE
    LOCATION 's3://{bucket_name}/raw/'
    TBLPROPERTIES ('skip.header.line.count'='1', 'serialization.null.format'='')
    """
    execute_athena_query(inflacion_query)
    
    print("Tablas creadas exitosamente en Athena.")

def create_combined_table():
    """
    Crear tabla combinada con tipo de cambio, tasa de interés e inflación.
    """
    print("Creando tabla combinada...")
    
    combined_query = f"""
    CREATE OR REPLACE VIEW indicadores_economicos AS
    SELECT
        tc.date,
        tc.tipo_de_cambio,
        ti.tasa_de_interes,
        inf.inflacion
    FROM
        tipo_de_cambio tc
    JOIN
        tasa_de_interes ti ON DATE_TRUNC('MONTH', tc.date) = DATE_TRUNC('MONTH', ti.date)
    JOIN
        inflacion inf ON DATE_TRUNC('MONTH', tc.date) = DATE_TRUNC('MONTH', inf.date)
    ORDER BY
        tc.date
    """
    execute_athena_query(combined_query)
    
    print("Tabla combinada creada exitosamente.")

def main():
    """
    Función principal.
    """
    print("Iniciando proceso ELT...")
    
    # Crear base de datos
    create_database()
    
    # Crear tablas
    create_tables()
    
    # Crear tabla combinada
    create_combined_table()
    
    print("Proceso ELT completado.")

if __name__ == "__main__":
    main() 