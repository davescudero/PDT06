.. Análisis de Indicadores Económicos documentation master file, created by
   sphinx-quickstart on Wed Mar 19 22:18:07 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Análisis de Indicadores Económicos
=================================

.. image:: /_static/mexico_flag.png
   :width: 100px
   :align: right
   :alt: Bandera de México

Bienvenido a la documentación del proyecto de Análisis de Indicadores Económicos de México. 
Este sistema integra procesos ETL/ELT, análisis de datos y visualizaciones a través de 
aplicaciones Streamlit para proporcionar insights sobre los principales indicadores económicos
del país.

.. toctree::
   :maxdepth: 2
   :caption: Contenido:

   introduccion
   arquitectura
   etl_proceso
   analisis_datos
   visualizacion
   desarrollo
   errores_soluciones
   conclusiones
   
Características Principales
--------------------------

* Extracción de datos económicos directamente de fuentes oficiales (Banxico e INEGI)
* Transformación y limpieza de datos para análisis estadístico
* Modelos de regresión para estudiar las relaciones entre indicadores
* Visualización interactiva de resultados con Streamlit
* Predicción temporal mediante modelos ARIMA

Estructura del Proyecto
----------------------

El proyecto está organizado en cuatro versiones principales que representan el progreso 
del desarrollo:

1. **v1_extraccion**: Procesos ETL/ELT para obtener datos económicos
2. **v2_analisis**: Scripts de análisis estadístico y generación de modelos
3. **v3_streamlit**: Aplicaciones de visualización en diferentes etapas
4. **v4_final**: Versión final optimizada y profesional

Indices y Tablas
==============

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
