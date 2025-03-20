Proceso ETL/ELT
=============

Este proyecto implementa una arquitectura ETL (Extract, Transform, Load) robusta para la obtención y 
procesamiento de datos económicos de México. Esta sección describe en detalle las etapas de este proceso.

Fuentes de Datos
--------------

El sistema está diseñado para extraer datos de dos fuentes oficiales:

1. **Banco de México (Banxico)**: API oficial para series de tiempo económicas
   - Tipo de cambio FIX (Peso mexicano frente al dólar estadounidense)
   - Tasa de interés interbancaria de equilibrio (TIIE) a 28 días

2. **Instituto Nacional de Estadística y Geografía (INEGI)**: API oficial para indicadores económicos
   - Índice Nacional de Precios al Consumidor (INPC) para cálculo de inflación

Ambas APIs requieren autenticación mediante tokens que se gestionan de forma segura a través de un 
archivo de configuración.

Extracción (Extract)
------------------

.. code-block:: python

   def extraer_tipo_cambio_banxico(token, fecha_inicio, fecha_fin):
       """
       Extrae datos del tipo de cambio de Banxico.
       
       Args:
           token: Token de autenticación para la API
           fecha_inicio: Fecha de inicio para la consulta
           fecha_fin: Fecha fin para la consulta
           
       Returns:
           DataFrame con los datos del tipo de cambio
       """
       url = f"https://www.banxico.org.mx/SieAPIRest/service/v1/series/SF43718/datos/{fecha_inicio}/{fecha_fin}"
       headers = {"Bmx-Token": token}
       response = requests.get(url, headers=headers)
       # Procesamiento de la respuesta...

El sistema implementa un mecanismo de redundancia para garantizar la disponibilidad de datos:

1. Intenta primero obtener datos de Banxico
2. Si falla, recurre a INEGI
3. Como último recurso, genera datos sintéticos realistas

Transformación (Transform)
-----------------------

Una vez obtenidos los datos crudos, se aplican diversas transformaciones:

1. **Normalización de formatos**:
   - Conversión uniforme de fechas
   - Estandarización de nombres de columnas
   - Transformación de tipos de datos

2. **Limpieza inicial**:
   - Eliminación de filas con valores nulos
   - Detección de duplicados
   - Ordenamiento cronológico

3. **Procesamiento avanzado**:
   - Cálculo de la inflación a partir del INPC
   - Agregación mensual de datos diarios
   - Alineación temporal entre las diferentes series

Carga (Load)
-----------

Los datos procesados se cargan en diferentes destinos:

1. **Almacenamiento local**:
   - Archivos CSV estructurados
   - Copias de respaldo con timestamps

2. **Almacenamiento en la nube**:
   - Amazon S3 para persistencia a largo plazo
   - Estructurado en buckets por tipo de dato y fecha

.. code-block:: python

   def cargar_en_s3(df, nombre_archivo, bucket):
       """
       Carga el DataFrame en un bucket de S3.
       
       Args:
           df: DataFrame a cargar
           nombre_archivo: Nombre del archivo en S3
           bucket: Nombre del bucket
       """
       csv_buffer = StringIO()
       df.to_csv(csv_buffer)
       s3_resource = boto3.resource('s3')
       s3_resource.Object(bucket, nombre_archivo).put(Body=csv_buffer.getvalue())

Limpieza Avanzada de Datos
------------------------

Posterior al proceso ETL básico, se realiza una limpieza más avanzada:

1. **Detección y tratamiento de outliers**:
   - Método de rango intercuartílico (IQR)
   - Evaluación de Z-scores

2. **Corrección de inconsistencias**:
   - Identificación de secuencias de valores idénticos poco realistas
   - Generación de variaciones realistas basadas en patrones estadísticos

3. **Validación final**:
   - Verificación de integridad referencial
   - Comprobación de rangos válidos para cada indicador
   - Análisis de consistencia temporal

Esta limpieza avanzada genera el conjunto de datos definitivo (`indicadores_economicos_clean_v2.csv`) 
que se utiliza para todos los análisis posteriores.

Diagrama del Proceso
------------------

.. code-block::

   +----------------+     +------------------+     +----------------+
   |                |     |                  |     |                |
   |  Extracción    |---->|  Transformación  |---->|    Carga      |
   |  (Banxico,     |     |  (Limpieza,      |     |  (Local, S3)  |
   |   INEGI)       |     |   Normalización) |     |                |
   +----------------+     +------------------+     +----------------+
                                                          |
                                                          v
                                              +------------------------+
                                              |                        |
                                              |  Limpieza avanzada     |
                                              |  (Outliers, Validación)|
                                              |                        |
                                              +------------------------+
                                                          |
                                                          v
                                              +------------------------+
                                              |                        |
                                              |  Conjunto final        |
                                              |  para análisis         |
                                              |                        |
                                              +------------------------+

La arquitectura modular del sistema permite su fácil mantenimiento y extensión para incorporar 
nuevas fuentes de datos o indicadores económicos adicionales en el futuro. 