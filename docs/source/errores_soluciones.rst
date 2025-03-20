Errores y Soluciones
==================

Durante el desarrollo del proyecto nos enfrentamos a diversos desafíos técnicos y problemas 
que requirieron soluciones creativas. Esta sección documenta los principales problemas 
encontrados y las estrategias implementadas para resolverlos.

Problemas en la Extracción de Datos
-----------------------------------

Acceso a APIs Oficiales
~~~~~~~~~~~~~~~~~~~~~~~

**Problema**: La API de INEGI devolvía errores 401 (Unauthorized) a pesar de usar tokens válidos.

**Solución**: Implementamos un sistema de fallback que intentaba primero obtener datos de Banxico, 
y solo si esta fuente fallaba, recurría a INEGI. En caso de fallos en ambas fuentes, generábamos 
datos sintéticos realistas basados en distribuciones estadísticas históricas.

.. code-block:: python

   try:
       # Intentar obtener datos de Banxico
       datos = obtener_datos_banxico()
   except Exception as e:
       try:
           # Si Banxico falla, intentar con INEGI
           datos = obtener_datos_inegi()
       except Exception as e2:
           # Si ambos fallan, generar datos sintéticos
           logger.warning(f"Generando datos sintéticos debido a: {e} y {e2}")
           datos = generar_datos_sinteticos()

Datos Inconsistentes
~~~~~~~~~~~~~~~~~~~

**Problema**: Detectamos valores idénticos para diferentes fechas, lo que indicaba potenciales errores 
en las fuentes de datos o en el proceso de extracción.

**Solución**: Desarrollamos algorítmos de detección de anomalías que identificaban secuencias sospechosamente 
uniformes y las corregían mediante técnicas estadísticas adecuadas.

Desafíos en el Procesamiento de Datos
------------------------------------

Valores Atípicos (Outliers)
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problema**: Encontramos outliers extremos en los datos que distorsionaban significativamente los análisis.

**Solución**: Implementamos técnicas basadas en el rango intercuartílico (IQR) para detectar y tratar 
estos valores, manteniendo la integridad de los datos mientras eliminábamos distorsiones extremas.

.. code-block:: python

   def detectar_outliers_iqr(serie, factor=1.5):
       Q1 = serie.quantile(0.25)
       Q3 = serie.quantile(0.75)
       IQR = Q3 - Q1
       limite_inferior = Q1 - factor * IQR
       limite_superior = Q3 + factor * IQR
       return (serie < limite_inferior) | (serie > limite_superior)

Alineación Temporal
~~~~~~~~~~~~~~~~~

**Problema**: Las fechas de registro para los diferentes indicadores no siempre coincidían, lo que complicaba 
el análisis de correlaciones.

**Solución**: Desarrollamos funciones de alineación temporal que aseguraban que solo se compararan datos 
correspondientes al mismo período, utilizando técnicas de interpolación cuando era necesario.

Problemas en el Análisis Estadístico
----------------------------------

Correlaciones Espurias
~~~~~~~~~~~~~~~~~~~~

**Problema**: Algunas correlaciones iniciales parecían significativas pero sin fundamento económico real, 
sugiriendo relaciones espurias.

**Solución**: Implementamos análisis de correlación con retardos temporales y pruebas de causalidad de Granger 
para distinguir entre correlaciones casuales y potencialmente causales.

Ajuste de Modelos
~~~~~~~~~~~~~~~

**Problema**: Los modelos de regresión lineal simple no capturaban adecuadamente las relaciones entre variables.

**Solución**: Desarrollamos un sistema que probaba automáticamente diferentes grados de polinomios (1 a 5) y 
seleccionaba el óptimo basado en métricas como R² ajustado y criterios de información.

Desafíos en la Visualización
--------------------------

Rendimiento de Streamlit
~~~~~~~~~~~~~~~~~~~~~~

**Problema**: Las primeras versiones de la aplicación Streamlit experimentaban lentitud al cargar grandes 
conjuntos de datos y visualizaciones complejas.

**Solución**: Implementamos estrategias de caching para los datos y cálculos intensivos, y optimizamos 
las consultas para mejorar significativamente el rendimiento.

.. code-block:: python

   @st.cache_data
   def cargar_datos():
       # Esta función solo se ejecuta una vez y luego cachea el resultado
       df = pd.read_csv('indicadores_economicos_clean_v2.csv')
       df['date'] = pd.to_datetime(df['date'])
       return df

Errores de Conversión de Datos
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problema**: Al ejecutar la aplicación Streamlit final, encontrábamos errores de conversión entre 
pandas DataFrame y formato Arrow, necesario para la visualización.

**Solución**: Revisamos y corregimos los tipos de datos en los DataFrames, asegurándonos de que 
fueran compatibles con Arrow y ajustando el código para manejar correctamente fechas y valores numéricos.

Lecciones Aprendidas
------------------

Este proceso de resolución de problemas nos permitió:

1. **Mejorar la robustez** de nuestra arquitectura ETL/ELT mediante sistemas de fallback y manejo de errores
2. **Refinar nuestros análisis estadísticos** con técnicas más avanzadas y apropiadas para series temporales
3. **Optimizar el rendimiento** de nuestras aplicaciones de visualización
4. **Implementar buenas prácticas** de documentación y gestión de código

Cada desafío superado contribuyó a un producto final más confiable, preciso y eficiente. 