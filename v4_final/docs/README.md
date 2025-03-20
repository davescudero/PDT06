# Tarea 06: Análisis de Indicadores Económicos con ETL/ELT y AWS

Este proyecto implementa un análisis completo de indicadores económicos mexicanos utilizando una arquitectura de procesamiento de datos basada en ETL (Extract, Transform, Load) y ELT (Extract, Load, Transform) con AWS.

## Objetivo

Analizar la relación entre tres indicadores económicos clave de México:
- 💱 **Tipo de cambio PESO/USD**
- 💹 **Tasa de interés**
- 📈 **Inflación**

Utilizando datos reales del Banco de México y el INEGI, procesados a través de una arquitectura moderna de analítica de datos en la nube.

## Arquitectura

![Arquitectura](../tareas/arquitectura05.png)

El proyecto implementa la siguiente arquitectura:
1. **ETL**: Extracción de datos de APIs del Banco de México y el INEGI
2. **Almacenamiento**: Amazon S3 para datos crudos
3. **Procesamiento**: AWS Glue y Amazon Athena para análisis SQL
4. **Análisis**: Modelos estadísticos y de Machine Learning en Python
5. **Visualización**: Aplicación web con Streamlit

## Componentes del Proyecto

### 1. ETL (Extract, Transform, Load)

El proceso ETL extrae datos del Banco de México y el INEGI, transforma los datos para hacerlos compatibles y los carga en Amazon S3.

```python
# Ejemplo de extracción de datos del Banco de México
url = f"https://www.banxico.org.mx/SieAPIRest/service/v1/series/{series_id}/datos/{start_date_str}/{end_date_str}"
headers = {'Bmx-Token': banxico_token}
```

### 2. ELT (Extract, Load, Transform)

El proceso ELT crea tablas en AWS Glue y utiliza Amazon Athena para realizar consultas SQL sobre los datos almacenados en S3.

```sql
-- Ejemplo de creación de tablas en Athena
CREATE EXTERNAL TABLE IF NOT EXISTS tipo_de_cambio (
    date DATE,
    tipo_de_cambio DOUBLE
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION 's3://bucket-name/raw/'
```

### 3. Análisis Avanzado

Se implementaron diversos modelos estadísticos y de Machine Learning:

- **Análisis de Series Temporales**: Modelos ARIMA para pronóstico
- **Regresiones Polinómicas**: Captura de relaciones no lineales entre variables
- **Pruebas de Estacionariedad**: Test de Dickey-Fuller para analizar comportamiento a largo plazo
- **Causalidad de Granger**: Análisis de relaciones causales entre indicadores económicos
- **Análisis de Promedio Móvil**: Identificación de tendencias a diferentes escalas temporales

### 4. Visualización con Streamlit

Una aplicación web interactiva desarrollada con Streamlit que permite explorar los datos, visualizar las relaciones y entender los resultados de los modelos.

## Resultados Clave

### Correlación Entre Variables

Los resultados muestran una correlación débil entre las variables estudiadas:

```
                 tipo_de_cambio  tasa_de_interes  inflacion
tipo_de_cambio         1.000000         0.046121   0.045490
tasa_de_interes        0.046121         1.000000   0.046121
inflacion              0.045490         0.046121   1.000000
```

### Modelos de Regresión Polinómica

Las regresiones polinómicas mejoran ligeramente los resultados de los modelos lineales, capturando relaciones más complejas:

- **Tipo de Cambio ~ Tasa de Interés**: R² = 0.0092 (grado 3)
- **Tasa de Interés ~ Inflación**: R² = 0.0094 (grado 3)
- **Tipo de Cambio ~ Inflación**: R² = 0.0092 (grado 3)

### Modelos ARIMA

Los modelos ARIMA proporcionan una capacidad predictiva moderada para las series temporales:

- **Tipo de Cambio**: MSE = 0.9571
- **Tasa de Interés**: MSE = 19.3385
- **Inflación**: MSE = 8.3385

### Causalidad de Granger

El análisis de causalidad de Granger revela relaciones temporales significativas:

- **La tasa de interés causa el tipo de cambio** con un retardo de 9 meses (p-value: 0.0216)
- **La tasa de interés causa la inflación** con un retardo de 9 meses (p-value: 0.0000)
- **La inflación causa el tipo de cambio** con un retardo de 4 meses (p-value: 0.0000)
- **La inflación causa la tasa de interés** con un retardo de 11 meses (p-value: 0.0000)

Esto sugiere que hay relaciones causales bidireccionales entre inflación y tasa de interés, y que ambas preceden a cambios en el tipo de cambio.

## Conclusiones

1. **Relaciones Complejas**: Las relaciones entre los indicadores económicos son más complejas que simples correlaciones lineales.

2. **Efectos Temporales**: Existen efectos retardados entre las variables, donde los cambios en una variable afectan a otra después de varios meses.

3. **Bidireccionalidad**: La relación entre la inflación y la tasa de interés es bidireccional, lo que refleja la complejidad de la economía y la política monetaria.

4. **Predictibilidad Limitada**: A pesar de utilizar modelos avanzados, la predictibilidad sigue siendo moderada, lo que refleja la influencia de factores externos no incluidos en el análisis.

5. **Implicaciones para Políticas**: Las decisiones de política monetaria (tasa de interés) tienen impactos en el tipo de cambio e inflación que se manifiestan con retardos temporales significativos.

## Tecnologías Utilizadas

- **Python**: Pandas, NumPy, Scikit-learn, Statsmodels
- **AWS**: S3, Athena, Glue
- **APIs**: Banco de México, INEGI
- **Visualización**: Matplotlib, Seaborn, Plotly, Streamlit

## Estructura del Proyecto

```
tarea06/
├── config.ini            # Configuración de APIs y AWS
├── etl/
│   └── extract_transform_load.py  # Script ETL
├── elt/
│   └── create_tables.py  # Script ELT para crear tablas
├── notebooks/
│   ├── analysis.py       # Análisis básico
│   └── analysis_advanced.py  # Análisis avanzado
├── streamlit_app.py      # Aplicación de visualización
└── informe_analisis_avanzado.md  # Reporte detallado
```

## Ejecución del Proyecto

1. Configurar credenciales en `config.ini`
2. Ejecutar el ETL: `python etl/extract_transform_load.py`
3. Ejecutar el ELT: `python elt/create_tables.py`
4. Ejecutar el análisis: `python notebooks/analysis_advanced.py`
5. Iniciar la aplicación: `streamlit run streamlit_app.py`

## Autor

**David Escudero**  
Instituto Tecnológico Autónomo de México (ITAM)  
Maestría en Ciencia de Datos  
Arquitectura de Producto de Datos

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - vea el archivo [LICENSE](LICENSE) para detalles.

---

*Desarrollado como parte del curso de Arquitectura de Producto de Datos - 2025* 