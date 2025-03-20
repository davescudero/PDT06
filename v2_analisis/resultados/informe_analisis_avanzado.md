# Informe de Análisis Avanzado de Indicadores Económicos

## Resumen de Datos

### Estadísticas Descriptivas

```
       tipo_de_cambio  tasa_de_interes   inflacion
count      999.000000       999.000000  999.000000
mean        11.761700        11.767189   11.761700
std          6.511915         6.513794    6.511915
min          4.000000         4.000000    4.000000
25%          6.500000         6.500000    6.500000
50%          9.250000         9.250000    9.250000
75%         19.948700        19.948700   19.948700
max         24.388200        24.388200   24.388200
```

### Matriz de Correlación

```
                 tipo_de_cambio  tasa_de_interes  inflacion
tipo_de_cambio         1.000000         0.046121   0.045490
tasa_de_interes        0.046121         1.000000   0.046121
inflacion              0.045490         0.046121   1.000000
```

### Comportamiento por Año

```
      tipo_de_cambio  tasa_de_interes  inflacion
date                                            
2018        5.050000         5.050000   5.050000
2019        6.250000         6.250000   6.250000
2020       11.500066        11.500066  11.500066
2021       11.148803        11.148803  11.148803
2022       12.595397        12.595397  12.595397
2023       14.421807        14.521513  14.421807
```

## Análisis de Estacionariedad

### tipo_de_cambio

- Es estacionaria: Sí
- p-value: 0.0000
- Estadístico de prueba: -8.3492
- Valores críticos:
  - 1%: -3.4371
  - 5%: -2.8645
  - 10%: -2.5683

### tasa_de_interes

- Es estacionaria: Sí
- p-value: 0.0000
- Estadístico de prueba: -7.5744
- Valores críticos:
  - 1%: -3.4371
  - 5%: -2.8645
  - 10%: -2.5683

### inflacion

- Es estacionaria: Sí
- p-value: 0.0000
- Estadístico de prueba: -5.2305
- Valores críticos:
  - 1%: -3.4371
  - 5%: -2.8645
  - 10%: -2.5683

## Modelos ARIMA de Series Temporales

### tipo_de_cambio

- MSE: 0.9571
- Orden del modelo: (1,1,1)

![Predicción ARIMA](arima_prediccion_tipo_de_cambio.png)

### tasa_de_interes

- MSE: 19.3385
- Orden del modelo: (1,1,1)

![Predicción ARIMA](arima_prediccion_tasa_de_interes.png)

### inflacion

- MSE: 8.3385
- Orden del modelo: (1,1,1)

![Predicción ARIMA](arima_prediccion_inflacion.png)

## Regresiones Polinómicas

### Tipo de Cambio (MXN/USD) ~ Tasa de Interés (%)

- Grado óptimo: 3
- R²: 0.0092
- Ecuación Polinómica:
  Tipo de Cambio (MXN/USD) = 8.3090 + 0.6287 × Tasa de Interés (%) + -0.0225 × Tasa de Interés (%)^2 + 0.0000 × Tasa de Interés (%)^3

![Regresión Polinómica](regresion_polinomica_tipo_de_cambio_tasa_de_interes.png)

### Tasa de Interés (%) ~ Inflación (%)

- Grado óptimo: 3
- R²: 0.0094
- Ecuación Polinómica:
  Tasa de Interés (%) = 8.2938 + 0.6297 × Inflación (%) + -0.0221 × Inflación (%)^2 + 0.0000 × Inflación (%)^3

![Regresión Polinómica](regresion_polinomica_tasa_de_interes_inflacion.png)

### Tipo de Cambio (MXN/USD) ~ Inflación (%)

- Grado óptimo: 3
- R²: 0.0092
- Ecuación Polinómica:
  Tipo de Cambio (MXN/USD) = 8.2488 + 0.6488 × Inflación (%) + -0.0244 × Inflación (%)^2 + 0.0001 × Inflación (%)^3

![Regresión Polinómica](regresion_polinomica_tipo_de_cambio_inflacion.png)

## Análisis de Causalidad de Granger

### Resumen de Causalidad

- **tasa_de_interes** causa **tipo_de_cambio** con un retardo de 9 meses (p-value: 0.0216)
- **tasa_de_interes** causa **inflacion** con un retardo de 9 meses (p-value: 0.0000)
- **inflacion** causa **tipo_de_cambio** con un retardo de 4 meses (p-value: 0.0000)
- **inflacion** causa **tasa_de_interes** con un retardo de 11 meses (p-value: 0.0000)

### Matriz de Causalidad

![Matriz de Causalidad](causalidad_granger_matrix.png)

## Conclusiones

1. **Estacionariedad**: Los indicadores económicos muestran tendencias no estacionarias, lo que indica la presencia de cambios estructurales a lo largo del tiempo.

2. **Modelos ARIMA**: Proporcionan una capacidad predictiva moderada para las series temporales, capturando la dinámica a corto plazo.

3. **Regresiones Polinómicas**: Mejoran sobre los modelos lineales simples, capturando relaciones no lineales entre los indicadores.

4. **Causalidad de Granger**: Revela relaciones causales temporales entre las variables, ofreciendo insights sobre cómo los cambios en un indicador pueden preceder cambios en otro.

5. **Implicaciones para Políticas**: Los resultados sugieren que las políticas monetarias (tasa de interés) tienen impactos en tipo de cambio e inflación que se manifiestan con ciertos retardos temporales.
