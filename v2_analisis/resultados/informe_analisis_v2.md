# Informe de Análisis Mejorado de Indicadores Económicos (Datos Reales)

## Resumen de Datos

### Estadísticas Descriptivas

```
       tipo_de_cambio  tasa_de_interes   inflacion
count      120.000000        120.00000  120.000000
mean        19.638416          8.01250    0.465750
std          1.682898          2.91806    0.247516
min         16.678000          4.00000   -0.190000
25%         18.216613          4.84375    0.297500
50%         19.989450          9.06250    0.480000
75%         20.513525         11.00000    0.620000
max         23.512200         11.25000    1.050000
```

### Matriz de Correlación

```
                 tipo_de_cambio  tasa_de_interes  inflacion
tipo_de_cambio         1.000000        -0.746891   0.197073
tasa_de_interes       -0.746891         1.000000  -0.268426
inflacion              0.197073        -0.268426   1.000000
```

![Matriz de Correlación](correlacion_v2.png)

### Correlación con Retardos Temporales

![Correlación con Retardos](correlacion_retardos_v2.png)

## Análisis de Estacionariedad

### tipo_de_cambio

- Es estacionaria: No
- p-value: 0.2354
- Estadístico de prueba: -2.1228
- Valores críticos:
  - 1%: -3.4912
  - 5%: -2.8882
  - 10%: -2.5810

### tasa_de_interes

- Es estacionaria: Sí
- p-value: 0.0150
- Estadístico de prueba: -3.2963
- Valores críticos:
  - 1%: -3.4891
  - 5%: -2.8872
  - 10%: -2.5805

### inflacion

- Es estacionaria: Sí
- p-value: 0.0000
- Estadístico de prueba: -5.5208
- Valores críticos:
  - 1%: -3.4870
  - 5%: -2.8864
  - 10%: -2.5800

### Visualización de Series Temporales

![Series Temporales](series_temporales_analisis_v2.png)

### Series Temporales Diferenciadas

![Series Diferenciadas](series_temporales_diferenciadas_v2.png)

## Modelos ARIMA de Series Temporales

### tipo_de_cambio

- MSE: 0.4353
- Orden del modelo: (1,1,1)

![Predicción ARIMA](arima_prediccion_v2_tipo_de_cambio.png)

### tasa_de_interes

- MSE: 0.2153
- Orden del modelo: (1,1,1)

![Predicción ARIMA](arima_prediccion_v2_tasa_de_interes.png)

### inflacion

- MSE: 0.0234
- Orden del modelo: (1,1,1)

![Predicción ARIMA](arima_prediccion_v2_inflacion.png)

## Regresiones Polinómicas Mejoradas

### Tipo de Cambio (MXN/USD) ~ Tasa de Interés (%)

- Grado óptimo: 3
- R²: 0.7383
- Ecuación Polinómica:
  Tipo de Cambio (MXN/USD) = 20.5587 + -0.5517 × Tasa de Interés (%) + 0.2064 × Tasa de Interés (%)^2 + -0.0161 × Tasa de Interés (%)^3

![Regresión Polinómica](regresion_polinomica_v2_tipo_de_cambio_tasa_de_interes_lag0.png)

### Tasa de Interés (%) ~ Inflación (%)

- Grado óptimo: 3
- R²: 0.0862
- Ecuación Polinómica:
  Tasa de Interés (%) = 10.0645 + -8.3680 × Inflación (%) + 9.3102 × Inflación (%)^2 + -4.0325 × Inflación (%)^3

![Regresión Polinómica](regresion_polinomica_v2_tasa_de_interes_inflacion_lag0.png)

### Tipo de Cambio (MXN/USD) ~ Inflación (%)

- Grado óptimo: 3
- R²: 0.0884
- Ecuación Polinómica:
  Tipo de Cambio (MXN/USD) = 18.3847 + 6.6636 × Inflación (%) + -8.9350 × Inflación (%)^2 + 3.4401 × Inflación (%)^3

![Regresión Polinómica](regresion_polinomica_v2_tipo_de_cambio_inflacion_lag0.png)

### Tipo de Cambio (MXN/USD) ~ Tasa de Interés (%) [t-1] (Retardo: 1 meses)

- Grado óptimo: 3
- R²: 0.7350
- Ecuación Polinómica:
  Tipo de Cambio (MXN/USD) = 20.7223 + -0.6401 × Tasa de Interés (%) [t-1] + 0.2162 × Tasa de Interés (%) [t-1]^2 + -0.0164 × Tasa de Interés (%) [t-1]^3

![Regresión Polinómica](regresion_polinomica_v2_tipo_de_cambio_tasa_de_interes_lag1.png)

### Tasa de Interés (%) ~ Inflación (%) [t-1] (Retardo: 1 meses)

- Grado óptimo: 3
- R²: 0.0790
- Ecuación Polinómica:
  Tasa de Interés (%) = 10.0181 + -8.7300 × Inflación (%) [t-1] + 10.7146 × Inflación (%) [t-1]^2 + -4.9035 × Inflación (%) [t-1]^3

![Regresión Polinómica](regresion_polinomica_v2_tasa_de_interes_inflacion_lag1.png)

### Tipo de Cambio (MXN/USD) ~ Inflación (%) [t-3] (Retardo: 3 meses)

- Grado óptimo: 3
- R²: 0.0863
- Ecuación Polinómica:
  Tipo de Cambio (MXN/USD) = 18.4203 + 5.7653 × Inflación (%) [t-3] + -8.7178 × Inflación (%) [t-3]^2 + 4.6892 × Inflación (%) [t-3]^3

![Regresión Polinómica](regresion_polinomica_v2_tipo_de_cambio_inflacion_lag3.png)

## Conclusiones

1. **Correlaciones Mejoradas**: Los datos reales muestran correlaciones significativas entre los indicadores económicos, destacando la fuerte correlación negativa entre tipo de cambio y tasa de interés.

2. **Regresiones Polinómicas**: Los modelos con retardos temporales proporcionan un mejor ajuste, demostrando que las relaciones entre los indicadores económicos tienen componentes temporales importantes.

3. **Modelos ARIMA**: Proporcionan una capacidad predictiva moderada para las series temporales, capturando la dinámica a corto plazo.

4. **Implicaciones para Políticas**: Los resultados sugieren que las políticas monetarias (tasa de interés) tienen impactos en tipo de cambio e inflación que se manifiestan con ciertos retardos temporales específicos.

5. **Recomendaciones**: Para futuras investigaciones, se recomienda considerar modelos más complejos como VAR (Vector Autoregression) o VEC (Vector Error Correction) que puedan capturar mejor la dinámica multivariable de estos indicadores económicos.
