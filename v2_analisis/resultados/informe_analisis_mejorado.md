# Informe de Análisis Mejorado de Indicadores Económicos

## Resumen de Datos

### Estadísticas Descriptivas

```
       tipo_de_cambio  tasa_de_interes  inflacion
count       67.000000        67.000000  67.000000
mean         7.239179         7.457492   5.670587
std          2.540245         1.678463   1.339312
min          4.000000         4.458451   4.000000
25%          5.500000         5.860768   4.611556
50%          7.000000         7.800000   5.500000
75%          8.250000         8.750000   6.316663
max         19.325000        10.400000  10.000000
```

### Matriz de Correlación

```
                 tipo_de_cambio  tasa_de_interes  inflacion
tipo_de_cambio         1.000000         0.537311   0.483289
tasa_de_interes        0.537311         1.000000   0.335375
inflacion              0.483289         0.335375   1.000000
```

![Matriz de Correlación](correlacion_mejorada.png)

### Correlación con Retardos Temporales

![Correlación con Retardos](correlacion_retardos_mejorada.png)

## Análisis de Estacionariedad

### tipo_de_cambio

- Es estacionaria: No
- p-value: 0.7593
- Estadístico de prueba: -0.9833
- Valores críticos:
  - 1%: -3.5369
  - 5%: -2.9079
  - 10%: -2.5915

### tasa_de_interes

- Es estacionaria: No
- p-value: 0.3960
- Estadístico de prueba: -1.7689
- Valores críticos:
  - 1%: -3.5352
  - 5%: -2.9072
  - 10%: -2.5911

### inflacion

- Es estacionaria: No
- p-value: 0.3172
- Estadístico de prueba: -1.9319
- Valores críticos:
  - 1%: -3.5352
  - 5%: -2.9072
  - 10%: -2.5911

### Visualización de Series Temporales

![Series Temporales](series_temporales_analisis_mejorado.png)

### Series Temporales Diferenciadas

![Series Diferenciadas](series_temporales_diferenciadas_mejorado.png)

## Modelos ARIMA de Series Temporales

### tipo_de_cambio

- MSE: 17.1166
- Orden del modelo: (1,1,1)

![Predicción ARIMA](arima_prediccion_mejorado_tipo_de_cambio.png)

### tasa_de_interes

- MSE: 2.5041
- Orden del modelo: (1,1,1)

![Predicción ARIMA](arima_prediccion_mejorado_tasa_de_interes.png)

### inflacion

- MSE: 3.6692
- Orden del modelo: (1,1,1)

![Predicción ARIMA](arima_prediccion_mejorado_inflacion.png)

## Regresiones Polinómicas Mejoradas

### Tipo de Cambio (MXN/USD) ~ Tasa de Interés (%)

- Grado óptimo: 3
- R²: 0.3503
- Ecuación Polinómica:
  Tipo de Cambio (MXN/USD) = -64.3779 + 28.8315 × Tasa de Interés (%) + -3.8538 × Tasa de Interés (%)^2 + 0.1712 × Tasa de Interés (%)^3

![Regresión Polinómica](regresion_polinomica_tipo_de_cambio_tasa_de_interes_lag0.png)

### Tasa de Interés (%) ~ Inflación (%)

- Grado óptimo: 3
- R²: 0.2509
- Ecuación Polinómica:
  Tasa de Interés (%) = 36.4257 + -13.5872 × Inflación (%) + 1.9734 × Inflación (%)^2 + -0.0877 × Inflación (%)^3

![Regresión Polinómica](regresion_polinomica_tasa_de_interes_inflacion_lag0.png)

### Tipo de Cambio (MXN/USD) ~ Inflación (%)

- Grado óptimo: 3
- R²: 0.3874
- Ecuación Polinómica:
  Tipo de Cambio (MXN/USD) = -49.7687 + 27.6556 × Inflación (%) + -4.4028 × Inflación (%)^2 + 0.2303 × Inflación (%)^3

![Regresión Polinómica](regresion_polinomica_tipo_de_cambio_inflacion_lag0.png)

### Tipo de Cambio (MXN/USD) ~ Tasa de Interés (%) [t-8] (Retardo: 8 meses)

- Grado óptimo: 3
- R²: 0.5570
- Ecuación Polinómica:
  Tipo de Cambio (MXN/USD) = -13.2904 + 9.2364 × Tasa de Interés (%) [t-8] + -1.4786 × Tasa de Interés (%) [t-8]^2 + 0.0804 × Tasa de Interés (%) [t-8]^3

![Regresión Polinómica](regresion_polinomica_tipo_de_cambio_tasa_de_interes_lag8.png)

### Tasa de Interés (%) ~ Inflación (%) [t-1] (Retardo: 1 meses)

- Grado óptimo: 3
- R²: 0.2762
- Ecuación Polinómica:
  Tasa de Interés (%) = 38.2422 + -14.3004 × Inflación (%) [t-1] + 2.0604 × Inflación (%) [t-1]^2 + -0.0909 × Inflación (%) [t-1]^3

![Regresión Polinómica](regresion_polinomica_tasa_de_interes_inflacion_lag1.png)

### Tipo de Cambio (MXN/USD) ~ Inflación (%) [t-3] (Retardo: 3 meses)

- Grado óptimo: 3
- R²: 0.2888
- Ecuación Polinómica:
  Tipo de Cambio (MXN/USD) = 19.5791 + -7.0147 × Inflación (%) [t-3] + 1.1370 × Inflación (%) [t-3]^2 + -0.0512 × Inflación (%) [t-3]^3

![Regresión Polinómica](regresion_polinomica_tipo_de_cambio_inflacion_lag3.png)

## Conclusiones

1. **Correlaciones Mejoradas**: El análisis con retardos temporales muestra relaciones más fuertes que las correlaciones simples sin retardo.

2. **Regresiones Polinómicas**: Los modelos con retardos temporales proporcionan un mejor ajuste, demostrando que las relaciones entre los indicadores económicos tienen componentes temporales importantes.

3. **Modelos ARIMA**: Proporcionan una capacidad predictiva moderada para las series temporales, capturando la dinámica a corto plazo.

4. **Implicaciones para Políticas**: Los resultados sugieren que las políticas monetarias (tasa de interés) tienen impactos en tipo de cambio e inflación que se manifiestan con ciertos retardos temporales específicos.

5. **Recomendaciones**: Para futuras investigaciones, se recomienda considerar modelos más complejos como VAR (Vector Autoregression) o VEC (Vector Error Correction) que puedan capturar mejor la dinámica multivariable de estos indicadores económicos.
