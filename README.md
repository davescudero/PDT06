# 🇲🇽 Análisis de Indicadores Económicos Mexicanos

Análisis integral de la relación entre tres indicadores económicos fundamentales para la economía mexicana: tipo de cambio PESO/USD, tasa de interés e inflación.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ejemplo-streamlit-app.streamlit.app/)

## 📋 Descripción

Este proyecto presenta un análisis completo de la dinámica entre indicadores económicos clave de México, utilizando una arquitectura ETL/ELT para obtener, procesar y analizar datos de fuentes oficiales.

La aplicación Streamlit resultante permite visualizar:
- Series temporales y estadísticas descriptivas
- Análisis de correlaciones directas y con retardos
- Modelos de regresión polinómica
- Modelos predictivos ARIMA
- Un informe completo con hallazgos y recomendaciones

## 🚀 Instalación y ejecución

### Prerrequisitos

- Python 3.8+
- pip o conda para gestión de paquetes

### Pasos de instalación

1. **Clonar el repositorio**
   ```bash
   git clone https://github.com/tu_usuario/analisis-economico-mexico.git
   cd analisis-economico-mexico
   ```

2. **Crear y activar entorno virtual**
   
   Con pip:
   ```bash
   python -m venv venv
   # En Windows
   venv\Scripts\activate
   # En macOS/Linux
   source venv/bin/activate
   ```
   
   Con conda:
   ```bash
   conda create -n economia python=3.8
   conda activate economia
   ```

3. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configuración (opcional)**
   
   Si deseas usar tus propias credenciales para las APIs, copia el archivo de ejemplo:
   ```bash
   cp config.example.ini config.ini
   ```
   Y luego edita `config.ini` con tus credenciales.

### Ejecución

1. **Ejecutar la aplicación Streamlit**
   ```bash
   cd v4_final/app
   streamlit run streamlit_app_final.py
   ```

2. **Acceder a la aplicación**
   
   Abre tu navegador y ve a: http://localhost:8501

## 📊 Generación de datos

La aplicación puede funcionar de dos formas:

1. **Con datos reales**: Si dispones de tokens API de Banxico e INEGI, y configuras `config.ini`.
2. **Con datos sintéticos**: Si no configuraste credenciales, la app generará automáticamente datos sintéticos realistas.

Para generar manualmente los datos de muestra:
```bash
cd v4_final/data
python sample_data_generator.py
```

## 📁 Estructura del proyecto

```
tarea06/
├── v1_extraccion/        # ETL básico
├── v2_analisis/          # Análisis preliminar
├── v3_visualizacion/     # Visualización básica 
├── v4_final/             # Versión final optimizada
│   ├── app/
│   │   └── streamlit_app_final.py  # Aplicación principal
│   ├── data/
│   │   └── sample_data_generator.py  # Generador de datos sintéticos
│   └── img/              # Imágenes para visualizaciones
├── docs/                 # Documentación completa
├── .gitignore            # Archivos excluidos del repositorio
├── config.example.ini    # Ejemplo de configuración
└── README.md             # Este archivo
```

## 📖 Documentación

Para ver la documentación completa del proyecto:
```bash
cd docs/build
python -m http.server 8000
```
Luego abre tu navegador en: http://localhost:8000

## 👥 Autor

**David Escudero**  
Instituto Tecnológico Autónomo de México (ITAM)  
Maestría en Ciencia de Datos  
Arquitectura de Producto de Datos

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - vea el archivo [LICENSE](LICENSE) para detalles.

---

*Desarrollado como parte del curso de Arquitectura de Producto de Datos - 2025* 