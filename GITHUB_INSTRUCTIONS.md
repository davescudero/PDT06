# Instrucciones para Subir el Proyecto a GitHub

Este documento contiene las instrucciones paso a paso para subir el proyecto de Análisis de Indicadores Económicos a GitHub y permitir que otros usuarios puedan ejecutarlo sin necesidad de credenciales.

## Prerrequisitos

1. Tener una cuenta en GitHub
2. Tener Git instalado en tu computadora
3. Tener configurado tu usuario y email en Git

## Pasos para Crear el Repositorio en GitHub

1. **Inicia sesión en GitHub** en https://github.com

2. **Crea un nuevo repositorio**:
   - Haz clic en el botón "+" en la esquina superior derecha y selecciona "New repository"
   - Nombre sugerido: `analisis-economico-mexico`
   - Descripción: `Análisis de Indicadores Económicos Mexicanos con arquitectura ETL/ELT y Streamlit`
   - Elige la visibilidad (público o privado)
   - No inicialices el repositorio con archivos
   - Haz clic en "Create repository"

## Preparación del Proyecto Local

1. **Abre una terminal** y navega al directorio del proyecto:
   ```bash
   cd /ruta/a/tarea06
   ```

2. **Inicializa Git** en el proyecto:
   ```bash
   git init
   ```

3. **Ejecuta el script de configuración** para asegurarte de que todas las carpetas existan:
   ```bash
   python setup.py
   ```

4. **Verifica el archivo .gitignore**:
   Asegúrate de que el archivo `.gitignore` incluya las entradas necesarias para excluir archivos de credenciales y datos grandes.

## Preparación de Archivos para GitHub

1. **Elimina cualquier archivo de credenciales o datos sensibles**:
   ```bash
   # Verifica que no haya credenciales reales en el repositorio
   rm -f config.ini
   ```

2. **Crea carpetas necesarias** (si no existen):
   ```bash
   mkdir -p v4_final/data
   mkdir -p v4_final/img
   ```

## Subir el Proyecto a GitHub

1. **Agrega los archivos al área de preparación**:
   ```bash
   git add .
   ```

2. **Verifica que no se estén incluyendo archivos sensibles**:
   ```bash
   git status
   ```
   Confirma que no aparecen archivos con credenciales o datos grandes.

3. **Haz el primer commit**:
   ```bash
   git commit -m "Versión inicial del Análisis de Indicadores Económicos"
   ```

4. **Conecta tu repositorio local con GitHub**:
   ```bash
   git remote add origin https://github.com/TU_USUARIO/analisis-economico-mexico.git
   ```
   Reemplaza `TU_USUARIO` con tu nombre de usuario de GitHub.

5. **Sube los cambios a GitHub**:
   ```bash
   git push -u origin main
   ```
   O si estás usando la rama "master":
   ```bash
   git push -u origin master
   ```

## Verificación y Documentación

1. **Verifica que el repositorio se haya subido correctamente** visitando la URL de tu repositorio:
   ```
   https://github.com/TU_USUARIO/analisis-economico-mexico
   ```

2. **Edita el README.md** en GitHub para actualizar la URL de Streamlit:
   - Reemplaza `ejemplo-streamlit-app.streamlit.app` con la URL real si planeas desplegar la aplicación en Streamlit Cloud.

## Configuración de Streamlit Cloud (Opcional)

Si deseas desplegar la aplicación en Streamlit Cloud:

1. Visita https://streamlit.io/cloud
2. Inicia sesión con tu cuenta de GitHub
3. Crea una nueva aplicación:
   - Selecciona tu repositorio
   - Establece la ruta de la aplicación: `v4_final/app/streamlit_app_final.py`
   - Establece el comando de ejecución: `streamlit run v4_final/app/streamlit_app_final.py`

4. Configura las variables de entorno (opcional):
   - Si necesitas credenciales, puedes configurarlas como secretos en Streamlit Cloud

## Notas Finales

- El proyecto está configurado para funcionar con datos de muestra cuando no se dispone de credenciales, por lo que cualquier persona que clone el repositorio podrá ejecutar la aplicación sin problemas.
- El script `v4_final/data/sample_data_generator.py` generará datos sintéticos cuando sea necesario.
- La documentación está disponible en la carpeta `docs/build` y puede ser vista con un servidor web simple. 