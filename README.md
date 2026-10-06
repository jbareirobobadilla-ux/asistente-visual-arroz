# Asistente visual de enfermedades del arroz

Proyecto final del diplomado ML/DL · **Jenny Bareiro** · 6 de octubre de 2026.

## Problema y objetivo

La revisión de fotografías de síntomas requiere criterio agronómico. Este proyecto evalúa si un modelo puede apoyar una primera clasificación visual de fotografías de arroz en **piricularia (`blast`), mancha parda (`brown_spot`) y aspecto normal (`normal`)**. El ahorro de tiempo es una hipótesis para un futuro piloto, no un beneficio medido.

Se construyó un prototipo reproducible, con aplicación Streamlit y evaluación sobre imágenes reservadas. La meta principal inicial fue **F1 macro ≥ 0,85**; la meta adicional fue **recall de piricularia ≥ 0,90**. Es una clasificación de apariencia actual; no predice brotes futuros ni resistencia genética.

## Archivos de la entrega

Mantener estos cinco archivos en la misma carpeta:

| Archivo | Contenido |
|---|---|
| `README.md` | Problema, método, resultados y ejecución |
| `notebook_final.ipynb` | Desarrollo completo con salidas y gráficos ejecutados |
| `model.pkl` | Extractor visual fijo, clasificador entrenado y metadatos |
| `datos_muestra.csv` | Exactamente 100 filas de prueba para inferencia y demostración |
| `app.py` | Funciones de entrenamiento y aplicación Streamlit |

La guía del módulo enumera cuatro nombres aunque anuncia cinco archivos. Se incluye `app.py` como quinto archivo para cumplir también la demostración de una aplicación funcional; esta elección es una interpretación del requisito. El conjunto completo y el entorno Python quedan fuera del ZIP.

## Datos públicos disponibles

Se utiliza la copia de Paddy Doctor publicada por **Project-AgML** en Hugging Face, revisión fijada `e7cd4e729b637a24d3b66fbf3cf2ead4761d5c33`. El conjunto completo contiene 10.407 imágenes y diez categorías. Para este proyecto se seleccionaron **4.467 imágenes** de las tres categorías indicadas.

Las fotografías originales se recopilaron en Tamil Nadu, India. Las etiquetas se asignaron mediante revisión visual con participación de un oficial agrícola. No se dispone de confirmación de laboratorio ni de identificadores de planta, parcela o sesión en la copia utilizada. No se emplean datos del lugar de trabajo de la autora.

Fuentes y reconocimiento de autoría:

- [Project-AgML: conjunto utilizado](https://huggingface.co/datasets/Project-AgML/paddy_disease_classification)
- [Paddy Doctor: autores y recolección original](https://paddydoc.github.io/dataset/)
- [Torchvision: MobileNetV3 Small y pesos ImageNet](https://docs.pytorch.org/vision/0.20/models/generated/torchvision.models.mobilenet_v3_small.html)

## Preparación y separación

Todas las imágenes seleccionadas pudieron abrirse. Se eliminaron **37 copias exactas redundantes** comparando SHA256, conservando **4.430 imágenes únicas**. Se agruparon las imágenes con pHash parecido (distancia de Hamming ≤ 4): 1.041 pares y 3.714 grupos. Cada grupo permanece en una sola partición. Esto reduce filtraciones por similitud detectada, pero no garantiza independencia biológica ni identifica todas las posibles fotos relacionadas.

La división estratificada por grupos usa semilla 42 y proporciones aproximadas de 70/15/15:

| Partición | Piricularia | Mancha parda | Normal | Total |
|---|---:|---:|---:|---:|
| Entrenamiento | 1.197 | 668 | 1.213 | 3.078 |
| Validación | 281 | 142 | 262 | 685 |
| Prueba | 250 | 143 | 274 | 667 |

Los modelos se compararon únicamente con validación. Cerrada la selección, se ajustó la configuración elegida con entrenamiento + validación (**3.763 imágenes**) y se evaluó la prueba reservada. No se cambiaron parámetros para mejorar los resultados de prueba.

## Método y módulos del curso

| Módulo | Aplicación |
|---|---|
| M5: redes neuronales / visión | MobileNetV3 Small preentrenada en ImageNet, con pesos fijos, genera 576 características por foto |
| M2: procesamiento | `ColumnTransformer` selecciona las características; `Pipeline` encadena imputación, escalado y clasificación |
| M3: aprendizaje supervisado | Comparación de regresión logística y Random Forest con etiquetas de enfermedades |

La red visual no se entrenó desde cero ni se ajustaron sus pesos al arroz. Se retiró su clasificador original y se usó como extractor. El procesamiento oficial redimensiona a 256, recorta el centro a 224 × 224 y normaliza RGB. El recorte puede omitir síntomas periféricos.

El `ColumnTransformer` usa exclusivamente `f_000` a `f_575`. Etiqueta, identificador, grupo y predicción guardada nunca entran al modelo. La imputación mediana y `StandardScaler` se ajustan dentro del pipeline con los datos de entrenamiento de cada comparación. No hubo valores faltantes en las características; la imputación conserva un procesamiento explícito y reutilizable.

## Comparación y resultados reales

Se compararon tres configuraciones de regresión logística, dos de Random Forest y una referencia que siempre responde la clase mayoritaria. Se eligió por F1 macro de validación: esta métrica da el mismo peso a cada categoría y combina precisión con recall.

| Configuración | F1 macro de validación | Recall de piricularia |
|---|---:|---:|
| Regresión logística, C = 0,1 | **0,8974** | 0,9181 |
| Regresión logística, C = 10 | 0,8886 | 0,9110 |
| Regresión logística, C = 1 | 0,8851 | 0,9004 |
| Random Forest, hoja mínima 5 | 0,8507 | 0,9110 |
| Random Forest, hoja mínima 2 | 0,8506 | 0,9359 |
| Referencia de clase mayoritaria | 0,1844 | 0,0000 |

Modelo elegido: regresión logística `C=0.1`, `class_weight='balanced'`, `max_iter=2000`, semilla 42, sobre características de MobileNetV3 Small.

| Medición final sobre 667 imágenes | Resultado |
|---|---:|
| F1 macro | **0,8890** |
| Accuracy | **89,96 %** |
| Aciertos / errores | **600 / 67** |
| Recall de piricularia | **88,40 %** (221 de 250) |
| F1 macro de la referencia final | 0,1818 |

**Se alcanzó la meta principal de F1 ≥ 0,85. No se alcanzó la meta adicional de recall ≥ 0,90.** La referencia final predice siempre piricularia porque es la clase mayoritaria al reunir entrenamiento y validación.

| Categoría | Precisión | Recall | F1 | Fotografías de prueba |
|---|---:|---:|---:|---:|
| Piricularia | 0,8947 | 0,8840 | 0,8893 | 250 |
| Mancha parda | 0,8231 | 0,8462 | 0,8345 | 143 |
| Normal | 0,9451 | 0,9416 | 0,9433 | 274 |

Matriz de confusión, filas = etiqueta del conjunto y columnas = predicción:

| Etiqueta / predicción | Piricularia | Mancha parda | Normal |
|---|---:|---:|---:|
| Piricularia | 221 | 20 | 9 |
| Mancha parda | 16 | 121 | 6 |
| Normal | 10 | 6 | 258 |

De las 250 fotos etiquetadas como piricularia, 29 se confundieron con otra categoría, incluyendo nueve con normal. El notebook muestra también errores. Los resultados evalúan concordancia con etiquetas visuales del conjunto, no confirmación de patógenos.

## Abrir la aplicación sin volver a entrenar

Usar **Python 3.12**. En macOS, desde Terminal en la carpeta de los cinco archivos:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1
python -m pip install numpy==2.3.5 pandas==2.2.3 scikit-learn==1.8.0 scipy==1.17.0 Pillow==11.3.0 joblib==1.5.3 pyarrow==19.0.1 streamlit==1.50.0 matplotlib==3.10.8 nbformat==5.10.4 ipykernel==6.30.1
python -m streamlit run app.py
```

Si el ejecutable disponible se llama `python3`, comprobar primero que `python3 --version` sea 3.12 y sustituir solo el comando inicial. Si Python no está instalado, descargar Python 3.12 para macOS desde [python.org](https://www.python.org/downloads/macos/).

En Windows crear el entorno con `py -3.12 -m venv .venv` y activarlo con `.venv\Scripts\Activate.ps1` en PowerShell. En Linux crear/activar el entorno como arriba. Para ambos, instalar las versiones CPU de PyTorch con este comando, seguido de las demás dependencias:

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
```

Streamlit indica la dirección local, normalmente `http://localhost:8501`. Elegir una foto pública del menú o cargar una fotografía propia de arroz. La aplicación muestra la categoría y puntajes comparativos. Los pesos ya están incluidos en `model.pkl`: la demostración no descarga el conjunto completo ni los pesos de la red.

La instalación inicial sí necesita internet. Los puntajes no están calibrados como certeza diagnóstica. No se validó el rechazo de otros cultivos, otras enfermedades ni imágenes ajenas al alcance; el sistema puede asignarles una de sus tres categorías.

## Ejecutar el notebook completo

Para leer resultados basta abrir el notebook entregado, que ya contiene salidas. Para repetir todo el desarrollo instalar además JupyterLab en el mismo entorno:

```bash
python -m pip install jupyterlab
python -m ipykernel install --user --name arroz-final --display-name "Python 3.12 - Arroz"
python -m jupyter lab
```

Abrir `notebook_final.ipynb`, seleccionar el kernel **Python 3.12 - Arroz** y ejecutar las celdas en orden desde la carpeta que contiene `app.py`. El primer paso descarga aproximadamente **820 MB** de datos públicos y unos 10 MB de pesos. Reservar al menos 2 GB libres para archivos y cachés, además del espacio de las dependencias. La instalación y ejecución tardan según la conexión y la computadora; reservar aproximadamente 20–60 minutos para repetirlo por primera vez.

La ejecución guarda datos y cachés en `datos_arroz/`. Puede reutilizarlos. Al repetir entrenamiento se sobrescriben el modelo y la muestra de la carpeta actual; conservar una copia del ZIP original antes de hacerlo. No incluir `datos_arroz/` ni `.venv/` en la entrega.

Verificación realizada: las nueve celdas Python se ejecutaron secuencialmente en el mismo proceso, capturando salidas y gráficos, sin errores. La creación de un kernel Jupyter con sockets estaba restringida en el entorno de preparación; por eso no se certifica aquí una ejecución mediante un servidor Jupyter. También se reprodujeron desde cero los grupos y particiones; la recarga del modelo reprodujo las 667 predicciones, y la aplicación se comprobó con pruebas de Streamlit y ejemplos de imagen.

## Muestra: 100 filas y 582 columnas

La muestra contiene 100 registros estratificados del conjunto de **prueba**, seleccionados con semilla 42. No reemplaza el conjunto de entrenamiento y no se eligieron únicamente aciertos.

| Columnas | Significado |
|---|---|
| `id`, `grupo`, `split` | Identificación, grupo de similitud y partición |
| `clase` | Etiqueta objetivo |
| `prediccion_guardada` | Predicción de referencia para comprobar la recarga |
| `f_000` … `f_575` | 576 características visuales, únicas entradas del clasificador |
| `imagen_jpeg_base64` | Fotografía original codificada como texto; permite ejemplos dentro del ZIP |

## ROI: tres escenarios hipotéticos

Supuestos del primer año: inversión de **Gs. 2.000.000**, operación de **Gs. 500.000** y valor del tiempo de **Gs. 50.000/hora**. Los minutos ahorrados son netos e incluyen revisión profesional. Estos valores ilustran escenarios; no son presupuestos ni tiempos observados.

Ahorro anual = consultas × minutos netos / 60 × valor de la hora.

ROI = (ahorro anual − costo total del primer año) / costo total × 100.

| Escenario | Consultas/año | Minutos netos/consulta | Valor anual del tiempo | Beneficio neto | ROI |
|---|---:|---:|---:|---:|---:|
| Conservador | 500 | 1 | Gs. 416.667 | Gs. −2.083.333 | −83,3 % |
| Intermedio | 1.500 | 3 | Gs. 3.750.000 | Gs. 1.250.000 | 50,0 % |
| Favorable | 3.000 | 4 | Gs. 10.000.000 | Gs. 7.500.000 | 300,0 % |

Liberar tiempo no implica necesariamente reducir gastos. No se estiman ahorros de fungicidas, aumentos de rendimiento ni pérdidas evitadas, porque no se midieron. El escenario conservador ilustra que podría no compensar el costo.

## Límites y próximos pasos

La utilidad agrícola y el beneficio económico siguen pendientes de evaluación. El resultado actual respalda una **prueba de concepto en este conjunto**, sin demostrar desempeño en Paraguay. Hay tres categorías, etiquetas visuales, posible relación entre fotos no identificada, recorte central y ausencia de evaluación fuera de distribución.

1. Reunir y etiquetar fotos locales con especialistas; reservar una prueba separada por parcela o sesión.
2. Ampliar categorías y evaluar calibración y rechazo de imágenes fuera del alcance.
3. Hacer un piloto con técnicos para medir tiempo, errores y costos antes de decidir una aplicación operativa.

Para la entrega: publicar el código en GitHub y mostrar la aplicación local o desplegada. Si corresponde la modalidad grabada, preparar un video no listado en YouTube de máximo diez minutos. Estas publicaciones y el envío al aula no están realizados por la preparación de este paquete.
