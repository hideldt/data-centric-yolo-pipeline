# 🎯 Data-Centric YOLO Vision Pipeline

> **End-to-End Computer Vision:** De datos crudos en Open Images V6 a inferencia de producción con Ultralytics YOLO y ONNX.

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python)](https://www.python.org/)
[![Ultralytics YOLO](https://img.shields.io/badge/YOLO-Ultralytics-00ffff?logo=yolo)](https://github.com/ultralytics/ultralytics)
[![FiftyOne](https://img.shields.io/badge/FiftyOne-Voxel51-ff6f00)](https://voxel51.com/fiftyone/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX-Runtime-005ced?logo=onnx)](https://onnxruntime.ai/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Descripción del Proyecto

Este repositorio contiene una **Masterclass y Pipeline Industrial completo de Computer Vision** basado en el paradigma **Data-Centric AI**. 

A diferencia de los enfoques tradicionales de fuerza bruta que procesan millones de imágenes sin filtrar, este proyecto implementa:
1. **Adquisición Inteligente (Subsampling):** Muestreo representativo de 100,000 imágenes desde Open Images V6 con `fiftyone`, reduciendo el costo de almacenamiento en un 98.8%.
2. **Curación y Reglas de Negocio:** Unificación semántica de clases equivalentes (ej. `Bus` $\rightarrow$ `Car`, `Human body` $\rightarrow$ `Person`), purga de anotaciones ruidosas y eliminación de fondos irrelevantes.
3. **Estratificación por Minoría (Minority-Priority Split):** Algoritmo algorítmico de partición multi-etiqueta 90/10 para evitar que clases raras queden fuera del conjunto de validación.
4. **Fine-Tuning con Transfer Learning:** Entrenamiento sobre la arquitectura SOTA de YOLO (Ultralytics) con regularización y *online data augmentation* (Mosaic, MixUp).
5. **Evaluación Rigurosa y Métricas COCO:** Validación explícita calculando $mAP50$, $mAP50-95$, matrices de confusión y reportes tabulares por clase.
6. **Exportación a ONNX e Inferencia:** Optimización del grafo computacional (*operator fusion*, *constant folding*) para despliegue en producción sin dependencias pesadas de PyTorch.

---

## 📂 Estructura del Repositorio

```text
├── GDataset_Ejecucion.ipynb   # Notebook principal del pipeline (12 Fases)
├── GData/                     # Directorio de datos y datasets generados
│   ├── dataset.yaml           # Catálogo original de clases de Open Images
│   ├── images/                # Imágenes descargadas por FiftyOne
│   ├── labels/                # Etiquetas crudas descargadas
│   ├── clean_labels/          # Etiquetas curadas post-reglas de negocio
│   └── YOLO_Dataset/          # Dataset final particionado (train/val) en formato YOLO
│       └── data.yaml          # Configuración final para entrenamiento YOLO
├── weights/                   # Pesos base pre-entrenados
├── runs/                      # Resultados, gráficas y checkpoints de entrenamiento
│   └── detect/train/weights/  # best.pt y last.pt
├── README.md                  # Documentación del proyecto
└── requirements.txt           # Dependencias del entorno
```

---

## ⚙️ Rutas Críticas que Debe Modificar el Usuario

Para ejecutar el notebook [GDataset_Ejecucion.ipynb](GDataset_Ejecucion.ipynb) en tu propia máquina, servidor o Google Colab, **revisa y ajusta las siguientes rutas**:

### 1. Ruta de Almacenamiento de Datos (Celda 1 - Fase 1)
En la **Celda 1**, se define la ruta absoluta donde FiftyOne descargará las imágenes y etiquetas originales:

```python
# 🔴 MODIFICAR: Cambia esta ruta por el directorio absoluto de tu máquina o entorno
data_dir = "/home/delga/Proyectos/cnnT/GData"

# 🟢 RECOMENDACIÓN PARA ENTORNOS PORTABLES:
# Puedes cambiarla por una ruta relativa al directorio de trabajo:
# import os
# data_dir = os.path.abspath("GData")
```

> **En Google Colab / Kaggle:** Si ejecutas en la nube con Google Drive montado:
> ```python
> data_dir = "/content/drive/MyDrive/Proyectos/cnnT/GData"
> ```

---

### 2. Directorio Relativo de Curación (Celdas 3, 5, 7 y 9)
En las celdas de las Fases 2 a 5 se utiliza la variable relativa:

```python
data_path = "GData"
```
* **Si ejecutas el notebook desde la carpeta raíz del proyecto:** No necesitas cambiar nada (`"GData"` se resolverá automáticamente).
* **Si tu dataset está en un disco externo o ruta separada:** Actualiza `data_path` a tu ruta personalizada (ej. `"/mnt/datasets/GData"`).

---

### 3. Pesos Pre-entrenados Base de YOLO (Celdas 11 y 15 - Fases 6 y 9)
En las celdas de entrenamiento se carga el modelo base de YOLO:

```python
# Verifica que el archivo de pesos exista en la raíz o en tu carpeta de pesos
model = YOLO('yolo26s.pt')

# Si deseas utilizar otros pesos oficiales de Ultralytics:
# model = YOLO('yolov8s.pt')   # o 'yolov8n.pt', 'yolo11s.pt'
```

---

### 4. Ruta del Archivo de Configuración del Dataset para Entrenamiento (Celdas 11, 15 y 19)
Al iniciar el entrenamiento (`model.train()`) y la validación (`model.val()`), se referencia el archivo de configuración generado por la Fase 5:

```python
# Por defecto se ubica en:
data = 'GData/YOLO_Dataset/data.yaml'
```
* Si moviste `YOLO_Dataset`, asegúrate de actualizar esta ruta relativa o absoluta para que apunte al archivo `data.yaml` correcto.

---

### 5. Directorio de Detección de Última Corrida (Celdas 17 y 19 - Fases 7, 8 y 10)
La función de ayuda `get_latest_run()` busca automáticamente la carpeta más reciente dentro de:
```python
search_paths = ['runs/detect/train*', 'runs/detect/runs/train/*', 'runs/train/*']
```
* Si configuraste un nombre personalizado en `model.train(name='mi_modelo_v1', project='mis_corridas')`, verifica que `latest_run_dir` apunte a tu carpeta de resultados para cargar `results.csv` y `weights/best.pt`.

---

## 🚀 Instalación y Puesta en Marcha

### 1. Clonar el repositorio
```bash
git clone https://github.com/tu-usuario/data-centric-yolo-pipeline.git
cd data-centric-yolo-pipeline
```

### 2. Crear y activar entorno virtual
```bash
python3 -m venv venv_yolo
source venv_yolo/bin/activate
```

### 3. Instalar dependencias
```bash
pip install ultralytics fiftyone onnx onnxruntime onnxsim pandas matplotlib pyyaml
```

> **Nota sobre MongoDB en FiftyOne (Linux moderno):**  
> Si experimentas `FiftyOneConfigError` al iniciar FiftyOne, consulta la guía paso a paso incluida en la **Celda 24 (Troubleshooting)** del notebook para instalar el binario embebido de `mongod`.

---

## 🎛️ Banderas Configurables en el Notebook

* **Sanity Check Visual (Fase 4 - Celda 7):**
  ```python
  ENABLE_SANITY_CHECK_VISUALIZATION = False  # Cambiar a True para inspeccionar bounding boxes
  NUM_SAMPLES_TO_VISUALIZE = 4               # Muestras a desplegar en cuadrícula
  ```
* **Exportación ONNX (Fase 12 - Celda 23):**
  ```python
  best_model.export(format="onnx", simplify=True)  # Exportación con simplificación de grafo
  ```

---

## 📊 Pipeline de Fases (Roadmap)

1. **📦 Fase 1:** Data Acquisition focalizada con la API de `fiftyone` (Open Images V6).
2. **🧹 Fase 2:** Curación de datos, unificación taxonómica y poda de ruido.
3. **📊 Fase 3:** Exploratory Data Analysis (EDA) y diagnóstico de desbalance de clases.
4. **🕵️ Fase 4:** Sanity Check visual y verificación de alineación espacial pixel/coordenadas.
5. **🔀 Fase 5:** Partición estratificada con algoritmo *Minority Priority Split* y remapeo continuo.
6. **🚀 Fase 6:** Fine-Tuning con Transfer Learning utilizando Ultralytics YOLO.
7. **📈 Fase 7:** Análisis de curvas de pérdida (`box_loss`, `cls_loss`, `dfl_loss`) y troubleshooting.
8. **🧠 Fase 8:** Anatomía de checkpoints (`best.pt` vs `last.pt`) y carpeta `runs/`.
9. **🌪️ Fase 9:** Data Augmentation en línea (Mosaic 1.0, MixUp, Random Erasing).
10. **🧪 Fase 10:** Validación explícita y reporte de métricas COCO por clase.
11. **🔮 Fase 11:** Inferencia en muestras no vistas.
12. **📦 Fase 12:** Exportación a formato ONNX para despliegue productivo de alta velocidad.
13. **🛠️ Troubleshooting:** Resolución de dependencias locales de MongoDB para FiftyOne.

---

## 📜 Licencia

Distribuido bajo la licencia MIT. Consulta `LICENSE` para más información.
