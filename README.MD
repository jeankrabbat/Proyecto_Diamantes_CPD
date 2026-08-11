

# Proyecto Valorador de Diamantes

**Exploración de Herramientas Paralelas y Distribuidas para el Análisis de Imágenes Gemológicas en Python**

Proyecto desarrollado para el curso de **Computación Paralela y Distribuida**, ULEAD, Costa Rica.

---

## Integrantes

- Daniel Lara Vargas
- Sebastián Blanco Perea
- Jean Carlo Rabbat Sánchez
- Esteban Ramírez Montano
- Gabriel Corrales Mora
- Jorge Chaves Zamora

**Profesor:** Johansell Villalobos Cubillo

---

## Descripción del Proyecto

El mercado de diamantes naturales presenta una marcada asimetría de información entre compradores y vendedores no especializados: dos piedras que a simple vista parecen idénticas pueden diferir drásticamente en valor según sus características gemológicas.

Este proyecto propone un sistema de valoración automática de diamantes basado en visión computacional. A partir de una fotografía, el sistema infiere los atributos gemológicos del diamante —**forma (shape)**, **color (colour)** y **pureza (clarity)**— mediante modelos de aprendizaje profundo entrenados sobre un conjunto de imágenes reales, con miras a integrarlos posteriormente con una API de precios de mercado para generar una estimación del valor en dólares.

La solución combina aprendizaje por transferencia, arquitecturas convolucionales modernas (EfficientNet, ResNet50, ConvNeXt), almacenamiento eficiente en HDF5 y pipelines de datos acelerados, en línea con los objetivos de cómputo paralelo y distribuido del curso.

---

## 📂 Estructura del Repositorio

```text
💎 Proyecto_Diamantes_CPD/
├── 📂 data/
│   └── 📄 muestra_prueba.csv            # Muestra de datos para pruebas rápidas
│
├── 📂 images/
│   ├── 📂 clarity-model-metrics/        # Métricas y gráficos del modelo de claridad
│   └── 📂 shape-model-metrics/          # Métricas y gráficos del modelo de forma
│
├── 📂 models/
│   ├── 🧠 ConvNeXt-Clarity-Model.pt     # Modelo final de claridad
│   ├── 🧠 Shape-Model.keras             # Modelo final de forma
│   ├── 🏷️ label_encoder.pkl             # Label encoder de claridad
│   └── 🏷️ label_encoder_shape.pkl       # Label encoder de forma
│
├── 📂 notebooks/
│   ├── 📓 EDA_Gems.ipynb                # Análisis exploratorio de datos
│   └── 📓 Model_Predict_Colour.ipynb    # Desarrollo del modelo de color
│
├── 📂 src/
│   ├── 🐍 Calibrate_IF_Threshold.py     # Calibración del umbral IF (claridad)
│   ├── 🐍 Check_Hardware.py             # Verificación de GPU/CUDA
│   ├── 🐍 ETL.py                        # Pipeline de extracción y transformación
│   ├── 🐍 Model_Predict_Clarity.py      # Entrenamiento del modelo de claridad
│   ├── 🐍 Model_Predict_Shape.py        # Entrenamiento del modelo de forma
│   ├── 🐍 Preparar_Dataset_HDF5.py      # Construcción del dataset HDF5
│   ├── 🐍 extract.py                    # Extracción de datos e imágenes
│   └── 🐍 transform.py                  # Limpieza y transformación de metadatos
│
├── ⚙️ .gitattributes
├── 🚫 .gitignore
├── 📖 README.md
└── 📦 requirements.txt
```

---

## Ramas del Repositorio

El desarrollo se organiza mediante ramas de trabajo por integrante/módulo, que se integran a `master` una vez validadas:

| Rama | Propósito |
|---|---|
| `master` | Rama principal, contiene la versión estable e integrada del proyecto |
| `eda-diamantes-jorge` | Análisis exploratorio de datos |
| `mejora-etl-diamantes` | Mejoras al pipeline de extracción, transformación y carga |
| `predict-clarity-esteban` | Desarrollo del modelo de clasificación de claridad |
| `predict-shape-gab` | Desarrollo del modelo de clasificación de forma |

---

## Ejecución en el Clúster Kabré (CeNAT)

Los entrenamientos de los modelos —especialmente el pipeline ETL sobre el dataset completo y el entrenamiento de los tres modelos de visión— se ejecutaron en **Kabré**, el supercomputador del [Centro Nacional de Alta Tecnología (CeNAT)](https://cnca.cenat.ac.cr/), operado por el Colaboratorio Nacional de Computación Avanzada (CNCA). Kabré cuenta con nodos GPU orientados a áreas de aprendizaje automático, simulación, *big data* y bioinformática, y utiliza **SLURM** (*Simple Linux Utility for Resource Management*) como gestor de colas de trabajos.

### Acceso

El acceso a Kabré requiere una cuenta institucional solicitada previamente ante el CNCA-CeNAT (https://kabre.cenat.ac.cr/registro). Una vez aprobada la solicitud, la conexión se realiza por SSH:

```bash
ssh usuario@kabre.cenat.ac.cr
```

### Comandos básicos de SLURM

| Comando | Función |
|---|---|
| `sinfo` | Muestra el estado de las particiones y nodos disponibles en el clúster |
| `squeue -u $USER` | Consulta el estado de los trabajos propios en la cola |
| `sbatch script.sh` | Envía un trabajo a la cola en forma de script (uso batch, no interactivo) |
| `srun` | Ejecuta una tarea de forma interactiva en los nodos de cómputo solicitados |
| `scancel <job_id>` | Cancela un trabajo en la cola usando su ID |
| `scontrol show job <job_id>` | Muestra el detalle de un trabajo específico |
| `module avail` | Lista los módulos de software (Python, CUDA, etc.) disponibles en el clúster |
| `module load <módulo>` | Carga un módulo específico en el entorno actual (por ejemplo, una versión de CUDA) |

### Ejemplo de script de envío (`sbatch`)

```bash
#!/bin/bash
#SBATCH --job-name=diamantes-shape
#SBATCH --partition=gpu            # Nombre de partición GPU (confirmar con CNCA)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module load cuda/12.8             # Ajustar según los módulos disponibles en Kabré
source .venv/bin/activate

python src/Model_Predict_Shape.py
```

Envío del trabajo y monitoreo:

```bash
mkdir -p logs
sbatch run_shape.sh
squeue -u $USER
```

> **Nota:** los nombres exactos de partición (`--partition`), los módulos disponibles (`module avail`) y las cuotas de almacenamiento/GPU son específicos de la configuración de Kabré y pueden variar; se recomienda confirmarlos con la documentación oficial del CNCA-CeNAT o el equipo de soporte antes de enviar trabajos de larga duración.

### Lección aprendida: almacenamiento

Durante el desarrollo del proyecto se identificó que guardar múltiples copias preprocesadas de las imágenes agotaba rápidamente la cuota de almacenamiento asignada en Kabré. Por ello, el pipeline de datos (ver `Preparar_Dataset_HDF5.py`) almacena las imágenes **una única vez, en su formato original**, dentro de un único archivo HDF5, y las decodifica bajo demanda durante el entrenamiento —priorizando el uso de cómputo en GPU/CPU (más disponible en el clúster) sobre el uso de almacenamiento en disco (más limitado).

---

## Requisitos

- **Python 3.11+**
- **GPU con soporte CUDA** (obligatorio para el entrenamiento de los modelos de claridad y recomendado para forma/color; los scripts de entrenamiento con PyTorch verifican la disponibilidad de CUDA al iniciar y detienen la ejecución si no la encuentran)
- Dependencias listadas en `requirements.txt`, entre las principales:

| Categoría | Librerías |
|---|---|
| Deep Learning | `tensorflow==2.21.0`, `keras==3.15.0`, `torch==2.11.0+cu128`, `torchvision==0.26.0+cu128` |
| Cómputo en GPU (CUDA) | `nvidia-cublas`, `nvidia-cudnn-cu12/cu13`, `nvidia-cufft`, `nvidia-curand`, `nvidia-cusolver`, `nvidia-nccl`, `triton`, entre otras |
| Datos y procesamiento | `pandas==3.0.3`, `numpy==2.5.0`, `polars==1.43.0`, `h5py==3.14.0`, `scikit-learn==1.9.0`, `scipy==1.18.0` |
| Imágenes | `opencv-python==4.13.0.92`, `opencv-python-headless==5.0.0.93`, `pillow==12.3.0` |
| Visualización | `matplotlib==3.11.0` |
| Dataset / utilidades | `kagglehub==1.0.2`, `kaggle==2.2.3`, `tqdm==4.68.3`, `joblib==1.5.3` |

### Instalación

```bash
git clone https://github.com/jeankrabbat/Proyecto_Diamantes_CPD.git
cd Proyecto_Diamantes_CPD

python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

> **Nota:** los paquetes `nvidia-*` y `torch`/`torchvision` con sufijo `+cu128` están compilados para CUDA 12.8. Si se ejecuta en un entorno con una versión distinta de CUDA (por ejemplo, un clúster HPC como Kabre), verificar la compatibilidad de drivers antes de instalar, o ejecutar `src/Check_Hardware.py` para validar la configuración de GPU disponible.

---

## Pipeline de Ejecución

El flujo de trabajo sigue el siguiente orden:

1. **Verificación de hardware**
```bash
   python src/Check_Hardware.py
```

2. **Extracción, transformación y construcción del dataset**
```bash
   python src/ETL.py
   python src/Preparar_Dataset_HDF5.py
```
   Genera el archivo consolidado `data/processed/diamonds_originales.h5`, que almacena las imágenes originales (sin redimensionar) junto con la metadata sincronizada y las etiquetas codificadas de forma, color y claridad.

3. **Entrenamiento y evaluación de los modelos**
```bash
   python src/Model_Predict_Shape.py
   python src/Model_Predict_Clarity.py
```
   El modelo de color se desarrolla en `notebooks/Model_Predict_Colour.ipynb`.

   Cada script genera, además del modelo serializado en `models/`, un conjunto de artefactos de evaluación en `images/<modelo>-model-metrics/`: curvas de entrenamiento, matrices de confusión, reportes de clasificación por clase y métricas agregadas (accuracy, precision, recall, F1).

4. **Calibración adicional (claridad)**
```bash
   python src/Calibrate_IF_Threshold.py
```
   Ajusta el umbral de decisión de la clase `IF` (Internally Flawless) del modelo de claridad.

---

## Modelos

| Atributo | Enfoque | Artefacto final |
|---|---|---|
| **Forma (shape)** | Transferencia de aprendizaje sobre EfficientNetB0 (TensorFlow/Keras), con entrenamiento en dos fases (backbone congelado + fine-tuning) | `models/Shape-Model.keras` |
| **Claridad (clarity)** | Comparación de tres arquitecturas preentrenadas en ImageNet (ResNet50, EfficientNet-B0, ConvNeXt Tiny) sobre PyTorch, con Focal Loss, ponderación de clases y selección del mejor modelo por F1-macro | `models/ConvNeXt-Clarity-Model.pt` |
| **Color (colour)** | CNN convolucional entrenada sobre las imágenes del dataset | `models/Colour-Model` |

---

## Fuente de Datos

El conjunto de datos se descarga mediante `kagglehub` y se procesa a través del pipeline ETL descrito arriba. Durante la transformación se excluyen las categorías de claridad `FL`, `I1`, `I2`, `I3` y los rangos de color fuera de la escala `D`–`N`, por representar una fracción insuficiente del conjunto de datos para un entrenamiento robusto. La forma `Marquise` se excluye del entrenamiento del modelo de forma por el mismo motivo.

---

## Documentación

El fundamento teórico, la justificación técnica y el plan de trabajo del proyecto se documentan en el informe del curso (formato IEEE), disponible en el repositorio.

---

## Licencia

Proyecto desarrollado con fines académicos para el curso de Computación Paralela y Distribuida, ULEAD, Costa Rica, 2026.
