import io
import json
from pathlib import Path
from datetime import datetime
import cv2
import h5py
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import tensorflow as tf
from tensorflow.keras import layers, models


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Crecimiento dinámico de memoria GPU activado.")
    except RuntimeError as e:
        print(e)
# -----------------------------------------------------------------------------
# 1. Rutas del Proyecto
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_DIR = PROJECT_ROOT  # alias para las nuevas rutas de métricas y modelos
HDF5_PATH = PROJECT_ROOT / "data" / "processed" / "diamonds_originales.h5"

ENCODER_OUTPUT_PATH = PROJECT_ROOT / "src" / "label_encoder_shape.pkl"

MODELS_DIR = PROJECT_DIR / "models"
MODEL_OUTPUT_PATH = MODELS_DIR / "Shape-Model.keras"

IMAGES_DIR = PROJECT_DIR / "images" / "shape-model-metrics"

TARGET_SIZE = (128, 128)  # Tamaño para redimensionar las imágenes durante la carga
BATCH_SIZE = 32
EPOCHS = 15

# -----------------------------------------------------------------------------
# 2. Carga de Metadatos (Directamente desde HDF5)
# -----------------------------------------------------------------------------
print("Cargando metadatos directamente desde el archivo HDF5...")

if not HDF5_PATH.exists():
    raise FileNotFoundError(
        f"No se encontró el archivo HDF5 en {HDF5_PATH}. "
        "Asegúrate de haber corrido 'src/Preparar_Dataset_HDF5.py' primero."
    )

with h5py.File(HDF5_PATH, 'r') as h5_file:
    # El script Preparar_Dataset_HDF5 guarda la metadata en el dataset 'metadata_csv'
    metadata_bytes = h5_file['metadata_csv'][()]
    if isinstance(metadata_bytes, bytes):
        metadata_str = metadata_bytes.decode('utf-8')
    else:
        metadata_str = str(metadata_bytes)

df_metadata = pd.read_csv(io.StringIO(metadata_str))

# -----------------------------------------------------------------------------
# 3. Filtrar clase 'marquise'
# -----------------------------------------------------------------------------
print("Filtrando la clase 'marquise'...")
valid_mask = df_metadata['shape'].str.lower() != 'marquise'
indices_validos = np.where(valid_mask)[0]

df_filtrado = df_metadata[valid_mask].copy()

# Generar LabelEncoder exclusivo para las 7 clases restantes
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df_filtrado['shape'])

# Guardar el LabelEncoder para usarlo en inferencia
joblib.dump(label_encoder, ENCODER_OUTPUT_PATH)
print(f"Clases a entrenar ({len(label_encoder.classes_)}): {list(label_encoder.classes_)}")

# -----------------------------------------------------------------------------
# 4. Cargar y Decodificar Imágenes desde el HDF5
# -----------------------------------------------------------------------------
def load_images_from_h5(h5_path, indices, target_size=(128, 128)):
    images = []
    print(f"Cargando {len(indices):,} imágenes desde el HDF5...")
    
    with h5py.File(h5_path, 'r') as h5_file:
        image_bytes_dataset = h5_file['image_bytes']
        
        for count, idx in enumerate(indices):
            if count % 2000 == 0 and count > 0:
                print(f"Procesadas {count:,}/{len(indices):,} imágenes...")
                
            raw_bytes = image_bytes_dataset[idx]
            # Decodificar bytes usando OpenCV
            img = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
            
            if img is not None:
                img = cv2.resize(img, target_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
            else:
                raise ValueError(f"Error al decodificar la imagen en el índice {idx}")
                
    return np.array(images, dtype=np.uint8)

X_images = load_images_from_h5(HDF5_PATH, indices_validos, target_size=TARGET_SIZE)

# Normalizar píxeles a [0, 1]
X_images = X_images.astype('float32') / 255.0

# -----------------------------------------------------------------------------
# 5. Separación de Datos (Train / Test)
# -----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_images, 
    y_encoded, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_encoded
)

# -----------------------------------------------------------------------------
# 6. Definición y Entrenamiento de la CNN
# -----------------------------------------------------------------------------
num_classes = len(label_encoder.classes_)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nIniciando entrenamiento...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test)
)

# -----------------------------------------------------------------------------
# 7. Evaluación del Modelo y Exportación de Métricas
# -----------------------------------------------------------------------------

print("\nEvaluando el modelo en el conjunto de test...")

IMAGES_DIR.mkdir(parents=True, exist_ok=True)

class_names = list(label_encoder.classes_)

y_test_probs = model.predict(X_test, verbose=0)
y_test_pred = np.argmax(y_test_probs, axis=1)

test_accuracy = accuracy_score(y_test, y_test_pred)
macro_f1 = f1_score(y_test, y_test_pred, average='macro')
weighted_f1 = f1_score(y_test, y_test_pred, average='weighted')

report_text = classification_report(y_test, y_test_pred, target_names=class_names, zero_division=0)
report_dict = classification_report(
    y_test, y_test_pred, target_names=class_names, output_dict=True, zero_division=0
)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# 7.1 Reporte de clasificación consolidado (texto)
with open(IMAGES_DIR / "classification_report.txt", "w") as f:
    f.write("Reporte de Evaluación - Modelo de Forma (Shape)\n")
    f.write(f"Fecha: {timestamp}\n")
    f.write(f"Accuracy global (test): {test_accuracy:.4f}\n")
    f.write(f"Macro F1: {macro_f1:.4f}\n")
    f.write(f"Weighted F1: {weighted_f1:.4f}\n\n")
    f.write(report_text)

# 7.2 Reporte de clasificación por clase (CSV)
pd.DataFrame(report_dict).transpose().to_csv(IMAGES_DIR / "classification_report.csv")

# 7.3 Historial de entrenamiento por época (CSV)
pd.DataFrame(history.history).to_csv(IMAGES_DIR / "training_history.csv", index_label="epoch")

# 7.4 Matriz de confusión (CSV)
cm = confusion_matrix(y_test, y_test_pred)
pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(IMAGES_DIR / "confusion_matrix.csv")

# 7.5 Matriz de confusión (imagen)
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(class_names)))
ax.set_yticks(range(len(class_names)))
ax.set_xticklabels(class_names, rotation=45, ha="right")
ax.set_yticklabels(class_names)
ax.set_xlabel("Predicción")
ax.set_ylabel("Real")
ax.set_title("Matriz de Confusión - Modelo de Forma")
thresh = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig(IMAGES_DIR / "confusion_matrix.png", dpi=150)
plt.close(fig)

print(f"Métricas de evaluación exportadas en: {IMAGES_DIR}")
print("  - classification_report.txt / .csv")
print("  - training_history.csv")
print("  - confusion_matrix.csv / .png")
print(f"\nAccuracy (test): {test_accuracy:.4f} | Macro F1: {macro_f1:.4f} | Weighted F1: {weighted_f1:.4f}")

# -----------------------------------------------------------------------------
# 8. Serialización del Modelo
# -----------------------------------------------------------------------------
MODELS_DIR.mkdir(parents=True, exist_ok=True)

model.save(MODEL_OUTPUT_PATH)

print(f"\n¡Entrenamiento finalizado!")
print(f"Modelo guardado en: {MODEL_OUTPUT_PATH}")
print(f"Label Encoder guardado en: {ENCODER_OUTPUT_PATH}")
print(f"Métricas de evaluación en: {IMAGES_DIR}")