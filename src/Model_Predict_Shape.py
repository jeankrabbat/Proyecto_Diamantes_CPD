import io
from pathlib import Path
import cv2
import h5py
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# -----------------------------------------------------------------------------
# Configuración y Hardware
# -----------------------------------------------------------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HDF5_PATH = PROJECT_ROOT / "data" / "processed" / "diamonds_originales.h5"
MODEL_OUTPUT_PATH = PROJECT_ROOT / "src" / "model_predict_shape.keras"
ENCODER_OUTPUT_PATH = PROJECT_ROOT / "src" / "label_encoder_shape.pkl"

# Mejoras: Mayor resolución
TARGET_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25

# -----------------------------------------------------------------------------
# Redimensionamiento con Relleno (Letterboxing)
# -----------------------------------------------------------------------------
def resize_with_padding(img, target_size=(224, 224)):
    """Mantiene la relación de aspecto original agregando bordes negros."""
    h, w = img.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    nh, nw = int(h * scale), int(w * scale)
    
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    
    top = (target_size[0] - nh) // 2
    bottom = target_size[0] - nh - top
    left = (target_size[1] - nw) // 2
    right = target_size[1] - nw - left
    
    return cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

def load_images_from_h5(h5_path, indices, target_size=(224, 224)):
    images = []
    print(f"Cargando {len(indices):,} imágenes desde el HDF5...")
    
    with h5py.File(h5_path, 'r') as h5_file:
        image_bytes_dataset = h5_file['image_bytes']
        for count, idx in enumerate(indices):
            if count % 2000 == 0 and count > 0:
                print(f"Procesadas {count:,}/{len(indices):,} imágenes...")
                
            raw_bytes = image_bytes_dataset[idx]
            img = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
            
            if img is not None:
                img = resize_with_padding(img, target_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
            else:
                raise ValueError(f"Error al decodificar la imagen en el índice {idx}")
                
    return np.array(images, dtype=np.uint8)

# -----------------------------------------------------------------------------
# Carga de Metadatos y Filtrado
# -----------------------------------------------------------------------------
if not HDF5_PATH.exists():
    raise FileNotFoundError(f"No se encontró el archivo HDF5 en {HDF5_PATH}.")

with h5py.File(HDF5_PATH, 'r') as h5_file:
    metadata_bytes = h5_file['metadata_csv'][()]
    metadata_str = metadata_bytes.decode('utf-8') if isinstance(metadata_bytes, bytes) else str(metadata_bytes)

df_metadata = pd.read_csv(io.StringIO(metadata_str))

valid_mask = df_metadata['shape'].str.lower() != 'marquise'
indices_validos = np.where(valid_mask)[0]
df_filtrado = df_metadata[valid_mask].copy()

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df_filtrado['shape'])
joblib.dump(label_encoder, ENCODER_OUTPUT_PATH)

X_images = load_images_from_h5(HDF5_PATH, indices_validos, target_size=TARGET_SIZE)
X_images = X_images.astype('float32') / 255.0

X_train, X_test, y_train, y_test = train_test_split(
    X_images, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# -----------------------------------------------------------------------------
# Definición del Modelo con Transfer Learning y Data Augmentation
# -----------------------------------------------------------------------------
num_classes = len(label_encoder.classes_)

# Augmentación de datos en GPU
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.5),  # Rotación libre de 0 a 360 grados
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.15),
])

# Base preentrenada en ImageNet
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights='imagenet',
    input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3)
)
base_model.trainable = False  # Congelar la base para la primera fase

# Construcción de la arquitectura
inputs = layers.Input(shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------------------------------------------------------
# Entrenamiento con Callbacks
# -----------------------------------------------------------------------------
cb_list = [
    callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)
]

print("\nIniciando entrenamiento...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    callbacks=cb_list
)

model.save(MODEL_OUTPUT_PATH)
print(f"\n¡Entrenamiento finalizado y guardado en {MODEL_OUTPUT_PATH}!")

# -----------------------------------------------------------------------------
# Guardar Historial e Imprimir Métricas
# -----------------------------------------------------------------------------
import json

# Guardar historial completo en un archivo JSON
history_path = PROJECT_ROOT / "src" / "history_shape.json"
with open(history_path, "w") as f:
    json.dump(history.history, f)

# Extraer mejor época (basada en el menor val_loss)
best_epoch_idx = int(np.argmin(history.history['val_loss']))
train_acc = history.history['accuracy'][best_epoch_idx] * 100
val_acc = history.history['val_accuracy'][best_epoch_idx] * 100
total_epochs = len(history.history['loss'])

print("\n" + "="*50)
print("📊 DATOS DE EVALUACIÓN PARA TU TABLA")
print("="*50)
print(f"Modelo          : Shape (EfficientNetB0)")
print(f"Accuracy Train  : {train_acc:.2f}%")
print(f"Accuracy Val    : {val_acc:.2f}%")
print(f"Épocas ejecutadas: {total_epochs} (Mejor época: {best_epoch_idx + 1})")
print("="*50)