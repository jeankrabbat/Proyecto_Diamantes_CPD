import io
import json
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
HISTORY_OUTPUT_PATH = PROJECT_ROOT / "src" / "history_shape.json"

TARGET_SIZE = (224, 224)
BATCH_SIZE = 32

# Fase 1: solo se entrena el head, con el backbone congelado.
EPOCHS_HEAD = 15
LR_HEAD = 1e-3

# Fase 2: fine-tuning, se descongelan las capas superiores del backbone
# con un learning rate mucho más bajo para no destruir los pesos de
# ImageNet.
EPOCHS_FINE_TUNE = 15
LR_FINE_TUNE = 1e-5
UNFREEZE_FROM_LAYER = 100  # capas >= este índice quedan entrenables


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

# -----------------------------------------------------------------------------
# IMPORTANTE — Normalización
# -----------------------------------------------------------------------------
# EfficientNet (familia Keras Applications) ya incluye internamente su
# propia capa de preprocesamiento/rescaling y espera píxeles en el rango
# [0, 255], NO en [0, 1]. Dividir aquí entre 255 provoca una doble
# normalización que aplasta la señal de la imagen antes de que el
# backbone la reciba, y fue la causa principal del bajo accuracy
# (~43%) obtenido en la corrida anterior. Por eso se deja el array en
# float32 SIN dividir entre 255.
X_images = X_images.astype('float32')

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

# Preprocesamiento oficial de EfficientNet (reemplaza la normalización
# manual). Se aplica dentro del grafo del modelo para que quede
# encapsulado y no se pueda olvidar/duplicar en inferencia.
preprocess_input = tf.keras.applications.efficientnet.preprocess_input

# Base preentrenada en ImageNet
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights='imagenet',
    input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3)
)
base_model.trainable = False  # Congelada durante la Fase 1

# Construcción de la arquitectura
inputs = layers.Input(shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR_HEAD),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------------------------------------------------------
# Fase 1 — Entrenamiento del head (backbone congelado)
# -----------------------------------------------------------------------------
cb_list = [
    callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)
]

print("\nFase 1: entrenando el head (backbone congelado)...")
history_head = model.fit(
    X_train, y_train,
    epochs=EPOCHS_HEAD,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    callbacks=cb_list
)

# -----------------------------------------------------------------------------
# Fase 2 — Fine-tuning (se descongelan las capas superiores del backbone)
# -----------------------------------------------------------------------------
# Congelar el backbone durante toda la corrida limita cuánto puede
# adaptarse el modelo al dominio específico de fotografías de
# diamantes (muy distinto a ImageNet). Se descongelan las capas
# superiores y se re-entrena con un learning rate mucho más bajo para
# afinar esas capas sin destruir los pesos preentrenados.
print("\nFase 2: fine-tuning del backbone...")
base_model.trainable = True
for layer in base_model.layers[:UNFREEZE_FROM_LAYER]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR_FINE_TUNE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    X_train, y_train,
    epochs=EPOCHS_FINE_TUNE,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    callbacks=cb_list
)

model.save(MODEL_OUTPUT_PATH)
print(f"\n¡Entrenamiento finalizado y guardado en {MODEL_OUTPUT_PATH}!")

# -----------------------------------------------------------------------------
# Guardar Historial e Imprimir Métricas (combinando ambas fases)
# -----------------------------------------------------------------------------
history_completo = {
    key: history_head.history[key] + history_fine.history[key]
    for key in history_head.history
}

with open(HISTORY_OUTPUT_PATH, "w") as f:
    json.dump(history_completo, f)

best_epoch_idx = int(np.argmin(history_completo['val_loss']))
train_acc = history_completo['accuracy'][best_epoch_idx] * 100
val_acc = history_completo['val_accuracy'][best_epoch_idx] * 100
total_epochs = len(history_completo['loss'])

print("\n" + "=" * 50)
print("📊 DATOS DE EVALUACIÓN PARA TU TABLA")
print("=" * 50)
print(f"Modelo          : Shape (EfficientNetB0, fine-tuned)")
print(f"Accuracy Train  : {train_acc:.2f}%")
print(f"Accuracy Val    : {val_acc:.2f}%")
print(f"Épocas ejecutadas: {total_epochs} (Mejor época: {best_epoch_idx + 1})")
print("=" * 50)