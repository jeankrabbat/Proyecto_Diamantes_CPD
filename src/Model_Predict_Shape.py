import json
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
HISTORY_OUTPUT_PATH = PROJECT_ROOT / "src" / "history_shape.json"

TARGET_SIZE = (224, 224)
BATCH_SIZE = 32
SHUFFLE_BUFFER = 512  # buffer de reshuffling; NO son todas las imágenes en RAM

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


# -----------------------------------------------------------------------------
# Pipeline tf.data — decodificación y carga BAJO DEMANDA
# -----------------------------------------------------------------------------
# En lugar de cargar las ~47k imágenes completas a un np.array en RAM
# (lo cual agotaba la memoria del nodo en Kabre — ver traceback de
# cv2.error: Insufficient memory), este generador abre el HDF5 una vez
# y decodifica/redimensiona una imagen a la vez, entregándolas a Keras
# por lotes a través de tf.data. Esto mantiene en memoria solo el lote
# actual (BATCH_SIZE imágenes), no el dataset completo.
def make_dataset(h5_path, indices, labels, target_size, batch_size, shuffle):
    indices = np.asarray(indices)
    labels = np.asarray(labels)

    def generador():
        # Se abre el archivo una vez por cada pasada completa del
        # dataset (una por época), y se cierra al terminar.
        orden = np.arange(len(indices))
        if shuffle:
            np.random.shuffle(orden)

        with h5py.File(h5_path, "r") as h5_file:
            image_bytes_dataset = h5_file["image_bytes"]
            for pos in orden:
                idx = indices[pos]
                raw_bytes = image_bytes_dataset[idx]
                img = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)

                if img is None:
                    # Ya fueron filtradas en extract.py, pero por
                    # seguridad se evita romper el entrenamiento.
                    img = np.zeros((*target_size, 3), dtype=np.uint8)
                else:
                    img = resize_with_padding(img, target_size)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                yield img.astype(np.float32), labels[pos]

    dataset = tf.data.Dataset.from_generator(
        generador,
        output_signature=(
            tf.TensorSpec(shape=(*target_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int64),
        ),
    )

    if shuffle:
        dataset = dataset.shuffle(SHUFFLE_BUFFER, reshuffle_each_iteration=True)

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


# -----------------------------------------------------------------------------
# Carga de Metadatos y Filtrado
# -----------------------------------------------------------------------------
if not HDF5_PATH.exists():
    raise FileNotFoundError(f"No se encontró el archivo HDF5 en {HDF5_PATH}.")

with h5py.File(HDF5_PATH, "r") as h5_file:
    metadata_bytes = h5_file["metadata_csv"][()]
    metadata_str = metadata_bytes.decode("utf-8") if isinstance(metadata_bytes, bytes) else str(metadata_bytes)

df_metadata = pd.read_csv(io.StringIO(metadata_str))

valid_mask = df_metadata["shape"].str.lower() != "marquise"
indices_validos = np.where(valid_mask)[0]  # índices reales dentro del HDF5
df_filtrado = df_metadata[valid_mask].copy()

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df_filtrado["shape"])
joblib.dump(label_encoder, ENCODER_OUTPUT_PATH)

num_classes = len(label_encoder.classes_)
print(f"Total de imágenes válidas (sin Marquise): {len(indices_validos):,}")
print(f"Clases: {list(label_encoder.classes_)}")

# Split se hace sobre los ÍNDICES, no sobre las imágenes cargadas.
train_idx, val_idx, y_train, y_val = train_test_split(
    indices_validos, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Imágenes de entrenamiento: {len(train_idx):,}")
print(f"Imágenes de validación:    {len(val_idx):,}")

train_dataset = make_dataset(
    HDF5_PATH, train_idx, y_train, TARGET_SIZE, BATCH_SIZE, shuffle=True
)
val_dataset = make_dataset(
    HDF5_PATH, val_idx, y_val, TARGET_SIZE, BATCH_SIZE, shuffle=False
)

# -----------------------------------------------------------------------------
# Definición del Modelo con Transfer Learning y Data Augmentation
# -----------------------------------------------------------------------------
# Augmentación de datos en GPU
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.5),  # Rotación libre de 0 a 360 grados
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.15),
])

# Preprocesamiento oficial de EfficientNet. IMPORTANTE: EfficientNet
# espera píxeles en [0, 255], no en [0, 1]. Dividir manualmente entre
# 255 (como en la versión anterior) provoca una doble normalización
# que aplasta la señal de la imagen y fue la causa principal del bajo
# accuracy (~43%) obtenido antes. Al aplicarlo aquí, dentro del grafo
# del modelo, queda encapsulado y no se puede volver a duplicar.
preprocess_input = tf.keras.applications.efficientnet.preprocess_input

# Base preentrenada en ImageNet
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
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
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR_HEAD),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------------------------------------------------------
# Fase 1 — Entrenamiento del head (backbone congelado)
# -----------------------------------------------------------------------------
cb_list = [
    callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6)
]

print("\nFase 1: entrenando el head (backbone congelado)...")
history_head = model.fit(
    train_dataset,
    epochs=EPOCHS_HEAD,
    validation_data=val_dataset,
    callbacks=cb_list
)

# -----------------------------------------------------------------------------
# Fase 2 — Fine-tuning (se descongelan las capas superiores del backbone)
# -----------------------------------------------------------------------------
print("\nFase 2: fine-tuning del backbone...")
base_model.trainable = True
for layer in base_model.layers[:UNFREEZE_FROM_LAYER]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR_FINE_TUNE),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_fine = model.fit(
    train_dataset,
    epochs=EPOCHS_FINE_TUNE,
    validation_data=val_dataset,
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

best_epoch_idx = int(np.argmin(history_completo["val_loss"]))
train_acc = history_completo["accuracy"][best_epoch_idx] * 100
val_acc = history_completo["val_accuracy"][best_epoch_idx] * 100
total_epochs = len(history_completo["loss"])

print("\n" + "=" * 50)
print("📊 DATOS DE EVALUACIÓN PARA TU TABLA")
print("=" * 50)
print("Modelo          : Shape (EfficientNetB0, fine-tuned)")
print(f"Accuracy Train  : {train_acc:.2f}%")
print(f"Accuracy Val    : {val_acc:.2f}%")
print(f"Épocas ejecutadas: {total_epochs} (Mejor época: {best_epoch_idx + 1})")
print("=" * 50)