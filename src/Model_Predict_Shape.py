import json
import io
import time
from pathlib import Path

import cv2
import h5py
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Backend sin pantalla para entornos HPC/Slurm
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support

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
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR = PROJECT_ROOT / "images" / "shape-model-metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_OUTPUT_PATH = MODELS_DIR / "model_predict_shape.keras"
ENCODER_OUTPUT_PATH = MODELS_DIR / "label_encoder_shape.pkl"
HISTORY_OUTPUT_PATH = SRC_DIR / "history_shape.json"

TARGET_SIZE = (224, 224)
BATCH_SIZE = 32
SHUFFLE_BUFFER = 512

EPOCHS_HEAD = 15
LR_HEAD = 1e-3

EPOCHS_FINE_TUNE = 15
LR_FINE_TUNE = 1e-5
UNFREEZE_FROM_LAYER = 100


# -----------------------------------------------------------------------------
# Redimensionamiento con Relleno (Letterboxing)
# -----------------------------------------------------------------------------
def resize_with_padding(img, target_size=(224, 224)):
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
# Pipeline tf.data
# -----------------------------------------------------------------------------
def make_dataset(h5_path, indices, labels, target_size, batch_size, shuffle):
    indices = np.asarray(indices)
    labels = np.asarray(labels)

    def generador():
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
# Carga de Metadatos y Splits
# -----------------------------------------------------------------------------
if not HDF5_PATH.exists():
    raise FileNotFoundError(f"No se encontró el archivo HDF5 en {HDF5_PATH}.")

with h5py.File(HDF5_PATH, "r") as h5_file:
    metadata_bytes = h5_file["metadata_csv"][()]
    metadata_str = metadata_bytes.decode("utf-8") if isinstance(metadata_bytes, bytes) else str(metadata_bytes)

df_metadata = pd.read_csv(io.StringIO(metadata_str))

valid_mask = df_metadata["shape"].str.lower() != "marquise"
indices_validos = np.where(valid_mask)[0]
df_filtrado = df_metadata[valid_mask].copy()

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df_filtrado["shape"])
joblib.dump(label_encoder, ENCODER_OUTPUT_PATH)

class_names = list(label_encoder.classes_)
num_classes = len(class_names)

train_idx, val_idx, y_train, y_val = train_test_split(
    indices_validos, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

train_dataset = make_dataset(HDF5_PATH, train_idx, y_train, TARGET_SIZE, BATCH_SIZE, shuffle=True)
val_dataset = make_dataset(HDF5_PATH, val_idx, y_val, TARGET_SIZE, BATCH_SIZE, shuffle=False)

# -----------------------------------------------------------------------------
# Definición del Modelo
# -----------------------------------------------------------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.5),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.15),
])

preprocess_input = tf.keras.applications.efficientnet.preprocess_input

base_model = tf.keras.applications.EfficientNetB0(
    include_top=False, weights="imagenet", input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3)
)
base_model.trainable = False

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

cb_list = [
    callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6)
]

# -----------------------------------------------------------------------------
# Entrenamientos (Fase 1 y Fase 2)
# -----------------------------------------------------------------------------
t0 = time.time()

print("\nFase 1: entrenando el head (backbone congelado)...")
history_head = model.fit(train_dataset, epochs=EPOCHS_HEAD, validation_data=val_dataset, callbacks=cb_list)

print("\nFase 2: fine-tuning del backbone...")
base_model.trainable = True
for layer in base_model.layers[:UNFREEZE_FROM_LAYER]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR_FINE_TUNE),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_fine = model.fit(train_dataset, epochs=EPOCHS_FINE_TUNE, validation_data=val_dataset, callbacks=cb_list)
total_train_time = time.time() - t0

model.save(MODEL_OUTPUT_PATH)

# Combinar historial de entrenamiento
history_completo = {
    key: history_head.history[key] + history_fine.history[key]
    for key in history_head.history
}
with open(HISTORY_OUTPUT_PATH, "w") as f:
    json.dump(history_completo, f)

# -----------------------------------------------------------------------------
# Evaluación Extendida y Generación de Métricas
# -----------------------------------------------------------------------------
print("\n>>> Evaluando modelo final sobre el conjunto de validación...")
y_pred_probs = model.predict(val_dataset)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = y_val

# Métricas Sklearn
precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
val_acc = np.mean(y_true == y_pred)
best_epoch_idx = int(np.argmin(history_completo["val_loss"]))

# Accuracy/loss de entrenamiento en la mejor época (para la tabla del
# paper, que reporta train y val lado a lado en un mismo punto de corte).
train_acc_best = history_completo["accuracy"][best_epoch_idx]
train_loss_best = history_completo["loss"][best_epoch_idx]
val_loss_best = history_completo["val_loss"][best_epoch_idx]

# 1. Matriz de Confusión (.png)
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues", xticks_rotation=45, ax=ax)
plt.title("Matriz de Confusión - Shape Model")
plt.tight_layout()
plt.savefig(METRICS_DIR / "confusion_matrix_shape.png", dpi=150)
plt.close()

# 2. Curvas de Entrenamiento (.png)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history_completo["loss"], label="Train Loss")
plt.plot(history_completo["val_loss"], linestyle="--", label="Val Loss")
plt.axvline(x=len(history_head.history["loss"])-0.5, color="gray", linestyle=":", label="Fine-Tuning Start")
plt.title("Curvas de Pérdida (Loss)")
plt.xlabel("Época")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_completo["accuracy"], label="Train Accuracy")
plt.plot(history_completo["val_accuracy"], linestyle="--", label="Val Accuracy")
plt.axvline(x=len(history_head.history["accuracy"])-0.5, color="gray", linestyle=":", label="Fine-Tuning Start")
plt.title("Curvas de Precisión (Accuracy)")
plt.xlabel("Época")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig(METRICS_DIR / "learning_curves_shape.png", dpi=150)
plt.close()

# 3. Guardar CSV por Época
df_epochs = pd.DataFrame({
    "epoch": range(1, len(history_completo["loss"]) + 1),
    "train_loss": history_completo["loss"],
    "train_acc": history_completo["accuracy"],
    "val_loss": history_completo["val_loss"],
    "val_acc": history_completo["val_accuracy"],
})
df_epochs.to_csv(METRICS_DIR / "epoch_history_shape.csv", index=False)

# 4. Reporte Consolidado en Texto (.txt)
rep_text = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

report_lines = [
    "=" * 70,
    "REPORTE DE ENTRENAMIENTO - CLASIFICADOR DE FORMA (SHAPE)",
    "=" * 70,
    f"Modelo Architecture : EfficientNetB0 (Fine-Tuned)",
    f"Resolución de Imagen : {TARGET_SIZE[0]}x{TARGET_SIZE[1]}",
    f"Tiempo de Ejecución : {total_train_time / 60:.2f} minutos",
    f"Épocas Totales      : {len(history_completo['loss'])} (Mejor época: {best_epoch_idx + 1})",
    "\nMÉTRICAS EN LA MEJOR ÉPOCA (min val_loss)",
    "-" * 70,
    f"Train Accuracy      : {train_acc_best:.4f}",
    f"Train Loss          : {train_loss_best:.4f}",
    f"Val Accuracy        : {val_acc:.4f}",
    f"Val Loss            : {val_loss_best:.4f}",
    f"Precision (Macro)   : {precision_m:.4f}",
    f"Recall (Macro)      : {recall_m:.4f}",
    f"F1-Score (Macro)    : {f1_m:.4f}",
    "\nREPORTE DETALLADO POR CLASE",
    "-" * 70,
    rep_text,
    "=" * 70
]

report_full = "\n".join(report_lines)
with open(METRICS_DIR / "reporte_entrenamiento_shape.txt", "w") as f:
    f.write(report_full)

# 5. Resumen en JSON — pensado para llenar la tabla del paper sin
# transcribir números a mano desde el .txt (evita errores de copia).
metrics_summary = {
    "modelo": "Shape (EfficientNetB0, fine-tuned)",
    "train_accuracy": round(float(train_acc_best) * 100, 2),
    "val_accuracy": round(float(val_acc) * 100, 2),
    "train_loss": round(float(train_loss_best), 4),
    "val_loss": round(float(val_loss_best), 4),
    "precision_macro": round(float(precision_m), 4),
    "recall_macro": round(float(recall_m), 4),
    "f1_macro": round(float(f1_m), 4),
    "epocas_totales": len(history_completo["loss"]),
    "mejor_epoca": best_epoch_idx + 1,
    "tiempo_entrenamiento_min": round(total_train_time / 60, 2),
}
with open(METRICS_DIR / "metrics_summary_shape.json", "w") as f:
    json.dump(metrics_summary, f, indent=2, ensure_ascii=False)

print("\n" + report_full)
print(f"\n¡Artefactos visuales y reporte de texto guardados en {METRICS_DIR}!")
print(f"Resumen JSON para la tabla del paper: {METRICS_DIR / 'metrics_summary_shape.json'}")