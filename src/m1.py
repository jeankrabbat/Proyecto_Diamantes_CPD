from pathlib import Path
import cv2
import joblib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# 1. Rutas del Proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "src" / "model_predict_shape.keras"
ENCODER_PATH = PROJECT_ROOT / "src" / "label_encoder_shape.pkl"
TEST_FOLDER = PROJECT_ROOT / "data" / "test_images"

# Mapeo de imágenes con su clase real
MAPEO_CLASES_REALES = {
    "p1": "round",
    "p2": "round",
    "p3": "round",
    "p4": "oval",
    "p5": "pear",
    "p6": "cushion"
}

# 2. Cargar artefactos
model = tf.keras.models.load_model(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)
class_names = list(label_encoder.classes_)
target_size = (model.input_shape[1], model.input_shape[2])

# 3. Cargar imágenes
extensiones = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.jfif")
image_paths = []
for ext in extensiones:
    image_paths.extend(TEST_FOLDER.glob(ext))

if not image_paths:
    print(f"❌ No se encontraron imágenes en {TEST_FOLDER}")
    exit()

X_test = []
y_true_labels = []

for img_path in image_paths:
    img = cv2.imread(str(img_path))
    if img is not None:
        img_resized = cv2.resize(img, target_size)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        X_test.append(img_rgb.astype("float32") / 255.0)
        
        nombre_archivo = img_path.stem.lower()
        clase_real = MAPEO_CLASES_REALES.get(nombre_archivo, nombre_archivo.split("_")[0])
        y_true_labels.append(clase_real)

X_test = np.array(X_test)

# 4. Predicción
y_pred_probs = model.predict(X_test)
y_pred_idx = np.argmax(y_pred_probs, axis=1)

# 5. Mapear etiquetas
y_true_idx = []
for label in y_true_labels:
    if label in class_names:
        y_true_idx.append(list(label_encoder.classes_).index(label))
    else:
        y_true_idx.append(-1)

valid_indices = [i for i, idx in enumerate(y_true_idx) if idx != -1]

if valid_indices:
    y_true = np.array(y_true_idx)[valid_indices]
    y_pred = np.array(y_pred_idx)[valid_indices]

    # Mapeo explícito de las 7 clases entrenadas
    all_class_indices = list(range(len(class_names)))

    print("\n📊 REPORTE DE CLASIFICACIÓN (Muestra local):")
    print(classification_report(
        y_true, 
        y_pred, 
        labels=all_class_indices, 
        target_names=class_names, 
        zero_division=0
    ))

    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true, 
        y_pred, 
        labels=all_class_indices,
        display_labels=class_names, 
        cmap='Blues', 
        xticks_rotation=45, 
        ax=ax
    )
    plt.title('Matriz de Confusión - Imágenes de Prueba')
    plt.tight_layout()
    plt.show()