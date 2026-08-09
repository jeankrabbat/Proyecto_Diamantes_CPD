from pathlib import Path
import cv2
import joblib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "src" / "model_predict_shape.keras"
ENCODER_PATH = PROJECT_ROOT / "src" / "label_encoder_shape.pkl"
TEST_FOLDER = PROJECT_ROOT / "data" / "test_images"

MAPEO_CLASES_REALES = {
    "p1": "round", "p2": "round", "p3": "round",
    "p4": "oval", "p5": "pear", "p6": "cushion"
}

def resize_with_padding(img, target_size=(224, 224)):
    h, w = img.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    top = (target_size[0] - nh) // 2
    bottom = target_size[0] - nh - top
    left = (target_size[1] - nw) // 2
    right = target_size[1] - nw - left
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

model = tf.keras.models.load_model(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)
class_names = list(label_encoder.classes_)
target_size = (224, 224)

extensiones = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.jfif")
image_paths = []
for ext in extensiones:
    image_paths.extend(TEST_FOLDER.glob(ext))

if not image_paths:
    print(f"❌ No se encontraron imágenes en {TEST_FOLDER}")
    exit()

X_test, y_true_labels = [], []
for img_path in image_paths:
    img = cv2.imread(str(img_path))
    if img is not None:
        img_padded = resize_with_padding(img, target_size)
        img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
        X_test.append(img_rgb.astype("float32"))  # Mantener rango 0-255
        
        nombre = img_path.stem.lower()
        clase_real = MAPEO_CLASES_REALES.get(nombre, nombre.split("_")[0])
        y_true_labels.append(clase_real)

X_test = np.array(X_test)
y_pred_probs = model.predict(X_test)
y_pred_idx = np.argmax(y_pred_probs, axis=1)

y_true_idx = [class_names.index(lbl) if lbl in class_names else -1 for lbl in y_true_labels]
valid_indices = [i for i, idx in enumerate(y_true_idx) if idx != -1]

if valid_indices:
    y_true = np.array(y_true_idx)[valid_indices]
    y_pred = np.array(y_pred_idx)[valid_indices]

    print("\n📊 REPORTE DE CLASIFICACIÓN (Muestra local):")
    print(classification_report(y_true, y_pred, labels=list(range(len(class_names))), target_names=class_names, zero_division=0))

    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=list(range(len(class_names))), display_labels=class_names, cmap='Blues', xticks_rotation=45, ax=ax)
    plt.title('Matriz de Confusión - Imágenes de Prueba')
    plt.tight_layout()
    plt.show()