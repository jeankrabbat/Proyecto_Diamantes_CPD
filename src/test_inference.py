from pathlib import Path
import cv2
import joblib
import numpy as np
import tensorflow as tf

# -----------------------------------------------------------------------------
# 1. Configuración y Rutas
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "src" / "model_predict_shape.keras"
ENCODER_PATH = PROJECT_ROOT / "src" / "label_encoder_shape.pkl"
TARGET_SIZE = (224, 224)


# -----------------------------------------------------------------------------
# 2. Redimensionamiento con Relleno (Letterboxing)
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
# 3. Función de Inferencia Individual
# -----------------------------------------------------------------------------
def predecir_forma(image_path, threshold=0.40):
    """Recibe la ruta de una imagen, aplica el modelo y evalúa la regla para Marquise."""
    image_path = Path(image_path)

    if not image_path.exists():
        print(f"❌ Error: La imagen {image_path} no existe.")
        return

    # Cargar artefactos
    model = tf.keras.models.load_model(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)

    # Preprocesamiento idéntico al del entrenamiento
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ Error: No se pudo leer la imagen en {image_path}.")
        return

    # Redimensionamiento letterboxing a (224, 224)
    img_padded = resize_with_padding(img, TARGET_SIZE)
    img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)

    # Rango [0, 255] float32 (preprocess_input interno de EfficientNet gestiona la escala)
    input_tensor = np.expand_dims(img_rgb.astype("float32"), axis=0)

    # Inferencia
    probs = model.predict(input_tensor, verbose=0)[0]
    max_prob = np.max(probs)
    predicted_idx = np.argmax(probs)

    # Regla de Negocio
    if max_prob < threshold:
        prediccion_final = "marquise"
        regla = (
            f"Baja confianza ({max_prob:.2%}). Asignado a Marquise por regla por defecto."
        )
    else:
        prediccion_final = label_encoder.inverse_transform([predicted_idx])[0]
        regla = f"Confianza alta ({max_prob:.2%})."

    # Despliegue de Resultados
    print("\n" + "=" * 50)
    print("💎 RESULTADO DE LA PREDICCIÓN")
    print("=" * 50)
    print(f"📷 Imagen analizada : {image_path.name}")
    print(f"🏷️  Forma estimada  : {prediccion_final.upper()}")
    print(f"📌 Detalle         : {regla}")
    print("-" * 50)
    print("📊 Probabilidades por clase:")
    for idx, clase in enumerate(label_encoder.classes_):
        print(f"  - {clase:<10}: {probs[idx]:.2%}")
    print("=" * 50 + "\n")


# -----------------------------------------------------------------------------
# 4. Ejecución Lote de Prueba Local
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    test_folder = PROJECT_ROOT / "data" / "test_images"
    extensiones = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.jfif")
    imagenes_encontradas = []

    for ext in extensiones:
        imagenes_encontradas.extend(test_folder.glob(ext))

    if not imagenes_encontradas:
        print(f"❌ No se encontraron imágenes en la ruta: {test_folder}")
    else:
        print(
            f"🔍 Evaluando {len(imagenes_encontradas)} imágenes en {test_folder.name}..."
        )
        for imagen in imagenes_encontradas:
            predecir_forma(imagen, threshold=0.40)