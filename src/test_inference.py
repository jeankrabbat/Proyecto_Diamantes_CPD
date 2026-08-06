from pathlib import Path
import cv2
import joblib
import numpy as np
import tensorflow as tf

# 1. Rutas relativas del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "src" / "model_predict_shape.keras"
ENCODER_PATH = PROJECT_ROOT / "src" / "label_encoder_shape.pkl"


def predecir_forma(image_path, threshold=0.40):
    """Recibe la ruta de una imagen, aplica el modelo y evalúa la regla para Marquise."""
    image_path = Path(image_path)

    if not image_path.exists():
        print(f"❌ Error: La imagen {image_path} no existe.")
        return

    # 2. Cargar artefactos
    model = tf.keras.models.load_model(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)

    # 3. Preprocesamiento (igual al de entrenamiento)
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ Error: No se pudo leer la imagen en {image_path}.")
        return

    img_resized = cv2.resize(img, (128, 128))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb.astype("float32") / 255.0
    input_tensor = np.expand_dims(
        img_normalized, axis=0
    )  # Shape de entrada: (1, 128, 128, 3)

    # 4. Inferencia
    probs = model.predict(input_tensor, verbose=0)[0]
    max_prob = np.max(probs)
    predicted_idx = np.argmax(probs)

    # 5. Aplicar Regla de Negocio
    if max_prob < threshold:
        prediccion_final = "marquise"
        regla = (
            f"Baja confianza ({max_prob:.2%}). Asignado a Marquise por regla por"
            " defecto."
        )
    else:
        prediccion_final = label_encoder.inverse_transform([predicted_idx])[0]
        regla = f"Confianza alta ({max_prob:.2%})."

    # 6. Despliegue de Resultados
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


if __name__ == "__main__":
    # Ruta a la carpeta con las imágenes de prueba
    test_folder = PROJECT_ROOT / "data" / "test_images"

    # Extensiones de imagen soportadas (incluyendo .jfif)
    extensiones = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.jfif")
    imagenes_encontradas = []

    for ext in extensiones:
        imagenes_encontradas.extend(test_folder.glob(ext))

    if not imagenes_encontradas:
        print(f"❌ No se encontraron imágenes en la ruta: {test_folder}")
    else:
        print(f"🔍 Evaluando {len(imagenes_encontradas)} imágenes encontradas en {test_folder.name}...")
        for imagen in imagenes_encontradas:
            predecir_forma(imagen, threshold=0.40)