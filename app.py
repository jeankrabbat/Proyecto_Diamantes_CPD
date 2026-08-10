import base64
import json
import time
from io import BytesIO
from pathlib import Path

import cv2
import h5py
import jax.numpy as jnp
import joblib
import numpy as np
import orbax.checkpoint as ocp
import pandas as pd
import psutil
import requests
import streamlit as st
import tensorflow as tf
import torch
import torchvision.transforms as T
from flax import nnx
from PIL import Image
from streamlit_option_menu import option_menu
from torchvision.models import convnext_tiny


APP_TITLE = "Predict Gem Price with AI"
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"

COLORS = {
    "bg": "#f7f8fb",
    "panel": "#ffffff",
    "panel_soft": "#f4f7fb",
    "text": "#151922",
    "muted": "#667085",
    "line": "#e6eaf0",
    "accent": "#2563eb",
    "accent_hover": "#1d4ed8",
    "accent_soft": "#eaf1ff",
    "price": "#0f172a",
    "price_accent": "#38bdf8",
}

TEXT = {
    "en": {
        "language": "Language",
        "upload_title": "Diamond image",
        "upload_hint": "Upload an image to preview it here.",
        "upload_button": "Select image",
        "carat": "Weight in grams",
        "carat_equivalent": "Equivalent in carats",
        "analyze": "Analyze",
        "progress": "Prediction progress",
        "colour": "Colour",
        "shape": "Shape",
        "clarity": "Clarity",
        "price": "Estimated price",
        "waiting": "Waiting for analysis",
        "required_image": "Select a diamond image before analyzing.",
        "invalid_carat": "Weight must be a positive decimal number in grams.",
        "running": "Running {step} model...",
        "done": "Analysis completed",
        "subtitle": "Find out the value of your diamond with just a photo!",
        "resources_title": "Resource usage",
        "time_label": "Time",
        "ram_label": "RAM",
        "cpu_label": "CPU",
        "cores_unit": "cores",
    },
    "es": {
        "language": "Idioma",
        "upload_title": "Imagen del diamante",
        "upload_hint": "Carga una imagen para verla aqui.",
        "upload_button": "Seleccionar imagen",
        "carat": "Peso en gramos",
        "carat_equivalent": "Equivalente en quilates",
        "analyze": "Analizar",
        "progress": "Progreso de prediccion",
        "colour": "Colour",
        "shape": "Shape",
        "clarity": "Clarity",
        "price": "Precio estimado",
        "waiting": "Esperando analisis",
        "required_image": "Selecciona una imagen del diamante antes de analizar.",
        "invalid_carat": "El peso debe ser un numero decimal positivo en gramos.",
        "running": "Ejecutando modelo de {step}...",
        "done": "Analisis completado",
        "subtitle": "Conoce el precio de tu diamante con solo una foto!",
        "resources_title": "Uso de recursos",
        "time_label": "Tiempo",
        "ram_label": "RAM",
        "cpu_label": "CPU",
        "cores_unit": "nucleos",
    },
}


# ---------------------------------------------------------------------------
# Modelo de Color (JAX + Flax NNX, checkpoint Orbax)
#
# Arquitectura idéntica a la de Model_Predict_Colour2.py: un checkpoint de
# Orbax solo guarda el estado/pesos, no la arquitectura, así que hay que
# reconstruir exactamente el mismo módulo para poder restaurarlo con
# nnx.merge().
# ---------------------------------------------------------------------------

TAMANO_IMAGEN_COLOR = 96
NUM_CLASES_COLOR = 11  # colores D-N, ver Model_Predict_Colour2.py


class BloqueConvolucional(nnx.Module):
    def __init__(self, canales_entrada, canales_salida, *, rngs: nnx.Rngs):
        self.conv = nnx.Conv(
            canales_entrada,
            canales_salida,
            kernel_size=(3, 3),
            padding="SAME",
            rngs=rngs,
        )
        self.norma = nnx.BatchNorm(canales_salida, rngs=rngs)

    def __call__(self, x, *, entrenando: bool):
        x = self.conv(x)
        x = self.norma(x, use_running_average=not entrenando)
        x = nnx.relu(x)
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        return x


class DiamondColorCNN(nnx.Module):
    def __init__(self, num_clases, *, rngs: nnx.Rngs, canales=(32, 64, 128, 256)):
        self.bloque1 = BloqueConvolucional(3, canales[0], rngs=rngs)
        self.bloque2 = BloqueConvolucional(canales[0], canales[1], rngs=rngs)
        self.bloque3 = BloqueConvolucional(canales[1], canales[2], rngs=rngs)
        self.bloque4 = BloqueConvolucional(canales[2], canales[3], rngs=rngs)
        self.dropout = nnx.Dropout(rate=0.3, rngs=rngs)
        self.salida = nnx.Linear(canales[3], num_clases, rngs=rngs)

    def __call__(self, x, *, entrenando: bool = False):
        x = self.bloque1(x, entrenando=entrenando)
        x = self.bloque2(x, entrenando=entrenando)
        x = self.bloque3(x, entrenando=entrenando)
        x = self.bloque4(x, entrenando=entrenando)
        x = jnp.mean(x, axis=(1, 2))
        x = self.dropout(x, deterministic=not entrenando)
        return self.salida(x)


@st.cache_resource
def cargar_modelo_color():
    """Reconstruye la arquitectura y restaura el checkpoint de Orbax (se
    ejecuta una sola vez gracias a st.cache_resource)."""
    ruta_modelo = MODELS_DIR / "Colour-Model"

    abstract_model = nnx.eval_shape(
        lambda: DiamondColorCNN(NUM_CLASES_COLOR, rngs=nnx.Rngs(0))
    )
    graphdef, abstract_state = nnx.split(abstract_model)

    checkpointer = ocp.StandardCheckpointer()
    estado_restaurado = checkpointer.restore(ruta_modelo, target=abstract_state)
    checkpointer.close()

    modelo = nnx.merge(graphdef, estado_restaurado)

    # El modelo de color no guarda su propio label encoder: el mapeo de
    # clases vive en el atributo JSON "label_mappings" del HDF5 (ver
    # DiamondColorDataset en Model_Predict_Colour2.py). Solo se leen los
    # atributos, no las imagenes, asi que abrir el H5 aqui es liviano.
    ruta_h5 = PROJECT_ROOT / "data" / "processed" / "diamonds_originales.h5"
    with h5py.File(ruta_h5, "r") as archivo_h5:
        mapeos = json.loads(archivo_h5.attrs["label_mappings"])
    colour_mapping = mapeos["colour"]
    clases = [
        clase
        for clase, _ in sorted(colour_mapping.items(), key=lambda kv: kv[1])
    ]

    return modelo, clases


def preprocesar_imagen_color(image):
    """Replica exactamente _decodificar_imagen() de Model_Predict_Colour2.py
    (resize 96x96 con INTER_AREA + normalizacion /255.0), partiendo de una
    imagen PIL ya en RGB."""
    arreglo = np.array(image)
    arreglo = cv2.resize(
        arreglo,
        (TAMANO_IMAGEN_COLOR, TAMANO_IMAGEN_COLOR),
        interpolation=cv2.INTER_AREA,
    )
    arreglo = arreglo.astype(np.float32) / 255.0
    return arreglo[None, ...]


# ---------------------------------------------------------------------------
# Modelo de Clarity (PyTorch, ConvNeXt Tiny)
#
# El .pt guarda un dict (no el modelo completo ni un state_dict "pelado"):
# ver select_and_save_best_model() en Model_Predict_Clarity_v2.py.
# Trae model_state_dict, class_names e img_size, asi que es autocontenido:
# no depende de label_encoder.pkl para inferir.
# ---------------------------------------------------------------------------

CLARITY_MODEL_PATH = MODELS_DIR / "ConvNeXt-Clarity-Model.pt"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_convnext_tiny_clarity(num_classes):
    # weights=None: los pesos reales vienen del checkpoint, no hace falta
    # (ni conviene) descargar los pesos preentrenados de ImageNet aqui.
    modelo = convnext_tiny(weights=None)
    in_features = modelo.classifier[2].in_features
    modelo.classifier[2] = torch.nn.Linear(in_features, num_classes)
    return modelo


@st.cache_resource
def cargar_modelo_clarity():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CLARITY_MODEL_PATH, map_location=device)

    class_names = list(checkpoint["class_names"])
    img_size = checkpoint["img_size"]

    modelo = build_convnext_tiny_clarity(len(class_names))
    modelo.load_state_dict(checkpoint["model_state_dict"])
    modelo.to(device)
    modelo.eval()

    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return modelo, transform, class_names, device


# ---------------------------------------------------------------------------
# Modelo de Shape (TensorFlow/Keras)
#
# Formato .keras nativo: arquitectura + pesos autocontenidos, se carga
# directo con load_model(). El preprocesamiento (resize 128x128 + /255.0)
# se hacia fuera del modelo, en Model_Predict_Shape.py, asi que hay que
# replicarlo aqui. cv2.resize sin interpolation explicita = INTER_LINEAR
# (el default de OpenCV), igual que en el script de entrenamiento.
# ---------------------------------------------------------------------------

SHAPE_TARGET_SIZE = (128, 128)


@st.cache_resource
def cargar_modelo_shape():
    modelo = tf.keras.models.load_model(str(MODELS_DIR / "Shape-Model.keras"))
    encoder = joblib.load(MODELS_DIR / "label_encoder_shape.pkl")
    clases = list(encoder.classes_)
    return modelo, clases


def preprocesar_imagen_shape(image):
    arreglo = np.array(image)
    arreglo = cv2.resize(arreglo, SHAPE_TARGET_SIZE)
    arreglo = arreglo.astype(np.float32) / 255.0
    return arreglo[None, ...]


def configure_page():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.markdown(
        f"""
        <link
            rel="stylesheet"
            href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css"
        >
        <style>
            :root {{
                --app-bg: {COLORS["bg"]};
                --panel: {COLORS["panel"]};
                --panel-soft: {COLORS["panel_soft"]};
                --text: {COLORS["text"]};
                --muted: {COLORS["muted"]};
                --line: {COLORS["line"]};
                --accent: {COLORS["accent"]};
                --accent-hover: {COLORS["accent_hover"]};
                --accent-soft: {COLORS["accent_soft"]};
                --price: {COLORS["price"]};
                --price-accent: {COLORS["price_accent"]};
            }}

            html, body, [class*="css"], .stApp, .stMarkdown, .stTextInput, button, input, textarea {{
                font-family: "Segoe UI", Arial, sans-serif !important;
            }}

            * {{
                letter-spacing: 0 !important;
            }}

            .stApp {{
                background:
                    radial-gradient(circle at top left, rgba(37, 99, 235, 0.08), transparent 34rem),
                    var(--app-bg);
                color: var(--text);
            }}

            header[data-testid="stHeader"] {{
                background: #ffffff !important;
                border-bottom: 1px solid var(--line);
                box-shadow: none !important;
            }}

            div[data-testid="stToolbar"],
            div[data-testid="stDecoration"],
            div[data-testid="stStatusWidget"],
            #MainMenu {{
                background: #ffffff !important;
                color: var(--text) !important;
            }}

            div[data-testid="stDecoration"] {{
                height: 0 !important;
            }}

            .block-container {{
                max-width: 1550px;
                padding-top: 2.2rem;
                padding-bottom: 2.8rem;
            }}

            h1 {{
                color: var(--text);
                font-size: 2.25rem !important;
                font-weight: 750 !important;
                letter-spacing: 0 !important;
                margin-bottom: 0.25rem !important;
            }}

            .app-subtitle {{
                color: var(--muted);
                font-size: 0.98rem;
                margin-bottom: 1.4rem;
            }}

            .topbar {{
                display: flex;
                justify-content: flex-end;
                align-items: center;
                margin-bottom: 0.8rem;
            }}

            .language-label {{
                display: flex;
                align-items: center;
                justify-content: flex-end;
                gap: 0.4rem;
                color: #475467;
                font-size: 0.74rem;
                font-weight: 750;
                margin-bottom: 0.35rem;
            }}

            .language-label i {{
                color: var(--accent);
                font-size: 0.86rem;
            }}

            [data-testid="stVerticalBlockBorderWrapper"] {{
                border: 1px solid var(--line);
                border-radius: 16px;
                box-shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
                background: var(--panel);
                overflow: hidden;
            }}

            div[data-testid="stVerticalBlockBorderWrapper"] > div {{
                padding: 1.35rem;
            }}

            .section-title {{
                display: flex;
                align-items: center;
                gap: 0.55rem;
                color: var(--text);
                font-size: 1.08rem;
                font-weight: 700;
                margin: 0 0 1rem;
            }}

            .section-title i,
            .small-label i,
            .progress-header i,
            .result-card i,
            .price-card i,
            .upload-empty i {{
                color: var(--accent);
                font-size: 1rem;
            }}

            .image-shell {{
                border-radius: 16px;
                background: var(--panel-soft);
                border: 1px solid var(--line);
                min-height: 430px;
                padding: 1rem;
                box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.8);
                overflow: hidden;
                display: flex;
                align-items: center;
                justify-content: center;
            }}

            .image-shell img {{
                display: block;
                width: 100%;
                max-height: 430px;
                border-radius: 14px;
                box-shadow: 0 18px 38px rgba(15, 23, 42, 0.14);
                object-fit: contain;
            }}

            .upload-empty {{
                min-height: 430px;
                border-radius: 16px;
                border: 1px dashed #c9d3e1;
                background: var(--panel-soft);
                display: flex;
                flex-direction: column;
                gap: 0.6rem;
                align-items: center;
                justify-content: center;
                color: #475467;
                font-size: 0.98rem;
                text-align: center;
            }}

            .upload-empty i {{
                width: 46px;
                height: 46px;
                border-radius: 14px;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                background: var(--accent-soft);
                font-size: 1.35rem;
            }}

            div[data-testid="stFileUploader"] {{
                margin-top: 1rem;
            }}

            div[data-testid="stFileUploader"] section {{
                border: 1px solid var(--line);
                border-radius: 14px;
                background: var(--panel-soft);
                padding: 0.85rem;
                box-shadow: none;
            }}

            div[data-testid="stFileUploader"] button {{
                border-radius: 12px;
                border: 0;
                background: var(--accent-soft);
                color: var(--accent);
                font-weight: 700;
                transition: all 160ms ease;
                box-shadow: none;
            }}

            div[data-testid="stFileUploader"] button:hover {{
                border: 0;
                background: #dbe8ff;
                color: var(--accent-hover);
                transform: translateY(-1px);
            }}

            div[data-testid="stFileUploader"] svg,
            button svg {{
                color: currentColor;
                fill: currentColor;
            }}

            label, .small-label {{
                color: var(--muted) !important;
                font-size: 0.78rem !important;
                font-weight: 700 !important;
                letter-spacing: 0 !important;
            }}

            div[data-baseweb="input"] {{
                border-radius: 13px;
                border: 1px solid var(--line);
                background: var(--panel-soft);
                box-shadow: none;
            }}

            div[data-baseweb="input"] input {{
                min-height: 48px;
                color: var(--text);
                font-size: 1rem;
            }}

            .stButton > button {{
                width: 100%;
                min-height: 56px;
                border-radius: 14px;
                border: 0;
                background: var(--accent);
                color: white;
                font-size: 1rem;
                font-weight: 750;
                box-shadow: 0 12px 24px rgba(37, 99, 235, 0.24);
                transition: all 160ms ease;
            }}

            .stButton > button:hover {{
                background: var(--accent-hover);
                color: white;
                transform: translateY(-1px);
                box-shadow: 0 16px 30px rgba(37, 99, 235, 0.30);
            }}

            .progress-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 1rem;
                color: #475467;
                font-size: 0.88rem;
                margin-bottom: 0.55rem;
            }}

            .progress-header strong {{
                display: inline-flex;
                align-items: center;
                gap: 0.45rem;
                color: var(--text);
                font-size: 1rem;
            }}

            .stProgress > div > div {{
                height: 18px;
                border-radius: 999px;
                background: #e8edf5;
            }}

            .stProgress > div > div > div {{
                border-radius: 999px;
                background: linear-gradient(90deg, var(--accent), #38bdf8);
            }}

            .stProgress {{
                margin-bottom: 0;
            }}

            .result-grid {{
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 0.75rem;
                margin-top: 1.1rem;
            }}

            .result-card {{
                border-radius: 14px;
                border: 1px solid var(--line);
                background: var(--panel-soft);
                padding: 0.9rem 1rem;
                box-shadow: none;
            }}

            .result-card .label {{
                display: flex;
                align-items: center;
                gap: 0.45rem;
                color: #475467;
                font-size: 0.72rem;
                font-weight: 800;
                margin-bottom: 0.45rem;
            }}

            .result-card .value {{
                color: var(--text);
                font-size: 1.35rem;
                font-weight: 800;
            }}

            .conversion-card {{
                border-radius: 12px;
                border: 1px solid var(--line);
                background: #ffffff;
                padding: 0.78rem 0.9rem;
                margin: -0.25rem 0 1rem;
            }}

            .conversion-card .label {{
                display: flex;
                align-items: center;
                gap: 0.45rem;
                color: #475467;
                font-size: 0.72rem;
                font-weight: 800;
                margin-bottom: 0.35rem;
            }}

            .conversion-card .label i {{
                color: var(--accent);
            }}

            .conversion-card .value {{
                color: var(--text);
                font-size: 1.05rem;
                font-weight: 750;
            }}

            .price-card {{
                margin-top: 1rem;
                border-radius: 16px;
                background: var(--price);
                padding: 1.25rem 1.35rem;
                box-shadow: 0 18px 38px rgba(15, 23, 42, 0.18);
            }}

            .price-card .label {{
                display: flex;
                align-items: center;
                gap: 0.45rem;
                color: #d5deeb;
                font-size: 0.75rem;
                font-weight: 800;
                margin-bottom: 0.55rem;
            }}

            .price-card .label i {{
                color: var(--price-accent);
            }}

            .app-alert {{
                display: flex;
                align-items: flex-start;
                gap: 0.7rem;
                margin: 0.85rem 0 0;
                padding: 0.85rem 0.95rem;
                border-radius: 13px;
                border: 1px solid #fed7aa;
                background: #fff7ed;
                color: #9a3412;
                font-size: 0.92rem;
                font-weight: 650;
            }}

            .app-alert i {{
                color: #c2410c;
                font-size: 1.05rem;
                margin-top: 0.05rem;
            }}

            .price-card .value {{
                color: var(--price-accent);
                font-size: 2.15rem;
                font-weight: 850;
                line-height: 1;
            }}

            @media (max-width: 860px) {{
                .result-grid {{
                    grid-template-columns: 1fr;
                }}
                .image-shell,
                .upload-empty {{
                    min-height: 300px;
                }}
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def predict_colour(image):
    modelo, clases = cargar_modelo_color()
    entrada = preprocesar_imagen_color(image)
    logits = modelo(jnp.array(entrada), entrenando=False)
    indice = int(jnp.argmax(logits, axis=-1)[0])
    return clases[indice]


def predict_shape(image):
    modelo, clases = cargar_modelo_shape()
    entrada = preprocesar_imagen_shape(image)
    probabilidades = modelo.predict(entrada, verbose=0)
    indice = int(np.argmax(probabilidades, axis=1)[0])
    return clases[indice]


def predict_clarity(image):
    modelo, transform, class_names, device = cargar_modelo_clarity()
    entrada = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = modelo(entrada)
        indice = int(torch.argmax(logits, dim=1).item())
    return class_names[indice]


# ---------------------------------------------------------------------------
# Precio (OpenFacet API, vía su servidor MCP)
#
# get_diamond_price ya resuelve internamente la interpolacion de la matriz
# de precios por quilate/color/clarity y el ajuste por relacion L/W segun
# la forma, asi que no hace falta reimplementar esa logica aqui.
# ---------------------------------------------------------------------------

OPENFACET_MCP_URL = "https://mcp.openfacet.net/"
OPENFACET_PROTOCOL_VERSION = "2026-07-28"

# OpenFacet no tiene "princess" en su catalogo de formas (round, cushion,
# radiant, emerald, oval, pear, marquise, heart). Se aproxima con "radiant"
# solo para la consulta de precio; la tarjeta de shape sigue mostrando la
# prediccion real del modelo ("princess"), esto es solo para el precio.
OPENFACET_SHAPE_OVERRIDES = {
    "princess": "radiant",
}

# OpenFacet solo soporta color D-M. El modelo puede predecir "N", que se
# aproxima con "M" (el grado mas bajo que si soporta) solo para el precio;
# la tarjeta de color sigue mostrando "N" tal cual.
OPENFACET_COLOR_OVERRIDES = {
    "N": "M",
}


def _consultar_precio_openfacet(carat_value, colour, shape, clarity):
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "_meta": {
                "io.modelcontextprotocol/protocolVersion": OPENFACET_PROTOCOL_VERSION,
                "io.modelcontextprotocol/clientCapabilities": {},
            },
            "name": "get_diamond_price",
            "arguments": {
                "carat": round(carat_value, 4),
                "color": colour.strip().upper(),
                "clarity": clarity.strip().upper(),
                "shape": shape.strip().lower(),
            },
        },
    }
    encabezados = {
        "Content-Type": "application/json",
        "MCP-Protocol-Version": OPENFACET_PROTOCOL_VERSION,
        "Mcp-Method": "tools/call",
        "Mcp-Name": "get_diamond_price",
    }
    respuesta = requests.post(
        OPENFACET_MCP_URL, json=payload, headers=encabezados, timeout=10
    )
    respuesta.raise_for_status()
    datos = respuesta.json()
    if "error" in datos:
        raise RuntimeError(datos["error"].get("message", "Error de OpenFacet"))

    resultado = datos.get("result", {})
    if "structuredContent" not in resultado:
        contenido = resultado.get("content") or []
        mensaje = contenido[0]["text"] if contenido and "text" in contenido[0] else ""
        raise RuntimeError(
            f"OpenFacet no devolvio un precio para estos parametros. {mensaje}"
        )
    return resultado["structuredContent"]


def predict_price(carat_value, colour, shape, clarity):
    forma_normalizada = shape.strip().lower()
    forma_openfacet = OPENFACET_SHAPE_OVERRIDES.get(forma_normalizada, forma_normalizada)

    color_normalizado = colour.strip().upper()
    color_openfacet = OPENFACET_COLOR_OVERRIDES.get(color_normalizado, color_normalizado)

    resultado = _consultar_precio_openfacet(
        carat_value, color_openfacet, forma_openfacet, clarity
    )
    return f"${resultado['total_usd']:,.2f}"


# ---------------------------------------------------------------------------
# Monitoreo de recursos (RAM, CPU, tiempo) durante el analisis
# ---------------------------------------------------------------------------


def medir_recursos_inicio():
    psutil.cpu_percent(percpu=True)  # referencia inicial, se descarta
    return time.perf_counter()


def medir_recursos_fin(tiempo_inicio):
    tiempo_transcurrido = time.perf_counter() - tiempo_inicio
    cpu_por_nucleo = psutil.cpu_percent(percpu=True)
    memoria_sistema = psutil.virtual_memory()
    ram_proceso_gb = psutil.Process().memory_info().rss / (1024 ** 3)

    return {
        "tiempo_segundos": tiempo_transcurrido,
        "cpu_por_nucleo": cpu_por_nucleo,
        "cpu_promedio": sum(cpu_por_nucleo) / len(cpu_por_nucleo),
        "num_nucleos": len(cpu_por_nucleo),
        "ram_proceso_gb": ram_proceso_gb,
        "ram_sistema_usada_gb": memoria_sistema.used / (1024 ** 3),
        "ram_sistema_total_gb": memoria_sistema.total / (1024 ** 3),
    }


def initialize_state():
    defaults = {
        "language": "es",
        "status": TEXT["es"]["waiting"],
        "progress": 0,
        "colour_result": "",
        "shape_result": "",
        "clarity_result": "",
        "price_result": "--",
        "resource_usage": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def render_language_menu():
    st.markdown('<div class="topbar">', unsafe_allow_html=True)
    _, language_col = st.columns([1, 0.26])
    with language_col:
        st.markdown(
            '<div class="language-label"><i class="bi bi-translate"></i>Idioma</div>',
            unsafe_allow_html=True,
        )
        selected = option_menu(
            menu_title=None,
            options=["Espanol", "English"],
            icons=["translate", "globe2"],
            default_index=0 if st.session_state.language == "es" else 1,
            orientation="horizontal",
            styles={
                "container": {
                    "padding": "3px",
                    "background-color": "#ffffff",
                    "border": f"1px solid {COLORS['line']}",
                    "border-radius": "0",
                    "box-shadow": "0 8px 18px rgba(15, 23, 42, 0.05)",
                },
                "icon": {"color": COLORS["muted"], "font-size": "12px"},
                "nav-link": {
                    "font-family": '"Segoe UI", Arial, sans-serif',
                    "font-size": "12px",
                    "font-weight": "700",
                    "color": COLORS["muted"],
                    "background-color": "#ffffff",
                    "border-radius": "0",
                    "padding": "6px 8px",
                    "margin": "0",
                },
                "nav-link-selected": {
                    "background-color": COLORS["accent_soft"],
                    "color": COLORS["accent"],
                },
            },
        )
    st.markdown("</div>", unsafe_allow_html=True)
    st.session_state.language = "es" if selected == "Espanol" else "en"

def render_progress_header(text, progress):
    st.markdown(
        f"""
        <div class="progress-header">
            <strong><i class="bi bi-activity"></i>{text["progress"]}</strong>
            <span>{progress}% &middot; {st.session_state.status}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_card(label, value, icon):
    st.markdown(
        f"""
        <div class="result-card">
            <div class="label"><i class="bi bi-{icon}"></i>{label.upper()}</div>
            <div class="value">{value or "--"}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_price_card(label, value):
    st.markdown(
        f"""
        <div class="price-card">
            <div class="label"><i class="bi bi-cash-coin"></i>{label.upper()}</div>
            <div class="value">{value or "--"}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_carat_equivalent(label, carats):
    value = f"{carats:.4f} ct" if carats is not None else "-- ct"
    st.markdown(
        f"""
        <div class="conversion-card">
            <div class="label"><i class="bi bi-arrow-left-right"></i>{label.upper()}</div>
            <div class="value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label, value, icon):
    st.markdown(
        f"""
        <div class="conversion-card">
            <div class="label"><i class="bi bi-{icon}"></i>{label.upper()}</div>
            <div class="value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_resource_panel(text, usage):
    st.markdown(
        f'<div class="section-title"><i class="bi bi-speedometer2"></i>{text["resources_title"]}</div>',
        unsafe_allow_html=True,
    )

    if usage:
        tiempo_valor = f"{usage['tiempo_segundos']:.2f} s"
        ram_valor = f"{usage['ram_proceso_gb']:.2f} GB"
        cpu_valor = f"{usage['cpu_promedio']:.0f}% ({usage['num_nucleos']} {text['cores_unit']})"
    else:
        tiempo_valor = ram_valor = cpu_valor = "--"

    render_metric_card(text["time_label"], tiempo_valor, "clock-history")
    render_metric_card(text["ram_label"], ram_valor, "memory")
    render_metric_card(text["cpu_label"], cpu_valor, "cpu")

    if usage:
        nucleos_df = pd.DataFrame(
            {"CPU %": usage["cpu_por_nucleo"]},
            index=[f"C{i}" for i in range(usage["num_nucleos"])],
        )
        st.bar_chart(nucleos_df, height=110)


def render_alert(message):
    st.markdown(
        f"""
        <div class="app-alert">
            <i class="bi bi-exclamation-triangle"></i>
            <span>{message}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def image_to_data_uri(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def run_analysis(image, carat_value, text, progress_slot, progress_bar):
    steps = (
        ("Colour", lambda: predict_colour(image), "colour_result"),
        ("Shape", lambda: predict_shape(image), "shape_result"),
        ("Clarity", lambda: predict_clarity(image), "clarity_result"),
        (
            "Price",
            lambda: predict_price(
                carat_value,
                st.session_state.colour_result,
                st.session_state.shape_result,
                st.session_state.clarity_result,
            ),
            "price_result",
        ),
    )

    st.session_state.progress = 0
    st.session_state.colour_result = ""
    st.session_state.shape_result = ""
    st.session_state.clarity_result = ""
    st.session_state.price_result = "--"

    tiempo_inicio = medir_recursos_inicio()

    for index, (step_name, predictor, target_key) in enumerate(steps):
        st.session_state.status = text["running"].format(step=step_name)
        with progress_slot:
            render_progress_header(text, st.session_state.progress)
        st.session_state[target_key] = predictor()
        st.session_state.progress = (index + 1) * 25
        progress_bar.progress(st.session_state.progress)

    st.session_state.resource_usage = medir_recursos_fin(tiempo_inicio)

    st.session_state.status = text["done"]
    with progress_slot:
        render_progress_header(text, st.session_state.progress)


def main():
    configure_page()
    initialize_state()
    render_language_menu()

    text = TEXT[st.session_state.language]

    st.title(APP_TITLE)
    st.markdown(f'<div class="app-subtitle">{text["subtitle"]}</div>', unsafe_allow_html=True)

    image_col, controls_col, resources_col = st.columns([1.3, 1, 0.9], gap="large")

    with image_col:
        with st.container(border=True):
            st.markdown(
                f'<div class="section-title"><i class="bi bi-gem"></i>{text["upload_title"]}</div>',
                unsafe_allow_html=True,
            )
            uploaded_image = st.file_uploader(
                text["upload_button"],
                type=["png", "jpg", "jpeg", "webp", "bmp"],
                label_visibility="visible",
            )

            if uploaded_image:
                image = Image.open(uploaded_image).convert("RGB")
                image_uri = image_to_data_uri(image)
                st.markdown(
                    f'<div class="image-shell"><img src="{image_uri}" alt="{text["upload_title"]}"></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="upload-empty">
                        <i class="bi bi-image"></i>
                        <span>{text["upload_hint"]}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with controls_col:
        with st.container(border=True):
            weight_grams = st.text_input(text["carat"], value="", placeholder="0.15")
            carat_equivalent = None
            try:
                parsed_weight = float(weight_grams)
                if parsed_weight > 0:
                    carat_equivalent = parsed_weight / 0.2
            except ValueError:
                pass
            render_carat_equivalent(text["carat_equivalent"], carat_equivalent)

            analyze_clicked = st.button(text["analyze"], use_container_width=True)

            progress_header = st.empty()
            with progress_header:
                render_progress_header(text, st.session_state.progress)
            progress_bar = st.progress(st.session_state.progress)

            if analyze_clicked:
                if not uploaded_image:
                    render_alert(text["required_image"])
                else:
                    try:
                        weight_value = float(weight_grams)
                    except ValueError:
                        render_alert(text["invalid_carat"])
                    else:
                        if weight_value <= 0:
                            render_alert(text["invalid_carat"])
                        else:
                            carat_value = weight_value / 0.2
                            run_analysis(image, carat_value, text, progress_header, progress_bar)

            st.markdown('<div class="result-grid">', unsafe_allow_html=True)
            result_cols = st.columns(3)
            with result_cols[0]:
                render_result_card(text["colour"], st.session_state.colour_result, "palette")
            with result_cols[1]:
                render_result_card(text["shape"], st.session_state.shape_result, "hexagon")
            with result_cols[2]:
                render_result_card(text["clarity"], st.session_state.clarity_result, "stars")
            st.markdown("</div>", unsafe_allow_html=True)

            render_price_card(text["price"], st.session_state.price_result)

    with resources_col:
        with st.container(border=True):
            render_resource_panel(text, st.session_state.resource_usage)


if __name__ == "__main__":
    main()
