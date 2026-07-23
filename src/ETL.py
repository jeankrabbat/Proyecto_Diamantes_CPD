# 1. Configuracion general

from pathlib import Path
import sys
import os

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

os.chdir(PROJECT_ROOT)

SRC_PATH = str(PROJECT_ROOT / "src")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

PLOTS_OUTPUT_DIR = Path("/data/ulead-36/Proyecto/images")
PLOTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def mostrar(objeto):
    """Print tables in the console as a replacement for Jupyter mostrar()."""
    if hasattr(objeto, "to_string"):
        print(objeto.to_string())
    else:
        print(objeto)


def guardar_figura(nombre_archivo):
    """Save the current matplotlib figure to the configured plots folder."""
    ruta_salida = PLOTS_OUTPUT_DIR / nombre_archivo
    plt.savefig(ruta_salida, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot guardado: {ruta_salida}")

from extract import descargar_dataset, extraer_datos
from transform import (
    preparar_metadata,
    crear_muestra_balanceada,
    obtener_una_imagen_por_forma,
    guardar_metadata_muestra
)

print(f"Raiz del proyecto: {PROJECT_ROOT}")


# 2. Parámetros de ejecución

# True: procesa una muestra pequeña para pruebas.
# False: procesa todo el dataset.
MODO_PRUEBA = False

# Cambiar a True solamente si el dataset todavía no existe en data/raw.
DESCARGAR_DATASET = False

# Parámetros de prueba
MUESTRAS_POR_FORMA = 5
LIMITE_MUESTRA = 100
NUM_PROCESOS_PRUEBA = 2
TAMANO_LOTE_PRUEBA = 50

# Parámetros para el dataset completo
NUM_PROCESOS_COMPLETO = 8
TAMANO_LOTE_COMPLETO = 2000
CARPETA_SALIDA_PRUEBA = "data/processed_sample"
CARPETA_SALIDA_COMPLETA = "data/processed"

print("Modo seleccionado:", "PRUEBA" if MODO_PRUEBA else "DATASET COMPLETO")


# 3. Descargar el dataset de Kaggle (opcional)

if DESCARGAR_DATASET:
    print("Descargando dataset...")
    descargar_dataset(
        ruta_destino="data/raw"
    )
else:
    print("Descarga omitida. Se utilizará el dataset existente en data/raw.")


# 4. Extraer rutas, etiquetas y metadatos

rutas, labels, metadata = extraer_datos(
    carpeta_padre="data/raw"
)

print(f"Total de imágenes encontradas: {len(rutas)}")
print(f"Total de etiquetas: {len(labels)}")
print(f"Total de registros en el CSV: {len(metadata)}")

mostrar(metadata.head())


# 5. Analizar la distribución original

print("Distribución original por forma:")
mostrar(
    metadata["shape"]
    .value_counts(dropna=False)
    .rename_axis("shape")
    .reset_index(name="cantidad")
)

print("Distribución original por color:")
mostrar(
    metadata["colour"]
    .value_counts(dropna=False)
    .rename_axis("colour")
    .reset_index(name="cantidad")
)

print("Distribución original por claridad:")
mostrar(
    metadata["clarity"]
    .value_counts(dropna=False)
    .rename_axis("clarity")
    .reset_index(name="cantidad")
)


# 6. Preparar y validar los metadatos

datos_validos = preparar_metadata(
    metadata=metadata,
    carpeta_datos="data/raw"
)

print(f"Registros válidos para procesar: {len(datos_validos)}")

mostrar(
    datos_validos[
        [
            "path_to_img",
            "shape",
            "colour",
            "clarity",
            "full_path"
        ]
    ].head()
)


# 7. Seleccionar muestra o dataset completo

if MODO_PRUEBA:
    datos_a_procesar = crear_muestra_balanceada(
        metadata=datos_validos,
        muestras_por_forma=MUESTRAS_POR_FORMA,
        limite_total=LIMITE_MUESTRA,
        semilla=42
    )

    carpeta_salida = CARPETA_SALIDA_PRUEBA
    num_procesos = NUM_PROCESOS_PRUEBA
    tamano_lote = TAMANO_LOTE_PRUEBA

else:
    datos_a_procesar = datos_validos.copy()

    carpeta_salida = CARPETA_SALIDA_COMPLETA
    num_procesos = NUM_PROCESOS_COMPLETO
    tamano_lote = TAMANO_LOTE_COMPLETO

print(f"Imágenes a procesar: {len(datos_a_procesar)}")
print(f"Carpeta de salida: {carpeta_salida}")
print(f"Número de procesos: {num_procesos}")
print(f"Tamaño de lote: {tamano_lote}")


# 8. Revisar distribuciones de los datos seleccionados

def tabla_distribucion(serie, nombre_columna):
    """
    Construye una tabla de frecuencias con Cantidad y % (ordenada por índice).
    """
    cantidad = serie.value_counts().sort_index()
    porcentaje = (cantidad / cantidad.sum() * 100).round(2)

    tabla = pd.DataFrame({
        "Cantidad": cantidad,
        "%": porcentaje
    })
    tabla.index.name = nombre_columna
    return tabla


print("Distribución por forma (shape):")
mostrar(tabla_distribucion(datos_a_procesar["shape"], "shape"))

print("\nDistribución por color (colour):")
mostrar(tabla_distribucion(datos_a_procesar["colour"], "colour"))

print("\nDistribución por claridad (clarity):")
mostrar(tabla_distribucion(datos_a_procesar["clarity"], "clarity"))


print("\nValores faltantes:")
faltantes = datos_a_procesar[["shape", "colour", "clarity"]].isna().sum()
faltantes_pct = (faltantes / len(datos_a_procesar) * 100).round(2)

tabla_faltantes = pd.DataFrame({
    "Cantidad": faltantes,
    "%": faltantes_pct
})
mostrar(tabla_faltantes)


# 9. Visualizar una imagen por cada forma

muestra_visual = obtener_una_imagen_por_forma(
    metadata=datos_validos,
    semilla=42
)

cantidad = len(muestra_visual)
columnas = 4
filas = int(np.ceil(cantidad / columnas))

plt.figure(figsize=(16, filas * 4))

for posicion, (_, registro) in enumerate(
    muestra_visual.iterrows(),
    start=1
):
    imagen = cv2.imread(registro["full_path"])

    plt.subplot(filas, columnas, posicion)

    if imagen is not None:
        imagen = cv2.cvtColor(
            imagen,
            cv2.COLOR_BGR2RGB
        )
        plt.imshow(imagen)

    plt.title(
        f"{registro['shape'].title()}\n"
        f"Color: {registro['colour']} | "
        f"Claridad: {registro['clarity']}"
    )
    plt.axis("off")

plt.tight_layout()
guardar_figura("muestra_por_forma.png")


# 10. Analizar la distribución del dataset

# Orden estándar GIA para colores D-N + rangos extendidos que existen en el dataset
orden_colores = list("DEFGHIJKLMN") + ["O-P", "Q-R", "S-T", "U-V", "W-X", "Y-Z"]

# Orden estándar GIA para claridad
orden_claridad = [
    "FL", "IF", "VVS1", "VVS2", "VS1", "VS2",
    "SI1", "SI2", "I1", "I2", "I3"
]

def agregar_etiquetas_barras(ax):
    """
    Agrega la cantidad sobre cada barra del gráfico.
    """
    for barra in ax.patches:
        altura = barra.get_height()
        ax.annotate(
            f"{int(altura)}",
            (barra.get_x() + barra.get_width() / 2, altura),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 3),
            textcoords="offset points"
        )

# --------------------------
# Distribución por forma
# --------------------------
conteo_formas = (
    datos_a_procesar["shape"]
    .value_counts()
    .sort_index()
)

fig, ax = plt.subplots(figsize=(10, 5))
conteo_formas.plot(kind="bar", ax=ax)
ax.set_title("Distribución por forma")
ax.set_xlabel("Forma")
ax.set_ylabel("Cantidad")
ax.tick_params(axis="x", rotation=45)
agregar_etiquetas_barras(ax)
plt.tight_layout()
guardar_figura("distribucion_por_forma.png")

# --------------------------
# Distribución por color
# --------------------------
conteo_colores = (
    datos_a_procesar["colour"]
    .value_counts()
    .reindex(orden_colores, fill_value=0)
)

fig, ax = plt.subplots(figsize=(14, 5))
conteo_colores.plot(kind="bar", ax=ax)
ax.set_title("Distribución por color")
ax.set_xlabel("Color")
ax.set_ylabel("Cantidad")
ax.tick_params(axis="x", rotation=0)
agregar_etiquetas_barras(ax)
plt.tight_layout()
guardar_figura("distribucion_por_color.png")

# --------------------------
# Distribución por claridad
# --------------------------
conteo_claridad = (
    datos_a_procesar["clarity"]
    .value_counts()
    .reindex(orden_claridad, fill_value=0)
)

fig, ax = plt.subplots(figsize=(12, 5))
conteo_claridad.plot(kind="bar", ax=ax)
ax.set_title("Distribución por claridad")
ax.set_xlabel("Claridad")
ax.set_ylabel("Cantidad")
ax.tick_params(axis="x", rotation=45)
agregar_etiquetas_barras(ax)
plt.tight_layout()
guardar_figura("distribucion_por_claridad.png")


# 11. Filtrar categorías con muy pocas observaciones

# Categorías a eliminar

claridades_eliminar = [
    "FL",
    "I1",
    "I2",
    "I3"
]

colores_eliminar = [
    "O-P",
    "Q-R",
    "S-T",
    "U-V",
    "W-X",
    "Y-Z"
]

# Resumen inicial

registros_iniciales = len(datos_a_procesar)

print("=" * 60)
print("FILTRADO DEL DATASET")
print("=" * 60)

print(f"\nRegistros iniciales: {registros_iniciales:,}")

print("\nClaridades eliminadas:")
for claridad in claridades_eliminar:
    print(f"  • {claridad}")

print("\nColores eliminados:")
for color in colores_eliminar:
    print(f"  • {color}")

# Aplicar filtros

datos_a_procesar = (
    datos_a_procesar[
        (~datos_a_procesar["clarity"].isin(claridades_eliminar)) &
        (~datos_a_procesar["colour"].isin(colores_eliminar))
    ]
    .reset_index(drop=True)
)


registros_finales = len(datos_a_procesar)

print("RESULTADO DEL FILTRADO")

print(f"Registros finales:    {registros_finales:,}")
print(f"Registros eliminados: {registros_iniciales - registros_finales:,}")

print("\nDistribución final por forma:")
mostrar(
    datos_a_procesar["shape"]
    .value_counts()
    .sort_index()
    .to_frame("Cantidad")
)

print("\nDistribución final por color:")
mostrar(
    datos_a_procesar["colour"]
    .value_counts()
    .sort_index()
    .to_frame("Cantidad")
)

print("\nDistribución final por claridad:")
mostrar(
    datos_a_procesar["clarity"]
    .value_counts()
    .sort_index()
    .to_frame("Cantidad")
)


# 12. Guardar los metadatos de las imágenes seleccionadas

ruta_metadata_salida = os.path.join(
    carpeta_salida,
    "metadata_procesada.csv"
)

guardar_metadata_muestra(
    metadata=datos_a_procesar,
    ruta_salida=ruta_metadata_salida
)
