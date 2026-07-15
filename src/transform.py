import os
from pathlib import Path
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd


# Colores válidos según la escala solicitada: D hasta Z.
COLORES_VALIDOS = list("DEFGHIJKLMNOPQRSTUVWXYZ")

# Claridades estándar.
CLARIDADES_VALIDAS = [
    "FL",
    "IF",
    "VVS1",
    "VVS2",
    "VS1",
    "VS2",
    "SI1",
    "SI2",
    "I1",
    "I2",
    "I3"
]

# Nombres estandarizados de formas.
MAPEO_FORMAS = {
    "round brilliant": "round",
    "round brilliant cut": "round",
    "marquese": "marquise"
}


def normalizar_forma(valor):
    """
    Normaliza el nombre de la forma del diamante.
    """
    if pd.isna(valor):
        return None

    forma = str(valor).strip().lower()

    return MAPEO_FORMAS.get(forma, forma)


def normalizar_claridad(valor):
    """
    Convierte la claridad a mayúsculas y valida que corresponda
    a una categoría estándar.
    """
    if pd.isna(valor):
        return None

    claridad = str(valor).strip().upper()

    if claridad in CLARIDADES_VALIDAS:
        return claridad

    return None


def normalizar_color(valor):
    """
    Normaliza el color del diamante.

    Acepta:
    - Colores individuales: D, E, F, ..., Z.
    - Rangos como Y-Z, los cuales se conservan como rango.

    Returns
    -------
    str o None
        Color o rango de color válido.
    """
    if pd.isna(valor):
        return None

    color = str(valor).strip().upper().replace(" ", "")

    if color in COLORES_VALIDOS:
        return color

    if "-" in color:
        partes = color.split("-")

        if len(partes) == 2:
            inicio, fin = partes

            if inicio in COLORES_VALIDOS and fin in COLORES_VALIDOS:
                if COLORES_VALIDOS.index(inicio) <= COLORES_VALIDOS.index(fin):
                    return color

    return None


def obtener_grupo_color(valor):
    """
    Agrupa los colores según las categorías compartidas:

    D
    E-F
    G-H
    I-J
    K-M
    N-R
    S-Z

    Los rangos se clasifican usando su primera letra.
    """
    color = normalizar_color(valor)

    if color is None:
        return None

    letra_inicial = color.split("-")[0]

    if letra_inicial == "D":
        return "D"

    if letra_inicial in ["E", "F"]:
        return "E-F"

    if letra_inicial in ["G", "H"]:
        return "G-H"

    if letra_inicial in ["I", "J"]:
        return "I-J"

    if letra_inicial in ["K", "L", "M"]:
        return "K-M"

    if letra_inicial in ["N", "O", "P", "Q", "R"]:
        return "N-R"

    if letra_inicial in ["S", "T", "U", "V", "W", "X", "Y", "Z"]:
        return "S-Z"

    return None


def preparar_metadata(metadata, carpeta_datos="data/raw"):
    """
    Limpia y estandariza el DataFrame de metadatos.

    También construye la ruta completa de cada imagen utilizando
    la columna path_to_img.

    Parameters
    ----------
    metadata : pandas.DataFrame
        DataFrame original del CSV.

    carpeta_datos : str
        Carpeta donde se encuentra el dataset.

    Returns
    -------
    pandas.DataFrame
        Metadatos preparados y validados.
    """
    columnas_requeridas = {
        "path_to_img",
        "shape",
        "clarity",
        "colour"
    }

    columnas_faltantes = columnas_requeridas - set(metadata.columns)

    if columnas_faltantes:
        raise ValueError(
            "Faltan columnas necesarias en el CSV: "
            f"{sorted(columnas_faltantes)}"
        )

    datos = metadata.copy()

    datos["shape"] = datos["shape"].apply(normalizar_forma)
    datos["clarity"] = datos["clarity"].apply(normalizar_claridad)
    datos["colour"] = datos["colour"].apply(normalizar_color)
    datos["colour_group"] = datos["colour"].apply(obtener_grupo_color)

    carpeta_datos = Path(carpeta_datos)

    datos["full_path"] = datos["path_to_img"].apply(
        lambda ruta: str(carpeta_datos / Path(str(ruta)))
    )

    registros_iniciales = len(datos)

    datos = datos.dropna(
        subset=[
            "shape",
            "clarity",
            "colour",
            "colour_group",
            "full_path"
        ]
    ).copy()

    datos = datos[
        datos["full_path"].apply(os.path.isfile)
    ].copy()

    datos = datos.reset_index(drop=True)

    eliminados = registros_iniciales - len(datos)

    print(f"Registros iniciales: {registros_iniciales}")
    print(f"Registros válidos: {len(datos)}")
    print(f"Registros descartados: {eliminados}")

    return datos


def crear_muestra_balanceada(
    metadata,
    muestras_por_forma=5,
    limite_total=None,
    semilla=42
):
    """
    Crea una muestra pequeña y balanceada principalmente por forma.

    Dentro de cada forma intenta seleccionar variedad de:
    - color;
    - claridad.

    Conserva todas las columnas del DataFrame y evita duplicados.

    Parameters
    ----------
    metadata : pandas.DataFrame
        Metadatos previamente preparados.

    muestras_por_forma : int
        Cantidad máxima de imágenes por cada forma.

    limite_total : int o None
        Cantidad máxima de registros en la muestra.

    semilla : int
        Semilla para reproducir siempre la misma selección.

    Returns
    -------
    pandas.DataFrame
        Muestra balanceada.
    """
    if metadata.empty:
        raise ValueError(
            "No se puede crear una muestra porque los metadatos están vacíos."
        )

    muestras = []

    for numero_grupo, (forma, grupo) in enumerate(
        metadata.groupby("shape")
    ):
        cantidad = min(muestras_por_forma, len(grupo))

        # Primero se buscan combinaciones distintas de color y claridad.
        candidatos_diversos = (
            grupo
            .drop_duplicates(
                subset=["colour_group", "clarity"]
            )
            .sample(
                frac=1,
                random_state=semilla + numero_grupo
            )
        )

        seleccion = candidatos_diversos.head(cantidad)

        # Si todavía faltan registros, se completa con imágenes
        # de esa misma forma que no hayan sido seleccionadas.
        faltantes = cantidad - len(seleccion)

        if faltantes > 0:
            restantes = grupo.drop(
                index=seleccion.index,
                errors="ignore"
            )

            if not restantes.empty:
                adicionales = restantes.sample(
                    n=min(faltantes, len(restantes)),
                    random_state=semilla + numero_grupo
                )

                seleccion = pd.concat(
                    [seleccion, adicionales]
                )

        muestras.append(seleccion)

    muestra_final = pd.concat(
        muestras,
        ignore_index=True
    )

    muestra_final = muestra_final.drop_duplicates(
        subset=["path_to_img"],
        keep="first"
    )

    if limite_total is not None and len(muestra_final) > limite_total:
        muestra_final = muestra_final.sample(
            n=limite_total,
            random_state=semilla
        )

    muestra_final = (
        muestra_final
        .sort_values(["shape", "colour_group", "clarity"])
        .reset_index(drop=True)
    )

    print("\nMuestra balanceada creada.")
    print(f"Cantidad total: {len(muestra_final)}")

    print("\nDistribución por forma:")
    print(
        muestra_final["shape"]
        .value_counts()
        .sort_index()
    )

    print("\nDistribución por grupo de color:")
    print(
        muestra_final["colour_group"]
        .value_counts()
        .sort_index()
    )

    print("\nDistribución por claridad:")
    print(
        muestra_final["clarity"]
        .value_counts()
        .sort_index()
    )

    print("\nValores faltantes:")
    print(
        muestra_final[
            ["shape", "colour", "colour_group", "clarity"]
        ].isna().sum()
    )

    return muestra_final

def obtener_una_imagen_por_forma(metadata, semilla=42):
    """
    Obtiene exactamente una imagen por cada forma disponible.

    Esto evita que el muestreo visual repita varias imágenes Round.
    """
    if metadata.empty:
        raise ValueError(
            "Los metadatos están vacíos."
        )

    muestra = (
        metadata
        .groupby("shape", group_keys=False)
        .sample(n=1, random_state=semilla)
        .sort_values("shape")
        .reset_index(drop=True)
    )

    return muestra


def procesar_imagen(args):
    """
    Lee, redimensiona y normaliza una imagen.

    Parameters
    ----------
    args : tuple
        ruta_imagen, tamaño

    Returns
    -------
    tuple
        ruta de la imagen, imagen procesada o None.
    """
    ruta_imagen, tamaño = args

    img = cv2.imread(ruta_imagen)

    if img is None:
        return ruta_imagen, None

    try:
        img = cv2.resize(
            img,
            tamaño,
            interpolation=cv2.INTER_AREA
        )

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        img = img.astype(np.float32) / 255.0

        return ruta_imagen, img

    except Exception:
        return ruta_imagen, None


def procesar_lote(
    rutas_imagenes,
    tamaño=(224, 224),
    num_procesos=4,
    tamaño_lote=5000,
    carpeta_salida="data/processed",
    prefijo="lote"
):
    """
    Procesa imágenes por lotes y conserva el mismo orden de entrada.

    A diferencia de la versión anterior, utiliza Pool.map en lugar de
    imap_unordered. Esto evita que las imágenes se desordenen y pierdan
    correspondencia con sus etiquetas.

    Parameters
    ----------
    rutas_imagenes : list
        Rutas completas de las imágenes.

    tamaño : tuple
        Resolución final, por ejemplo (224, 224).

    num_procesos : int
        Número de procesos paralelos.

    tamaño_lote : int
        Número máximo de imágenes por archivo.

    carpeta_salida : str
        Carpeta donde se guardarán los lotes.

    prefijo : str
        Prefijo usado para los archivos NPZ.

    Returns
    -------
    list
        Información de los archivos generados.
    """
    if not rutas_imagenes:
        raise ValueError(
            "La lista de rutas de imágenes está vacía."
        )

    ancho, alto = tamaño

    carpeta_resolucion = Path(carpeta_salida) / f"{ancho}x{alto}"
    carpeta_resolucion.mkdir(parents=True, exist_ok=True)

    total = len(rutas_imagenes)
    archivos_generados = []

    contador_lote = 0

    for inicio in range(0, total, tamaño_lote):
        fin = min(inicio + tamaño_lote, total)
        rutas_lote = rutas_imagenes[inicio:fin]

        print(
            f"Procesando resolución {ancho}x{alto}, "
            f"lote {inicio}-{fin}/{total}..."
        )

        argumentos = [
            (ruta, tamaño)
            for ruta in rutas_lote
        ]

        with Pool(processes=num_procesos) as pool:
            resultados = pool.map(
                procesar_imagen,
                argumentos,
                chunksize=50
            )

        imagenes_validas = []
        rutas_validas = []
        rutas_fallidas = []

        for ruta, imagen in resultados:
            if imagen is None:
                rutas_fallidas.append(ruta)
            else:
                rutas_validas.append(ruta)
                imagenes_validas.append(imagen)

        if not imagenes_validas:
            print(
                "Advertencia: no se pudo procesar ninguna imagen "
                f"del lote {contador_lote}."
            )
            contador_lote += 1
            continue

        array_lote = np.stack(
            imagenes_validas,
            axis=0
        )

        ruta_archivo = (
            carpeta_resolucion
            / f"{prefijo}_{contador_lote}.npz"
        )

        np.savez_compressed(
            ruta_archivo,
            images=array_lote,
            paths=np.array(rutas_validas)
        )

        archivos_generados.append(
            {
                "archivo": str(ruta_archivo),
                "resolucion": f"{ancho}x{alto}",
                "imagenes_validas": len(imagenes_validas),
                "imagenes_fallidas": len(rutas_fallidas)
            }
        )

        print(
            f"  Lote guardado: {len(imagenes_validas)} imágenes"
        )

        if rutas_fallidas:
            print(
                f"  Imágenes fallidas: {len(rutas_fallidas)}"
            )

        contador_lote += 1

    print(
        f"\nProcesamiento {ancho}x{alto} finalizado. "
        f"{len(archivos_generados)} lotes guardados."
    )

    return archivos_generados


def procesar_multiples_resoluciones(
    rutas_imagenes,
    resoluciones=None,
    num_procesos=4,
    tamaño_lote=5000,
    carpeta_salida="data/processed"
):
    """
    Procesa las mismas imágenes en múltiples resoluciones.

    Por defecto genera:
    - 224 × 224
    - 256 × 256
    """
    if resoluciones is None:
        resoluciones = [
            (224, 224),
            (256, 256)
        ]

    resultados = {}

    for tamaño in resoluciones:
        clave = f"{tamaño[0]}x{tamaño[1]}"

        resultados[clave] = procesar_lote(
            rutas_imagenes=rutas_imagenes,
            tamaño=tamaño,
            num_procesos=num_procesos,
            tamaño_lote=tamaño_lote,
            carpeta_salida=carpeta_salida,
            prefijo="lote"
        )

    return resultados


def guardar_metadata_muestra(
    metadata,
    ruta_salida="data/processed/metadata_muestra.csv"
):
    """
    Guarda los metadatos utilizados en el procesamiento.
    """
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    metadata.to_csv(
        ruta_salida,
        index=False
    )

    print(
        f"Metadatos guardados en: {ruta_salida}"
    )