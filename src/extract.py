import os
from pathlib import Path

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi


EXTENSIONES_VALIDAS = {".jpg", ".jpeg", ".png"}


def descargar_dataset(
    ruta_destino="data/raw",
    dataset="aayushpurswani/diamond-images-dataset"
):
    """
    Descarga y descomprime el dataset de diamantes desde Kaggle.

    Parameters
    ----------
    ruta_destino : str
        Carpeta donde se descargará el dataset.

    dataset : str
        Identificador del dataset en Kaggle.
    """
    os.makedirs(ruta_destino, exist_ok=True)

    try:
        api = KaggleApi()
        api.authenticate()

        print("Descargando dataset desde Kaggle...")

        api.dataset_download_files(
            dataset,
            path=ruta_destino,
            unzip=True
        )

        print(f"Dataset descargado correctamente en: {ruta_destino}")

    except Exception as error:
        raise RuntimeError(
            "No fue posible descargar el dataset. "
            "Verifica que Kaggle esté configurado correctamente "
            "y que exista el archivo kaggle.json."
        ) from error


def obtener_rutas_imagenes(carpeta_padre):
    """
    Obtiene las rutas de todas las imágenes encontradas dentro de
    la carpeta web_scraped y sus subcarpetas.

    La búsqueda es recursiva, por lo que funciona aunque existan
    varios niveles de carpetas.

    Parameters
    ----------
    carpeta_padre : str
        Carpeta principal del dataset. Por ejemplo: data/raw

    Returns
    -------
    list
        Lista ordenada con las rutas de las imágenes.
    """
    carpeta_padre = Path(carpeta_padre)
    ruta_web_scraped = carpeta_padre / "web_scraped"

    if not carpeta_padre.exists():
        raise FileNotFoundError(
            f"No existe la carpeta principal: {carpeta_padre}"
        )

    if not ruta_web_scraped.exists():
        raise FileNotFoundError(
            f"No se encontró la carpeta: {ruta_web_scraped}"
        )

    rutas = [
        str(ruta)
        for ruta in ruta_web_scraped.rglob("*")
        if ruta.is_file()
        and ruta.suffix.lower() in EXTENSIONES_VALIDAS
    ]

    rutas = sorted(rutas)

    print(f"Se encontraron {len(rutas)} imágenes.")

    if not rutas:
        print(
            "Advertencia: no se encontraron archivos JPG, JPEG o PNG "
            "dentro de web_scraped."
        )

    return rutas


def obtener_labels(rutas):
    """
    Obtiene una etiqueta básica a partir del nombre de cada imagen.

    Parameters
    ----------
    rutas : list
        Lista de rutas de imágenes.

    Returns
    -------
    list
        Lista de nombres de archivos sin extensión.
    """
    labels = []

    for ruta in rutas:
        nombre_archivo = Path(ruta).stem
        labels.append(nombre_archivo)

    return labels


def buscar_archivo_csv(carpeta_padre):
    """
    Busca archivos CSV dentro de la carpeta del dataset.

    Parameters
    ----------
    carpeta_padre : str
        Carpeta principal del dataset.

    Returns
    -------
    str
        Ruta del archivo CSV encontrado.

    Raises
    ------
    FileNotFoundError
        Si no se encuentra ningún archivo CSV.

    RuntimeError
        Si se encuentra más de un CSV y no se puede decidir cuál usar.
    """
    carpeta_padre = Path(carpeta_padre)

    if not carpeta_padre.exists():
        raise FileNotFoundError(
            f"No existe la carpeta: {carpeta_padre}"
        )

    archivos_csv = sorted(carpeta_padre.rglob("*.csv"))

    if not archivos_csv:
        raise FileNotFoundError(
            f"No se encontró ningún archivo CSV dentro de {carpeta_padre}"
        )

    if len(archivos_csv) > 1:
        print("Se encontraron varios archivos CSV:")

        for indice, archivo in enumerate(archivos_csv, start=1):
            print(f"{indice}. {archivo}")

        print(
            f"Se utilizará el primer archivo encontrado: "
            f"{archivos_csv[0]}"
        )

    return str(archivos_csv[0])


def cargar_metadata(ruta_csv):
    """
    Carga el archivo CSV que contiene la información de los diamantes.

    También normaliza los nombres de las columnas para facilitar
    su utilización en las siguientes etapas del ETL.

    Parameters
    ----------
    ruta_csv : str
        Ruta del archivo CSV.

    Returns
    -------
    pandas.DataFrame
        DataFrame con los metadatos.
    """
    ruta_csv = Path(ruta_csv)

    if not ruta_csv.exists():
        raise FileNotFoundError(
            f"No existe el archivo CSV: {ruta_csv}"
        )

    try:
        dataframe = pd.read_csv(ruta_csv)

    except Exception as error:
        raise RuntimeError(
            f"No fue posible leer el archivo CSV: {ruta_csv}"
        ) from error

    if dataframe.empty:
        raise ValueError(
            f"El archivo CSV está vacío: {ruta_csv}"
        )

    dataframe.columns = (
        dataframe.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    print(f"Metadata cargada correctamente: {len(dataframe)} registros.")
    print(f"Columnas encontradas: {dataframe.columns.tolist()}")

    return dataframe


def extraer_datos(carpeta_padre="data/raw"):
    """
    Ejecuta la etapa completa de extracción.

    Busca:
    - Todas las imágenes.
    - El archivo CSV.
    - Los metadatos del dataset.

    Parameters
    ----------
    carpeta_padre : str
        Carpeta principal del dataset.

    Returns
    -------
    tuple
        rutas_imagenes, labels, metadata
    """
    print("\nIniciando etapa de extracción...")

    rutas_imagenes = obtener_rutas_imagenes(carpeta_padre)
    labels = obtener_labels(rutas_imagenes)

    ruta_csv = buscar_archivo_csv(carpeta_padre)
    metadata = cargar_metadata(ruta_csv)

    print("Etapa de extracción finalizada correctamente.\n")

    return rutas_imagenes, labels, metadata