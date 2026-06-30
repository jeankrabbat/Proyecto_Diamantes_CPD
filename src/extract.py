import os
from kaggle.api.kaggle_api_extended import KaggleApi

def descargar_dataset():
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files('aayushpurswani/diamond-images-dataset', 
                               path='data/raw/', 
                               unzip=True)
    print("Dataset descargado en data/raw/")

def obtener_rutas_imagenes(carpeta_padre):
    """Obtiene la lista de rutas de TODAS las imágenes en todas las subcarpetas"""
    rutas = []
    ruta_web_scraped = os.path.join(carpeta_padre, 'web_scraped')

    for subcarpeta in os.listdir(ruta_web_scraped):
        ruta_subcarpeta = os.path.join(ruta_web_scraped, subcarpeta)
        
        if not os.path.isdir(ruta_subcarpeta):
            continue
        
        for archivo in os.listdir(ruta_subcarpeta):
            if archivo.endswith(('.jpg', '.png', '.jpeg')):
                ruta_completa = os.path.join(ruta_subcarpeta, archivo)
                rutas.append(ruta_completa)
    
    print(f"Se encontraron {len(rutas)} imágenes")
    return rutas

def obtener_labels(rutas):
    labels = []
    
    for ruta in rutas:
        nombre_archivo = os.path.basename(ruta)
        nombre_sin_extension = nombre_archivo.split('.')[0]
        labels.append(nombre_sin_extension)
    
    return labels