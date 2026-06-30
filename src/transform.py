import cv2
import numpy as np
from multiprocessing import Pool
import os

def procesar_imagen(args):
    ruta_imagen, tamaño = args
    img = cv2.imread(ruta_imagen)
    
    if img is None:
        return None
    
    img = cv2.resize(img, tamaño)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    
    return img

def procesar_lote(rutas_imagenes, tamaño=(224, 224), num_procesos=4, tamaño_lote=5000):
    total = len(rutas_imagenes)
    contador = 0
    
    for inicio in range(0, total, tamaño_lote):
        fin = min(inicio + tamaño_lote, total)
        rutas_lote = rutas_imagenes[inicio:fin]
        
        print(f"Procesando lote {inicio}-{fin}/{total}...")
        
        argumentos = [(ruta, tamaño) for ruta in rutas_lote]
        
        with Pool(num_procesos) as pool:
            imagenes = pool.imap_unordered(procesar_imagen, argumentos, chunksize=50)
            imagenes_validas = [img for img in imagenes if img is not None]
        
        array_lote = np.array(imagenes_validas)
        np.savez_compressed(f'data/processed/lote_{contador}.npz', array_lote)
        print(f"  ✓ Lote guardado: {len(imagenes_validas)} imágenes")
        
        contador += 1
    
    print(f"\n✓ {contador} lotes guardados (comprimidos)")