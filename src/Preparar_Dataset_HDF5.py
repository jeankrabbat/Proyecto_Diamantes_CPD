"""Crear un dataset HDF5 con imagenes originales validadas.
Este script usa la metadata filtrada generada por ETL.py:

Acciones principales:

1. Lee la metadata procesada con Polars.
2. Revisa cada imagen con OpenCV para detectar archivos corruptos o ilegibles.
3. Excluye esas imagenes del HDF5 y de la metadata sincronizada.
4. Guarda las imagenes originales como bytes, sin redimensionarlas.
5. Guarda labels codificados para shape, colour y clarity.

Salidas:

    data/processed/diamonds_originales.h5
    data/processed/metadata_procesada_hdf5.csv
    data/processed/imagenes_corruptas.csv
"""

from pathlib import Path
import json
import time

import cv2
import h5py
import numpy as np
import polars as pl


METADATA_RELATIVE_PATH = Path("data/processed/metadata_procesada.csv")
HDF5_RELATIVE_PATH = Path("data/processed/diamonds_originales.h5")
SYNC_METADATA_RELATIVE_PATH = Path("data/processed/metadata_procesada_hdf5.csv")
CORRUPT_METADATA_RELATIVE_PATH = Path("data/processed/imagenes_corruptas.csv")

IMAGE_COLUMN = "full_path"
LABEL_COLUMNS = ["shape", "colour", "clarity"]


def get_project_root() -> Path:
    """Return project root assuming this file is inside src/ or code/."""
    script_dir = Path(__file__).resolve().parent
    if script_dir.name in {"src", "code"}:
        return script_dir.parent
    return script_dir


def validate_image(path: str) -> tuple[bool, dict]:
    """Validate that an image can be read and decoded by OpenCV."""
    image_path = Path(path)

    if not image_path.exists():
        return False, {
            "path": path,
            "error": "archivo_no_existe",
            "height": None,
            "width": None,
            "channels": None,
        }

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return False, {
            "path": path,
            "error": "imagen_corrupta_o_ilegible",
            "height": None,
            "width": None,
            "channels": None,
        }

    if image.ndim == 2:
        height, width = image.shape
        channels = 1
    else:
        height, width, channels = image.shape

    return True, {
        "path": path,
        "error": "",
        "height": int(height),
        "width": int(width),
        "channels": int(channels),
    }


def encode_labels(metadata: pl.DataFrame, column: str):
    """Encode string labels into integer IDs and return the mapping."""
    categories = (
        metadata
        .select(pl.col(column).cast(pl.Utf8).drop_nulls().unique().sort())
        .to_series()
        .to_list()
    )
    mapping = {value: idx for idx, value in enumerate(categories)}
    encoded = np.array(
        [mapping[value] for value in metadata.get_column(column).cast(pl.Utf8).to_list()],
        dtype=np.int64,
    )
    return encoded, mapping


def write_original_image_bytes(h5_file: h5py.File, paths: list[str]):
    """Store original image files as variable-length uint8 byte arrays."""
    total = len(paths)
    variable_uint8 = h5py.vlen_dtype(np.dtype("uint8"))
    image_bytes = h5_file.create_dataset(
        "image_bytes",
        shape=(total,),
        dtype=variable_uint8,
    )

    for idx, path in enumerate(paths):
        if idx % 1000 == 0:
            print(f"Guardando imagen {idx:,}/{total:,}")

        image_bytes[idx] = np.fromfile(path, dtype=np.uint8)


def main():
    start_time = time.time()
    project_root = get_project_root()

    metadata_path = project_root / METADATA_RELATIVE_PATH
    output_path = project_root / HDF5_RELATIVE_PATH
    sync_metadata_path = project_root / SYNC_METADATA_RELATIVE_PATH
    corrupt_metadata_path = project_root / CORRUPT_METADATA_RELATIVE_PATH

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CREAR DATASET HDF5 CON IMAGENES ORIGINALES")
    print("=" * 60)
    print(f"Proyecto: {project_root}")
    print(f"Metadata entrada: {metadata_path}")
    print(f"HDF5 salida: {output_path}")

    metadata = pl.read_csv(
        metadata_path,
        infer_schema_length=0,
        ).with_row_index("row_id")

    missing_columns = [
        column for column in [IMAGE_COLUMN, *LABEL_COLUMNS]
        if column not in metadata.columns
    ]
    if missing_columns:
        raise ValueError(f"Columnas faltantes en metadata: {missing_columns}")

    paths = metadata.get_column(IMAGE_COLUMN).cast(pl.Utf8).to_list()
    total_input = len(paths)
    print(f"Registros en metadata procesada: {total_input:,}")

    valid_row_ids = []
    image_info = []
    corrupt_rows = []

    for row_id, path in enumerate(paths):
        if row_id % 1000 == 0:
            print(f"Validando imagen {row_id:,}/{total_input:,}")

        is_valid, info = validate_image(path)
        info["row_id"] = row_id

        if is_valid:
            valid_row_ids.append(row_id)
            image_info.append(info)
        else:
            corrupt_rows.append(info)

    valid_metadata = (
        metadata
        .filter(pl.col("row_id").is_in(valid_row_ids))
        .drop("row_id")
    )
    valid_paths = valid_metadata.get_column(IMAGE_COLUMN).cast(pl.Utf8).to_list()

    corrupt_metadata = pl.DataFrame(corrupt_rows) if corrupt_rows else pl.DataFrame(
        schema={
            "path": pl.Utf8,
            "error": pl.Utf8,
            "height": pl.Int64,
            "width": pl.Int64,
            "channels": pl.Int64,
            "row_id": pl.Int64,
        }
    )
    image_info_metadata = pl.DataFrame(image_info)

    valid_metadata.write_csv(sync_metadata_path)
    corrupt_metadata.write_csv(corrupt_metadata_path)

    print(f"Imagenes validas: {len(valid_paths):,}")
    print(f"Imagenes corruptas o ilegibles: {len(corrupt_rows):,}")
    print(f"Metadata sincronizada: {sync_metadata_path}")
    print(f"Reporte de corruptas: {corrupt_metadata_path}")

    encoded_labels = {}
    label_mappings = {}

    for column in LABEL_COLUMNS:
        encoded, mapping = encode_labels(valid_metadata, column)
        encoded_labels[column] = encoded
        label_mappings[column] = mapping
        print(f"{column}: {len(mapping)} clases")

    string_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(output_path, "w") as h5_file:
        h5_file.attrs["image_storage"] = "original_file_bytes"
        h5_file.attrs["image_decoding_note"] = (
            "Cada registro en image_bytes contiene los bytes originales del archivo. "
            "Para entrenar, decodificar con cv2.imdecode o PIL."
        )
        h5_file.attrs["source_metadata"] = str(metadata_path)
        h5_file.attrs["synced_metadata"] = str(sync_metadata_path)
        h5_file.attrs["corrupt_report"] = str(corrupt_metadata_path)
        h5_file.attrs["label_columns"] = json.dumps(LABEL_COLUMNS)
        h5_file.attrs["label_mappings"] = json.dumps(label_mappings, ensure_ascii=False)

        write_original_image_bytes(h5_file, valid_paths)

        for column in LABEL_COLUMNS:
            h5_file.create_dataset(
                f"{column}_label",
                data=encoded_labels[column],
                dtype=np.int64,
            )

        h5_file.create_dataset(
            "paths",
            data=np.array(valid_paths, dtype=object),
            dtype=string_dtype,
        )

        h5_file.create_dataset(
            "image_info_csv",
            data=image_info_metadata.write_csv(),
            dtype=string_dtype,
        )

        h5_file.create_dataset(
            "metadata_csv",
            data=valid_metadata.write_csv(),
            dtype=string_dtype,
        )

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("HDF5 CREADO")
    print("=" * 60)
    print(f"Archivo: {output_path}")
    print(f"Imagenes guardadas: {len(valid_paths):,}")
    print(f"Imagenes excluidas: {len(corrupt_rows):,}")
    print(f"Tiempo total: {elapsed / 60:.2f} minutos")


if __name__ == "__main__":
    main()
