"""
Model_Predict_Clarity.py
----------------------------
Clasificador de claridad.
Clases Evaluadas: IF, VVS1, VVS2, VS1, VS2, SI1, SI2
"""

# PARTE 1. CONFIGURACION
# -----------------------------------------------------------------------------

import os
import io
import random
import warnings
from pathlib import Path
from dataclasses import dataclass, field
 
import numpy as np
import pandas as pd
import h5py
from PIL import Image
 
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
 
import time
import copy
import pickle
 
import matplotlib
matplotlib.use("Agg")  # backend sin pantalla, para entornos HPC sin display
import matplotlib.pyplot as plt
 
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)
 
warnings.filterwarnings("ignore", category=UserWarning)

# 1.1 Optimiza el rendimiento cuando todas las imágenes tienen el mismo tamaño
if not torch.cuda.is_available():
    raise RuntimeError(
        "Este proyecto requiere una GPU con CUDA para entrenar los modelos."
    )
DEVICE = torch.device("cuda")
torch.backends.cudnn.benchmark = True

# 1.2 Semillas aleatorias (reproducibilidad)
SEED = 42
def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(SEED)

# 1.3 Hiperparametros
@dataclass
class Config:
    # Datos y split
    img_size: int = 512 # px por imagen
    val_size: float = 0.15
    test_size: float = 0.15
    seed: int = SEED

    # Entrenamiento
    batch_size: int = 48 # Cantidad de epoca
    num_workers: int = min(16, os.cpu_count() or 4)
    epochs: int = 50
    lr: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    early_stopping_patience: int = 10
    use_mixed_precision: bool = True
    use_weighted_sampler: bool = False
    use_class_weight_loss: bool = True
    focal_gamma: float = 2.0
    manual_weight_boost: dict = field(default_factory=lambda: {"IF": 1.5, "VVS1": 1.2})

    # Orden de las clases según clarity_label almacenado en el HDF5.
    class_names: tuple = ("IF", "SI1", "SI2", "VS1", "VS2", "VVS1", "VVS2")

CFG = Config()

# 1.4 Rutas

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
H5_PATH = DATA_DIR / "diamonds_originales.h5"
METADATA_CSV_PATH = DATA_DIR / "metadata_procesada_hdf5.csv"
MODELS_DIR = PROJECT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
IMAGES_DIR = PROJECT_DIR / "images" / "clarity-model-metrics"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Orden ordinal real de claridad (mejor -> peor) segun GIA
ORDINAL_ORDER = ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2"]


def build_ordinal_rank_map(class_names, ordinal_order=ORDINAL_ORDER) -> np.ndarray:
    """Mapea el indice de clase (orden alfabetico, igual al encoding del H5)
    a su posicion real en la escala de claridad (0=IF ... 6=SI2)."""
    return np.array([ordinal_order.index(c) for c in class_names])


def accuracy_within_k(y_true, y_pred, class_names, k: int = 1) -> float:
    """Fraccion de predicciones que quedan a lo sumo k grados de distancia
    del valor real en la escala ORDINAL (no en el indice alfabetico crudo).
    """
    rank_map = build_ordinal_rank_map(class_names)
    true_ranks = rank_map[np.asarray(y_true)]
    pred_ranks = rank_map[np.asarray(y_pred)]
    return float(np.mean(np.abs(true_ranks - pred_ranks) <= k))

# PARTE 2. CARGA DE DATOS
# -----------------------------------------------------------------------------

def load_metadata(csv_path: Path) -> pd.DataFrame:
    """Carga la metadata sincronizada con el HDF5."""
    return pd.read_csv(csv_path)

def load_labels_from_h5(h5_path: Path) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        return f["clarity_label"][:]


def sanity_checks(meta_df: pd.DataFrame, clarity_label_h5: np.ndarray):
    """Realiza validaciones básicas antes de construir los splits."""

    print("=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)

    # Verificar que la metadata y las etiquetas estén sincronizadas
    assert len(meta_df) == len(clarity_label_h5), (
        f"Descalce de filas: metadata tiene {len(meta_df)} registros, "
        f"clarity_label tiene {len(clarity_label_h5)}."
    )

    # Verificar que todas las etiquetas sean válidas
    assert set(np.unique(clarity_label_h5)).issubset(range(len(CFG.class_names))), (
        "Se encontraron etiquetas de claridad fuera del rango esperado."
    )

    print("\nDistribución de clases (clarity):")
    print(meta_df["clarity"].value_counts())

    print("=" * 60)

def make_splits(labels: np.ndarray, cfg: Config):
    """Genera índices de entrenamiento, validación y prueba estratificados."""

    idx = np.arange(len(labels))

    train_idx, rest_idx = train_test_split(
        idx,
        test_size=cfg.val_size + cfg.test_size,
        stratify=labels,
        random_state=cfg.seed,
    )

    val_idx, test_idx = train_test_split(
        rest_idx,
        test_size=cfg.test_size / (cfg.val_size + cfg.test_size),
        stratify=labels[rest_idx],
        random_state=cfg.seed,
    )

    print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    return train_idx, val_idx, test_idx


# PARTE 3. DATASET
# -------------------------------------------------------------------------------

class DiamondDataset(Dataset):
    """
    Dataset personalizado para cargar imágenes de diamantes desde un archivo
    HDF5 junto con sus etiquetas de claridad.
    """

    def __init__(self, h5_path: Path, indices: np.ndarray, labels: np.ndarray, transform=None):
        self.h5_path = str(h5_path)
        self.indices = indices
        self.labels = labels
        self.transform = transform
        self._h5 = None

    def _ensure_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        self._ensure_open()
        real_idx = self.indices[i]

        raw_bytes = self._h5["image_bytes"][real_idx]
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        label = int(self.labels[real_idx])
        return img, label

    def __getstate__(self):
        # Evita serializar la conexión al archivo HDF5 al crear procesos worker.
        state = self.__dict__.copy()
        state["_h5"] = None
        return state


# 3.1 Transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def build_transforms(cfg: Config):
    train_tf = T.Compose([
        T.RandomResizedCrop(cfg.img_size, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_tf = T.Compose([
        T.Resize((cfg.img_size, cfg.img_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_tf, eval_tf


# 3.2 Construccion de Datasets y DataLoaders

def build_dataloaders(h5_path: Path, labels: np.ndarray, train_idx, val_idx, test_idx, cfg: Config):
    train_tf, eval_tf = build_transforms(cfg)

    train_ds = DiamondDataset(h5_path, train_idx, labels, transform=train_tf)
    val_ds = DiamondDataset(h5_path, val_idx, labels, transform=eval_tf)
    test_ds = DiamondDataset(h5_path, test_idx, labels, transform=eval_tf)

    # Sampler balanceado. Alternativa al uso de class_weight en la función de pérdida.
    train_sampler = None
    shuffle_train = True
    if cfg.use_weighted_sampler:
        train_labels = labels[train_idx]
        class_sample_count = np.bincount(train_labels, minlength=len(cfg.class_names))
        weight_per_class = 1.0 / np.clip(class_sample_count, 1, None)
        sample_weights = weight_per_class[train_labels]
        train_sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )
        shuffle_train = False

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )

    return train_loader, val_loader, test_loader


def compute_loss_class_weights(labels: np.ndarray, train_idx: np.ndarray, cfg: Config) -> torch.Tensor:
    train_labels = labels[train_idx]
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(cfg.class_names)),
        y=train_labels,
    )
    return torch.tensor(weights, dtype=torch.float32)

def build_resnet50(num_classes: int):
    from torchvision.models import resnet50, ResNet50_Weights
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model
 
 
def build_efficientnet_b0(num_classes: int):
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    return model
 
 
def build_convnext_tiny(num_classes: int):
    from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    # classifier = Sequential(LayerNorm2d, Flatten, Linear) -> reemplazar el Linear final
    in_features = model.classifier[2].in_features
    model.classifier[2] = torch.nn.Linear(in_features, num_classes)
    return model
 
 
'''
Calibrate_IF_Threshold.py
--------------------------
Carga el modelo de clasificacion ya entrenado (models/*-Clarity-Model.pt) y
ajusta el umbral de decision para la clase IF, priorizando recall a costa de
precision -- de forma explicita y medible, no como efecto secundario de la loss.

No reentrena nada: solo corre inferencia sobre val (para elegir el umbral) y
test (para reportar el resultado final con ese umbral).
'''

import glob

from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix

# Umbral objetivo de recall para IF. Bajar este numero = mas agresivo,
# mas falsos positivos aceptados a cambio de no dejar pasar IFs reales.
TARGET_RECALL_IF = 0.90


def find_latest_checkpoint(models_dir):
    candidates = sorted(glob.glob(str(models_dir / "*-Clarity-Model.pt")))
    if not candidates:
        raise FileNotFoundError(f"No se encontro ningun *-Clarity-Model.pt en {models_dir}")
    if len(candidates) > 1:
        print(f"Aviso: hay varios checkpoints en {models_dir}, se usa el mas reciente: {candidates[-1]}")
    return candidates[-1]


BUILDERS = {
    "resnet50": build_resnet50,
    "efficientnet_b0": build_efficientnet_b0,
    "convnext_tiny": build_convnext_tiny,
}


def load_trained_model(checkpoint_path, device=DEVICE):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name = ckpt["model_name"]
    class_names = ckpt["class_names"]
    img_size = ckpt["img_size"]

    model = BUILDERS[model_name](len(class_names))
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    return model, model_name, class_names, img_size


@torch.no_grad()
def get_probs(model, loader, device=DEVICE):
    all_probs, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=True):
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)
        all_probs.append(probs.cpu())
        all_labels.append(labels)
    return torch.cat(all_probs).numpy(), torch.cat(all_labels).numpy()


def find_threshold_for_recall(y_true_bin, probs_if, target_recall):
    '''
    Busca el umbral mas alto que aun logra el recall objetivo (mayor umbral =
    mejor precision posible para ese recall).
    '''
    precisions, recalls, thresholds = precision_recall_curve(y_true_bin, probs_if)
    best_threshold, best_precision = 0.0, 0.0
    for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds):
        if r >= target_recall and t > best_threshold:
            best_threshold, best_precision = t, p
    return best_threshold, precisions, recalls, thresholds


def apply_if_threshold(probs, class_names, if_threshold):
    '''
    Regla de decision: predice IF si prob(IF) >= if_threshold; si no,
    predice la clase de mayor probabilidad EXCLUYENDO IF.
    '''
    if_idx = class_names.index("IF")
    preds = np.zeros(len(probs), dtype=int)

    for i, row in enumerate(probs):
        if row[if_idx] >= if_threshold:
            preds[i] = if_idx
        else:
            row_no_if = row.copy()
            row_no_if[if_idx] = -1
            preds[i] = row_no_if.argmax()

    return preds


def plot_pr_curve_if(precisions, recalls, thresholds, chosen_threshold, save_path):
    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, precisions[:-1], label="Precision")
    plt.plot(thresholds, recalls[:-1], label="Recall")
    plt.axvline(chosen_threshold, color="gray", linestyle="--",
                label=f"Umbral elegido = {chosen_threshold:.3f}")
    plt.xlabel("Umbral de probabilidad para IF")
    plt.ylabel("Score")
    plt.title("Calibracion de umbral - clase IF")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    checkpoint_path = find_latest_checkpoint(MODELS_DIR)
    print(f"\nCargando checkpoint: {checkpoint_path}")
    model, model_name, class_names, img_size = load_trained_model(checkpoint_path)
    print(f"Arquitectura: {model_name} | Clases: {class_names} | img_size: {img_size}")

    meta_df = load_metadata(METADATA_CSV_PATH)
    clarity_label_h5 = load_labels_from_h5(H5_PATH)
    train_idx, val_idx, test_idx = make_splits(clarity_label_h5, CFG)

    cfg_infer = Config(img_size=img_size, class_names=class_names)
    _, val_loader, test_loader = build_dataloaders(
        H5_PATH, clarity_label_h5, train_idx, val_idx, test_idx, cfg_infer
    )

    if_idx = class_names.index("IF")

    print("\n>>> Corriendo inferencia sobre VAL (para elegir el umbral)...")
    val_probs, val_labels = get_probs(model, val_loader)
    val_true_bin = (val_labels == if_idx).astype(int)

    threshold, precisions, recalls, thresholds = find_threshold_for_recall(
        val_true_bin, val_probs[:, if_idx], TARGET_RECALL_IF
    )
    print(f"Umbral elegido para recall >= {TARGET_RECALL_IF:.0%}: {threshold:.4f}")

    plot_pr_curve_if(precisions, recalls, thresholds, threshold,
                      IMAGES_DIR / "if_threshold_calibration.png")

    print("\n>>> Aplicando umbral sobre TEST (reporte final, sin haber tocado test antes)...")
    test_probs, test_labels = get_probs(model, test_loader)

    print("\n--- Prediccion original (argmax, sin calibrar) ---")
    y_pred_default = test_probs.argmax(axis=1)
    print(classification_report(test_labels, y_pred_default, target_names=class_names, zero_division=0))

    print(f"\n--- Prediccion calibrada (umbral IF = {threshold:.4f}) ---")
    y_pred_calibrated = apply_if_threshold(test_probs, list(class_names), threshold)
    report_calibrated = classification_report(test_labels, y_pred_calibrated, target_names=class_names, zero_division=0)
    print(report_calibrated)

    cm_calibrated = confusion_matrix(test_labels, y_pred_calibrated, labels=list(range(len(class_names))))

    with open(IMAGES_DIR / "if_calibration_report.txt", "w") as f:
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Umbral IF elegido (recall objetivo {TARGET_RECALL_IF:.0%}): {threshold:.4f}\n\n")
        f.write("Reporte de clasificacion (calibrado):\n")
        f.write(report_calibrated)
        f.write("\nMatriz de confusion (calibrado, orden alfabetico de clases):\n")
        f.write(str(cm_calibrated))

    print(f"\nReporte guardado en: {IMAGES_DIR / 'if_calibration_report.txt'}")

    # --- Persistir el umbral en el .pt: sin esto, la app no sabe que debe
    # calibrar la decision de IF; solo tendria los pesos del modelo. ---
    ckpt_full = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_full["if_threshold"] = float(threshold)
    ckpt_full["if_target_recall"] = TARGET_RECALL_IF
    ckpt_full["decision_rule"] = (
        "Si prob[IF] >= if_threshold -> predecir IF. "
        "Si no -> argmax de las probabilidades EXCLUYENDO IF."
    )
    torch.save(ckpt_full, checkpoint_path)
    print(f"\nUmbral guardado DENTRO de {checkpoint_path} (campo 'if_threshold' = {threshold:.4f})")
    print("El .pt ahora es autocontenido: pesos + regla de decision calibrada, listo para la app.")
    print(f"Curva precision/recall guardada en: {IMAGES_DIR / 'if_threshold_calibration.png'}")
