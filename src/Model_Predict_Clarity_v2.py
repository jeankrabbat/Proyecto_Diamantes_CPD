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


# PARTE 4. FUNCIONES DE ENTRENAMIENTO
# -----------------------------------------------------------------------------
'''
 Infraestructura única y reutilizable para cualquier arquitectura de
 torchvision (ResNet50, EfficientNet-B0, ConvNeXt Tiny, etc.)

 Criterios usados:
   - Scheduler (ReduceLROnPlateau) y Early Stopping -> val_loss
   - Guardado del mejor checkpoint por modelo        -> val macro-F1

 Returns:
    El modelo con mayor F1-Score
'''
 
class FocalLoss(torch.nn.Module):
    """
    CrossEntropy ponderada con término focal para dar más importancia a los
    ejemplos difíciles y reducir el peso de los que el modelo ya clasifica
    correctamente. Con gamma=0 equivale a CrossEntropy ponderada
    """

    def __init__(self, weight: torch.Tensor = None, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits, targets, weight=self.weight,
            label_smoothing=self.label_smoothing, reduction="none",
        )
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


def apply_manual_weight_boost(class_weights: torch.Tensor, class_names, boosts: dict) -> torch.Tensor:
    """
    Multiplica el peso 'balanced' ya calculado por un factor extra en
    clases especificas
    """
    boosted = class_weights.clone()
    for name, factor in boosts.items():
        idx = class_names.index(name)
        boosted[idx] = boosted[idx] * factor
    return boosted


def train_epoch(model, loader, criterion, optimizer, scaler, device, use_amp: bool):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
 
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
 
        optimizer.zero_grad(set_to_none=True)
 
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            outputs = model(imgs)
            loss = criterion(outputs, labels)
 
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
 
        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.append(preds.detach().cpu())
        all_labels.append(labels.detach().cpu())
 
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
 
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
 
    return epoch_loss, epoch_acc
 
 
@torch.no_grad()
def validate_epoch(model, loader, criterion, device, use_amp: bool):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
 
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
 
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            outputs = model(imgs)
            loss = criterion(outputs, labels)
 
        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.append(preds.detach().cpu())
        all_labels.append(labels.detach().cpu())
 
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
 
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
 
    return epoch_loss, epoch_acc, epoch_f1_macro, all_preds, all_labels
 
 
def fit(model, train_loader, val_loader, cfg: Config, model_name: str,
        class_weights: torch.Tensor = None, device=DEVICE):
    '''
    Entrena el modelo y registra las métricas de entrenamiento y validación.

    Returns:
        history: Historial de métricas por época.
        best_val_f1: Mejor F1 macro obtenido en validación.
    '''
    model = model.to(device)
 
    weight_arg = class_weights.to(device) if (cfg.use_class_weight_loss and class_weights is not None) else None
    criterion = FocalLoss(weight=weight_arg, gamma=cfg.focal_gamma, label_smoothing=cfg.label_smoothing)
 
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
 
    scaler = torch.amp.GradScaler(enabled=cfg.use_mixed_precision)
 
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_f1_macro": [],
        "lr": [], "epoch_time": [],
    }
 
    best_val_f1 = -1.0
    best_val_loss_checkpoint = float("inf")
    best_val_loss_for_stopping = float("inf")
    epochs_no_improve = 0
    best_state_dict = None
 
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
 
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, cfg.use_mixed_precision
        )
        val_loss, val_acc, val_f1_macro, _, _ = validate_epoch(
            model, val_loader, criterion, device, cfg.use_mixed_precision
        )
 
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - t0
 
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1_macro"].append(val_f1_macro)
        history["lr"].append(current_lr)
        history["epoch_time"].append(epoch_time)
 
        print(
            f"[{model_name}] Epoch {epoch:03d}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1_macro={val_f1_macro:.4f} | "
            f"lr={current_lr:.2e} | {epoch_time:.1f}s"
        )
 
        # Guarda en memoria el mejor modelo según la pérdida de validación
        # El modelo final se guarda en 'models' al finalizar el entrenamiento
        if val_loss < best_val_loss_checkpoint:
            best_val_loss_checkpoint = val_loss
            best_val_f1 = val_f1_macro
            best_state_dict = copy.deepcopy(model.state_dict())
            print(f"  -> Mejor epoch hasta ahora (val_loss={best_val_loss_checkpoint:.4f}), guardado en memoria.")
 
        # --- Early stopping por val_loss ---
        if val_loss < best_val_loss_for_stopping - 1e-4:
            best_val_loss_for_stopping = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.early_stopping_patience:
                print(f"  Early stopping en epoch {epoch} (sin mejora de val_loss por {cfg.early_stopping_patience} epochs).")
                break
 
    # Restaurar los pesos del mejor checkpoint (por val_loss) antes de devolver el modelo
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
 
    return model, history, best_val_f1
 
 
@torch.no_grad()
def predict(model, loader, device=DEVICE, use_amp: bool = True):
    '''
    Genera predicciones para un conjunto de datos.

    Retorna:
        y_true: Etiquetas reales.
        y_pred: Etiquetas predichas.
        y_probs: Probabilidades de cada clase.
    '''
    model = model.to(device)
    model.eval()
 
    all_probs, all_preds, all_labels = [], [], []
 
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
 
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
 
        preds = probs.argmax(dim=1)
        all_probs.append(probs.detach().cpu())
        all_preds.append(preds.detach().cpu())
        all_labels.append(labels)
 
    y_probs = torch.cat(all_probs).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()
 
    return y_true, y_pred, y_probs
 
 
def evaluate_predictions(y_true, y_pred, class_names, average: str = "macro"):
    '''
    Calcula las métricas de evaluación del modelo, incluyendo Accuracy,
    Precision, Recall, F1, matriz de confusión y Accuracy Within 1
    '''
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average=average, zero_division=0),
        "accuracy_within_1": accuracy_within_k(y_true, y_pred, class_names, k=1),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(class_names)))),
    }
    return metrics
 
 
# PARTE 5-7. ARQUITECTURAS (ResNet50, EfficientNet-B0, ConvNeXt Tiny)
# --------------------------------------------------------------------

'''
Modelos preentrenados en ImageNet. Solo se reemplaza la capa de salida
para adaptarla a las 7 clases del problema.
'''
 
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
 
 
def run_experiment(build_fn, model_name: str, train_loader, val_loader, test_loader,
                    cfg: Config, class_weights: torch.Tensor):
    # Entrena una arquitectura y evalúa su rendimiento en el conjunto de prueba.
    
    print("\n" + "=" * 60)
    print(f"ENTRENANDO: {model_name}")
    print("=" * 60)
 
    model = build_fn(len(cfg.class_names))
 
    t0 = time.time()
    model, history, best_val_f1 = fit(
        model, train_loader, val_loader, cfg, model_name=model_name, class_weights=class_weights
    )
    train_time_sec = time.time() - t0
 
    print(f"\n>>> Evaluando {model_name} en test...")
    y_true, y_pred, y_probs = predict(model, test_loader)
    test_metrics = evaluate_predictions(y_true, y_pred, cfg.class_names)
 
    print(
        f"[{model_name}] TEST -> accuracy={test_metrics['accuracy']:.4f} "
        f"precision={test_metrics['precision_macro']:.4f} "
        f"recall={test_metrics['recall_macro']:.4f} "
        f"f1={test_metrics['f1_macro']:.4f} | "
        f"tiempo_entrenamiento={train_time_sec/60:.1f} min"
    )
 
    '''
    Guarda el mejor modelo ejecutado actualmente en memoria para
    seleccionar el modelo final.
    '''
    state_dict_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
 
    del model
    torch.cuda.empty_cache()
 
    return {
        "model_name": model_name,
        "build_fn": build_fn,
        "state_dict": state_dict_cpu,
        "history": history,
        "best_val_f1": best_val_f1,
        "test_metrics": test_metrics,
        "train_time_sec": train_time_sec,
        "y_true": y_true,
        "y_pred": y_pred,
    }
 
 
# PARTE 8. COMPARACION
# ------------------------------------------------------------
 
def build_comparison_table(results: list) -> pd.DataFrame:
    rows = []
    for r in results:
        m = r["test_metrics"]
        rows.append({
            "Modelo": r["model_name"],
            "Accuracy": round(m["accuracy"], 4),
            "Precision": round(m["precision_macro"], 4),
            "Recall": round(m["recall_macro"], 4),
            "F1": round(m["f1_macro"], 4),
            "Accuracy ±1": round(m["accuracy_within_1"], 4),
            "Tiempo (min)": round(r["train_time_sec"] / 60, 1),
        })
    df = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
    return df
 
 
# PARTE 9. VISUALIZACION
# -----------------------------------------------------------------------------

def _ordinal_permutation(class_names):
    # Indices para reordenar filas/columnas de alfabetico -> ordinal.
    return [class_names.index(c) for c in ORDINAL_ORDER]
 
 
def plot_loss_curves(results: list, save_path: Path):
    plt.figure(figsize=(8, 5))
    for r in results:
        h = r["history"]
        plt.plot(h["train_loss"], label=f"{r['model_name']} - train")
        plt.plot(h["val_loss"], linestyle="--", label=f"{r['model_name']} - val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Curvas de perdida")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
 
 
def plot_accuracy_curves(results: list, save_path: Path):
    plt.figure(figsize=(8, 5))
    for r in results:
        h = r["history"]
        plt.plot(h["train_acc"], label=f"{r['model_name']} - train")
        plt.plot(h["val_acc"], linestyle="--", label=f"{r['model_name']} - val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Curvas de accuracy")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
 
 
def plot_confusion_matrix(y_true, y_pred, class_names, model_name: str, save_path: Path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    perm = _ordinal_permutation(list(class_names))
    cm_ordered = cm[np.ix_(perm, perm)]
 
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_ordered, cmap="Blues")
    ax.set_xticks(range(len(ORDINAL_ORDER)))
    ax.set_yticks(range(len(ORDINAL_ORDER)))
    ax.set_xticklabels(ORDINAL_ORDER, rotation=45)
    ax.set_yticklabels(ORDINAL_ORDER)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusion - {model_name}")
 
    for i in range(cm_ordered.shape[0]):
        for j in range(cm_ordered.shape[1]):
            ax.text(j, i, cm_ordered[i, j], ha="center", va="center",
                     color="white" if cm_ordered[i, j] > cm_ordered.max() / 2 else "black", fontsize=8)
 
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
 
 
def print_classification_report(y_true, y_pred, class_names, model_name: str) -> str:
    report_text = classification_report(
        y_true, y_pred, labels=list(range(len(class_names))),
        target_names=class_names, zero_division=0,
    )
    print(f"\nReporte de clasificacion - {model_name}")
    print(report_text)
    return report_text
 
 
def plot_model_comparison(df: pd.DataFrame, save_path: Path):
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(df))
    width = 0.2
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1"]
    for i, metric in enumerate(metrics_to_plot):
        ax.bar(x + i * width, df[metric], width=width, label=metric)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(df["Modelo"])
    ax.set_ylabel("Score")
    ax.set_title("Comparacion entre modelos (test set)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
 
 
def save_epoch_history_csv(results: list):
    '''
    Guarda un CSV en 'clarity-model-metrics' con las métricas de entrenamiento
    y validación de cada época
    '''
    for r in results:
        h = r["history"]
        n_epochs = len(h["train_loss"])
        df = pd.DataFrame({
            "epoch": range(1, n_epochs + 1),
            "train_loss": h["train_loss"],
            "train_acc": h["train_acc"],
            "val_loss": h["val_loss"],
            "val_acc": h["val_acc"],
            "val_f1_macro": h["val_f1_macro"],
            "lr": h["lr"],
            "epoch_time_sec": h["epoch_time"],
        })
        out_path = IMAGES_DIR / f"epoch_history_{r['model_name']}.csv"
        df.to_csv(out_path, index=False)
        print(f"Historia por epoch guardada: {out_path}")


def generate_text_report(results: list, comparison_df: pd.DataFrame, best: dict,
                          classification_reports: dict, cfg: Config):
    # Genera un reporte con los resultados de cada modelo y el modelo final seleccionado
    
    lines = []
    lines.append("=" * 70)
    lines.append("REPORTE DE ENTRENAMIENTO - CLASIFICADOR DE CLARIDAD DE DIAMANTES")
    lines.append("=" * 70)
    lines.append(f"\nResolucion de imagen: {cfg.img_size}px")
    lines.append(f"Batch size: {cfg.batch_size}")
    lines.append(f"Loss: FocalLoss (gamma={cfg.focal_gamma}) + class_weight balanceado + boost manual {cfg.manual_weight_boost}")
    lines.append(f"Clases: {', '.join(cfg.class_names)}")

    lines.append("\n" + "-" * 70)
    lines.append("TABLA COMPARATIVA (test set)")
    lines.append("-" * 70)
    lines.append(comparison_df.to_string(index=False))

    for r in results:
        lines.append("\n" + "-" * 70)
        lines.append(f"REPORTE DE CLASIFICACION - {r['model_name']}")
        lines.append("-" * 70)
        lines.append(classification_reports[r["model_name"]])
        lines.append(f"Tiempo de entrenamiento: {r['train_time_sec']/60:.1f} min")
        n_epochs_run = len(r["history"]["train_loss"])
        lines.append(f"Epochs corridos (antes de early stopping o limite): {n_epochs_run}")

    lines.append("\n" + "=" * 70)
    lines.append("MODELO FINAL SELECCIONADO")
    lines.append("=" * 70)
    lines.append(f"Modelo: {best['model_name']}")
    lines.append(f"Criterio: F1-macro (test), desempate por Accuracy")
    lines.append(f"F1-macro (test): {best['test_metrics']['f1_macro']:.4f}")
    lines.append(f"Accuracy (test): {best['test_metrics']['accuracy']:.4f}")
    lines.append(f"Accuracy ±1 (test): {best['test_metrics']['accuracy_within_1']:.4f}")
    best_display_name = ALGO_DISPLAY_NAMES.get(best["model_name"], best["model_name"])
    best_filename = f"{best_display_name}-Clarity-Model.pt"
    lines.append(f"\nArtefactos de INFERENCIA (en models/): {best_filename}, label_encoder.pkl")
    lines.append(f"Artefactos de EVALUACION (en {IMAGES_DIR.name}/): graficos, CSVs de historia, este reporte")

    report_text = "\n".join(lines)
    report_path = IMAGES_DIR / "reporte_entrenamiento.txt"
    with open(report_path, "w") as f:
        f.write(report_text)

    print(f"\nReporte consolidado guardado en: {report_path}")
    return report_text


def generate_all_visualizations(results: list, comparison_df: pd.DataFrame, cfg: Config):
    plot_loss_curves(results, IMAGES_DIR / "loss_curves.png")
    plot_accuracy_curves(results, IMAGES_DIR / "accuracy_curves.png")
    plot_model_comparison(comparison_df, IMAGES_DIR / "model_comparison.png")

    classification_reports = {}
    for r in results:
        plot_confusion_matrix(
            r["y_true"], r["y_pred"], cfg.class_names, r["model_name"],
            IMAGES_DIR / f"confusion_matrix_{r['model_name']}.png",
        )
        report_text = print_classification_report(r["y_true"], r["y_pred"], cfg.class_names, r["model_name"])
        classification_reports[r["model_name"]] = report_text

    save_epoch_history_csv(results)

    print(f"\nGraficos y CSVs guardados en: {IMAGES_DIR}")
    return classification_reports


# PARTE 10. OBTENER MODELO FINAL
# ----------------------------------------------------------------------------

ALGO_DISPLAY_NAMES = {
    "resnet50": "ResNet50",
    "efficientnet_b0": "EfficientNet-B0",
    "convnext_tiny": "ConvNeXt",
}


def select_and_save_best_model(results: list, cfg: Config):
    '''
    Selecciona el mejor modelo según el F1 macro y, en caso de empate, Accuracy.
    Guarda únicamente el modelo final.
    '''
    best = max(
        results,
        key=lambda r: (round(r["test_metrics"]["f1_macro"], 6), round(r["test_metrics"]["accuracy"], 6)),
    )

    display_name = ALGO_DISPLAY_NAMES.get(best["model_name"], best["model_name"])
    best_model_path = MODELS_DIR / f"{display_name}-Clarity-Model.pt"

    torch.save({
        "model_name": best["model_name"],
        "model_state_dict": best["state_dict"],
        "class_names": cfg.class_names,
        "img_size": cfg.img_size,
        "test_metrics": {k: v for k, v in best["test_metrics"].items() if k != "confusion_matrix"},
    }, best_model_path)

    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder().fit(sorted(cfg.class_names))
    with open(MODELS_DIR / "label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    # Artefactos de analisis/evaluacion (NO se usan para inferencia)
    histories = {r["model_name"]: r["history"] for r in results}
    with open(IMAGES_DIR / "training_history.pkl", "wb") as f:
        pickle.dump(histories, f)

    print("\n" + "=" * 60)
    print(f"MODELO FINAL SELECCIONADO: {display_name} ({best['model_name']})")
    print(f"F1-macro (test): {best['test_metrics']['f1_macro']:.4f}")
    print(f"Accuracy (test): {best['test_metrics']['accuracy']:.4f}")
    print(f"Unico modelo serializado en: {best_model_path}")
    print("=" * 60)
    print("\nArtefactos para INFERENCIA (necesarios): "
          f"{best_model_path.name}, label_encoder.pkl")
    print("Artefactos de ANALISIS (no se usan para predecir): training_history.pkl,")
    print("comparison_table.csv, loss_curves.png, accuracy_curves.png, model_comparison.png,")
    print("confusion_matrix_*.png")
 
    return best
 
 
# INICIALIZACIÓN DEL PIPELINE DE DATOS
# --------------------------------------
if __name__ == "__main__":
    print("\n>>> Cargando metadata...")
    meta_df = load_metadata(METADATA_CSV_PATH)
    clarity_label_h5 = load_labels_from_h5(H5_PATH)
    sanity_checks(meta_df, clarity_label_h5)
 
    print("\n>>> Generando splits...")
    train_idx, val_idx, test_idx = make_splits(clarity_label_h5, CFG)
 
    print("\n>>> Construyendo DataLoaders...")
    train_loader, val_loader, test_loader = build_dataloaders(
        H5_PATH, clarity_label_h5, train_idx, val_idx, test_idx, CFG
    )
 
    print("\n>>> Probando un batch de entrenamiento (validacion de I/O + transforms)...")
    imgs, lbls = next(iter(train_loader))
    print("Shape de batch de imagenes:", imgs.shape)
    print("Shape de batch de labels:", lbls.shape)
    print("Rango de valores de pixel (normalizado):", imgs.min().item(), imgs.max().item())
 
    print("\n>>> Pesos de clase (balanced, base para Focal Loss):")
    class_weights = compute_loss_class_weights(clarity_label_h5, train_idx, CFG)
    for name, w in zip(CFG.class_names, class_weights.tolist()):
        print(f"  {name}: {w:.3f}")

    class_weights = apply_manual_weight_boost(class_weights, list(CFG.class_names), CFG.manual_weight_boost)
    print("\n>>> Pesos de clase FINALES (con boost manual aplicado):")
    for name, w in zip(CFG.class_names, class_weights.tolist()):
        print(f"  {name}: {w:.3f}")

    print("\nPartes 1-4 verificadas. Iniciando entrenamiento de las 3 arquitecturas...")
 
    architectures = [
        (build_resnet50, "resnet50"),
        (build_efficientnet_b0, "efficientnet_b0"),
        (build_convnext_tiny, "convnext_tiny"),
    ]
 
    results = []
    for build_fn, model_name in architectures:
        result = run_experiment(
            build_fn, model_name, train_loader, val_loader, test_loader,
            CFG, class_weights,
        )
        results.append(result)
 
    comparison_df = build_comparison_table(results)
    print("\n" + "=" * 60)
    print("TABLA COMPARATIVA (test set)")
    print("=" * 60)
    print(comparison_df.to_string(index=False))
    comparison_df.to_csv(IMAGES_DIR / "comparison_table.csv", index=False)
    classification_reports = generate_all_visualizations(results, comparison_df, CFG)
    best_model_info = select_and_save_best_model(results, CFG)
    generate_text_report(results, comparison_df, best_model_info, classification_reports, CFG)
    print("\nPipeline completo ejecutado correctamente.")