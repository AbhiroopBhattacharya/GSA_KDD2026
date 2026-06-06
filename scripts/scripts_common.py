"""
Shared utilities for the JEPA OOD pipeline (pretrain → train → evaluate).

All data paths are relative to the project root and start with Data_composites/.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoModel, AutoTokenizer

from data_jepa_tit_cftr import CompositeImageTextDataset, split_data
from jepa_tit_condtransformer import MAEStyleJEPACompositeModel
from utils import summarize_best_d
from viz_geodesics_rev1 import (
    aggregate_mean,
    calc_weights_geodesic_softanchors,
    precompute_id_geometry,
)

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "Data_composites"

MASTER_CSV = DATA_ROOT / "Comp1_data.csv"
IMAGE_FOLDER = DATA_ROOT / "microstructure"
TEXT_FOLDER = DATA_ROOT / "text"

CHECKPOINT_DIR = PROJECT_ROOT / "Results" / "Checkpoint"
JEPA_CHECKPOINT = CHECKPOINT_DIR / "jepa_pretrain_best.pth"
GEODESIC_WEIGHTS_PATH = CHECKPOINT_DIR / "geodesic_weights.json"
NORM_STATS_PATH = CHECKPOINT_DIR / "normalization_stats.json"

TEXT_MODEL_NAME = "m3rg-iitd/matscibert"
DEFAULT_TARGETS = [
    "Young_modulus",
    "Bulk_modulus",
    "TC1",
    "TC2",
    "Tensile_strength",
]
RANDOM_SEED = 1234


def get_device():
    """Return cuda:0 when available, otherwise cpu."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_jepa_model(device=None):
    """
    Instantiate the multimodal JEPA backbone with MatSciBERT text conditioning.

    The context/target encoders and masked-column predictor are trained during
    unsupervised pretraining; the property head is added later in scripts_train.py.
    """
    device = device or get_device()
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    matscibert = AutoModel.from_pretrained(TEXT_MODEL_NAME)

    model = MAEStyleJEPACompositeModel(
        tabular_input_dim=7,
        combined_dim=64,
        enable_contrastive=False,
        use_masked_tabular_modeling=True,
        hidden_dim=64,
        num_masked_columns=3,
        tokenizer=tokenizer,
        matscibert=matscibert,
    )
    model.to(device)
    return model, tokenizer


def split_qh1_by_conditions(dataset, test_filter_type="quantile"):
    """
    Split Comp1 data into in-domain (ID) train and out-of-domain (OOD) test sets.

    OOD samples are drawn from extreme quantiles of f_gra; ID samples lie in the
    central quantile band. This mimics covariate shift in fiber grading.
    """
    df = dataset.data
    test_mask = pd.Series([False] * len(df))
    train_mask = pd.Series([False] * len(df))

    if test_filter_type != "quantile":
        raise ValueError(f"Unsupported test_filter_type: {test_filter_type}")

    var1_train_upper = df["f_gra"].quantile(0.7)
    var1_train_lower = df["f_gra"].quantile(0.3)
    var1_upper = df["f_gra"].quantile(0.9)
    var1_lower = df["f_gra"].quantile(0.1)

    test_mask = (df["f_gra"] > var1_upper) | (df["f_gra"] < var1_lower)
    train_mask = (df["f_gra"] > var1_train_lower) & (df["f_gra"] < var1_train_upper)

    print(f"OOD test samples: {test_mask.sum()} | ID train samples: {train_mask.sum()}")

    test_indices = df[test_mask].index.tolist()
    train_indices = df[train_mask].index.tolist()
    return Subset(dataset, train_indices), Subset(dataset, test_indices)


def generate_ood_qh1_data(batch_size=128, target_name="Young_modulus"):
    """
    Build train / val / test dataloaders for the QH1 OOD benchmark.

    Returns:
        train_dataloader, val_dataloader, test_dataloader
    """
    master_dataset = CompositeImageTextDataset(
        str(MASTER_CSV),
        str(IMAGE_FOLDER),
        str(TEXT_FOLDER),
        target_name,
        num_augmentations=5,
        mode="Comp1",
        add_gaussian_noise=False,
        noise_std=0.01,
    )

    id_dataset, ood_test_dataset = split_qh1_by_conditions(master_dataset)
    train_dataset, val_dataset = split_data(id_dataset, train_ratio=0.80, random_seed=RANDOM_SEED)

    print(
        f"Split sizes — train: {len(train_dataset)}, "
        f"val: {len(val_dataset)}, test (OOD): {len(ood_test_dataset)}"
    )

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    loader_kwargs = dict(batch_size=batch_size, num_workers=0, generator=generator)
    train_dataloader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_dataloader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_dataloader = DataLoader(ood_test_dataset, shuffle=False, **loader_kwargs)
    return train_dataloader, val_dataloader, test_dataloader


def get_properties_mean_std(dataloader, device):
    """Compute per-feature and target normalization statistics from the train loader."""
    properties_list, targets_list = [], []
    for batch in dataloader:
        properties_list.append(batch["properties"].float())
        targets_list.append(batch["target"].float())

    properties_arr = np.concatenate(properties_list)
    targets_arr = np.concatenate(targets_list)

    properties_mean = torch.tensor(properties_arr.mean(axis=0), dtype=torch.float32).to(device)
    properties_std = torch.clamp(
        torch.tensor(properties_arr.std(axis=0), dtype=torch.float32), min=1e-8
    ).to(device)
    targets_mean = float(targets_arr.mean())
    targets_std = float(max(targets_arr.std(), 1e-8))
    return properties_mean, properties_std, targets_mean, targets_std


def save_normalization_stats(target_name, properties_mean, properties_std, targets_mean, targets_std):
    """Persist normalization stats so evaluation can reuse the training distribution."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    all_stats = {}
    if NORM_STATS_PATH.exists():
        with open(NORM_STATS_PATH) as f:
            all_stats = json.load(f)

    all_stats[target_name] = {
        "properties_mean": properties_mean.detach().cpu().tolist(),
        "properties_std": properties_std.detach().cpu().tolist(),
        "targets_mean": targets_mean,
        "targets_std": targets_std,
    }
    with open(NORM_STATS_PATH, "w") as f:
        json.dump(all_stats, f, indent=2)


def load_normalization_stats(target_name, device):
    """Load saved normalization stats for a given target property."""
    with open(NORM_STATS_PATH) as f:
        stats = json.load(f)[target_name]
    properties_mean = torch.tensor(stats["properties_mean"], dtype=torch.float32, device=device)
    properties_std = torch.tensor(stats["properties_std"], dtype=torch.float32, device=device)
    return properties_mean, properties_std, stats["targets_mean"], stats["targets_std"]


def save_jepa_checkpoint(model, epoch, loss, path=None):
    """Save the best unsupervised JEPA checkpoint."""
    path = path or JEPA_CHECKPOINT
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"epoch": epoch, "model_state_dict": model.state_dict(), "loss": loss},
        path,
    )
    print(f"JEPA checkpoint saved: {path}")


def load_jepa_checkpoint(model, path=None, device=None):
    """Load a pretrained JEPA checkpoint into an existing model instance."""
    path = path or JEPA_CHECKPOINT
    device = device or get_device()
    if not path.exists():
        raise FileNotFoundError(f"JEPA checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", 0.0)
    print(f"Loaded JEPA checkpoint from epoch {epoch} (loss={loss:.6f})")
    return model, epoch, loss


def get_tabular_embeddings(model, images, properties, text):
    """Extract frozen tabular embeddings (no masking) for downstream prediction."""
    with torch.no_grad():
        model.eval()
        output = model(images, properties, text, apply_masking=False)
        return output["original_embeddings"]["tabular"]


def fetch_embeddings(model, dataloader, properties_mean, properties_std):
    """Run the full dataloader through the encoder and collect embeddings + sample IDs."""
    embeddings_list, sample_ids = [], []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(model.device)
            text = batch["text"]
            properties = batch["properties"].to(model.device)
            props_norm = (properties - properties_mean) / properties_std

            sample_id = batch["id"]
            if torch.is_tensor(sample_id):
                sample_ids.extend(sample_id.detach().cpu().tolist())
            else:
                sample_ids.extend(sample_id)

            output = model(images, props_norm, text, apply_masking=False)
            z = output["original_embeddings"]["tabular"].detach().cpu().numpy()
            embeddings_list.append(z)

    return np.concatenate(embeddings_list, axis=0), sample_ids


def calculate_geodesic_weights(model, train_dataloader, test_dataloader, properties_mean, properties_std):
    """
    Compute per-sample importance weights for geodesic-weighted fine-tuning.

    Steps:
      1. Embed ID train and OOD test samples in the JEPA latent space.
      2. Build a kNN graph on ID embeddings and select anchor nodes.
      3. Measure geodesic distances from OOD points to ID anchors.
      4. Propagate distances back to ID training samples as weights.
    """
    z_id, train_sample_ids = fetch_embeddings(model, train_dataloader, properties_mean, properties_std)
    z_ood, _ = fetch_embeddings(model, test_dataloader, properties_mean, properties_std)

    graph_train, anchors_idx = precompute_id_geometry(z_id)
    anchors_idx = list(anchors_idx)
    anchor_pos = {int(node): pos for pos, node in enumerate(anchors_idx)}

    _, _, anchor_ood, best_d_ood = calc_weights_geodesic_softanchors(
        Z_train=z_id, Z_ood=z_ood, graph_train=graph_train, anchors_idx=anchors_idx
    )
    anchor_ood_slot = np.array([anchor_pos.get(int(a), -1) for a in anchor_ood], dtype=np.int64)

    d_np = np.asarray(best_d_ood.detach().cpu().numpy() if torch.is_tensor(best_d_ood) else best_d_ood)
    d_max_trust = np.quantile(d_np[np.isfinite(d_np)], 0.90)
    trust = np.isfinite(d_np) & (d_np <= d_max_trust) & (anchor_ood_slot >= 0)

    d_ood_f = d_np[trust]
    a_ood_f = anchor_ood_slot[trust]
    w_anchor = aggregate_mean(-d_ood_f, a_ood_f, num_anchors=len(anchors_idx))

    _, _, anchor_id, best_d_id = calc_weights_geodesic_softanchors(
        Z_train=z_id, Z_ood=z_id, graph_train=graph_train, anchors_idx=anchors_idx
    )
    anchor_id_slot = np.array([anchor_pos.get(int(a), -1) for a in anchor_id], dtype=np.int64)
    w_id = w_anchor[anchor_id_slot]
    weights_by_id = {train_sample_ids[i]: float(w_id[i]) for i in range(len(w_id))}

    summary_id = summarize_best_d(best_d_id)
    summary_ood = summarize_best_d(best_d_ood)
    print(f"Geodesic summary (ID):  {summary_id}")
    print(f"Geodesic summary (OOD): {summary_ood}")

    return weights_by_id


def save_geodesic_weights(weights_by_id, path=None):
    """Save geodesic weights as JSON (keys must be strings for JSON compatibility)."""
    path = path or GEODESIC_WEIGHTS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {str(k): v for k, v in weights_by_id.items()}
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Geodesic weights saved: {path}")


def load_geodesic_weights(path=None):
    """Load geodesic weights; restore integer sample IDs."""
    path = path or GEODESIC_WEIGHTS_PATH
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def predictor_checkpoint_path(target_name):
    """Return the canonical path for a fine-tuned property predictor."""
    safe_name = target_name.replace(" ", "_")
    return CHECKPOINT_DIR / f"property_predictor_{safe_name}.pth"
