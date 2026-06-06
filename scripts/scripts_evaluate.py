"""
Phase 3 — Evaluation on validation (ID) and test (OOD) splits.

Loads the pretrained JEPA encoder and fine-tuned property predictors, then
reports R², RMSE, and MAE for each target property.

Usage:
    python scripts_evaluate.py
    python scripts_evaluate.py --targets Young_modulus Bulk_modulus
    python scripts_evaluate.py --split test          # OOD test only
    python scripts_evaluate.py --split val test      # both splits
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import scripts_common as sc
from jepa_tit_condtransformer import SupervisedPropertyPredictor


def evaluate_split(
    model,
    property_predictor,
    dataloader,
    properties_mean,
    properties_std,
    targets_mean,
    targets_std,
    device,
    split_name="test",
):
    """
    Run inference on a dataloader and compute regression metrics.

    Returns:
        dict with keys: r2, rmse, mae, predictions, targets
    """
    model.eval()
    property_predictor.eval()
    all_predictions, all_targets = [], []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            text = batch["text"]
            properties = batch["properties"].to(device)
            targets = batch["target"].to(device).float()

            if not batch.get("has_image", torch.tensor([True])).any():
                model.use_conditioning = False

            props_norm = (properties - properties_mean) / properties_std
            output = model(images, props_norm, text, apply_masking=False)
            embeddings = output["original_embeddings"]["tabular"]

            preds_norm = property_predictor(embeddings).squeeze(-1)
            preds = preds_norm * targets_std + targets_mean

            all_predictions.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    predictions = np.concatenate(all_predictions).flatten()
    targets = np.concatenate(all_targets).flatten()

    r2 = r2_score(targets, predictions)
    rmse = float(np.sqrt(mean_squared_error(targets, predictions)))
    mae = float(mean_absolute_error(targets, predictions))

    print(f"  [{split_name}] R²={r2:.4f} | RMSE={rmse:.4f} | MAE={mae:.4f} | n={len(targets)}")

    return {"split": split_name, "r2": r2, "rmse": rmse, "mae": mae, "n": len(targets)}


def load_property_predictor(target, device):
    """Load a fine-tuned property head from disk."""
    ckpt_path = sc.predictor_checkpoint_path(target)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Predictor checkpoint not found: {ckpt_path}\n"
            f"Run scripts_train.py first for target '{target}'."
        )

    predictor = SupervisedPropertyPredictor(
        vit_hidden_dim=128,
        num_properties=1,
        hidden_dim=64,
        dropout=0.2,
    ).to(device)
    predictor.load_state_dict(torch.load(ckpt_path, map_location=device))
    predictor.eval()
    return predictor


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned composite property predictors")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--jepa-checkpoint",
        type=str,
        default=str(sc.JEPA_CHECKPOINT),
        help="Path to pretrained JEPA checkpoint",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=sc.DEFAULT_TARGETS,
        help="Target properties to evaluate",
    )
    parser.add_argument(
        "--split",
        nargs="+",
        default=["val", "test"],
        choices=["val", "test"],
        help="Which splits to evaluate (val=ID, test=OOD)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(sc.PROJECT_ROOT / "Results" / "evaluation_metrics.json"),
        help="Path to save evaluation metrics as JSON",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = sc.get_device()
    print(f"Using device: {device}")

    model, _, _ = sc.build_jepa_model(device=device)
    sc.load_jepa_checkpoint(model, path=args.jepa_checkpoint, device=device)

    all_results = {}

    for target in args.targets:
        print("\n" + "=" * 80)
        print(f"Evaluating target: {target}")
        print("=" * 80)

        train_loader, val_loader, test_loader = sc.generate_ood_qh1_data(
            batch_size=args.batch_size, target_name=target
        )

        # Prefer saved normalization stats from training; fall back to train-set stats
        if sc.NORM_STATS_PATH.exists():
            props_mean, props_std, tgt_mean, tgt_std = sc.load_normalization_stats(target, device)
        else:
            props_mean, props_std, tgt_mean, tgt_std = sc.get_properties_mean_std(train_loader, device)

        predictor = load_property_predictor(target, device)
        target_results = {}

        if "val" in args.split:
            target_results["val"] = evaluate_split(
                model, predictor, val_loader,
                props_mean, props_std, tgt_mean, tgt_std,
                device, split_name="val (ID)",
            )

        if "test" in args.split:
            target_results["test"] = evaluate_split(
                model, predictor, test_loader,
                props_mean, props_std, tgt_mean, tgt_std,
                device, split_name="test (OOD)",
            )

        all_results[target] = target_results

    # Persist metrics for downstream analysis / paper tables
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nMetrics saved to {output_path}")


if __name__ == "__main__":
    main()
