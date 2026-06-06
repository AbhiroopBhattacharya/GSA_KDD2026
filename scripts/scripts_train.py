"""
Phase 2 — Geodesic-weighted supervised fine-tuning.

Loads the pretrained JEPA encoder, computes geodesic importance weights in the
latent space, and fine-tunes a property prediction head for each target property.

Usage:
    python scripts_train.py
    python scripts_train.py --targets Young_modulus Bulk_modulus
    python scripts_train.py --jepa-checkpoint Results/Checkpoint/jepa_pretrain_best.pth
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import scripts_common as sc
from jepa_tit_condtransformer import SupervisedPropertyPredictor


def fine_tune_geodesic(
    model,
    property_predictor,
    train_dataloader,
    val_dataloader,
    properties_mean,
    properties_std,
    targets_mean,
    targets_std,
    weights_by_id_dict,
    num_epochs=15,
    lr=1e-4,
    device=None,
    lambda_boost=0.8,
):
    """
    Fine-tune the property head with geodesic sample reweighting.

    The JEPA backbone is frozen. Training loss combines a base MSE with an
    extra term that up-weights samples whose geodesic weight exceeds 1.0:

        loss = mean(MSE) + lambda_boost * mean(max(w - 1, 0) * MSE)
    """
    device = device or sc.get_device()

    # Freeze the pretrained JEPA encoder
    for module in (model.context_encoder, model.target_encoder, model.masked_column_predictor):
        for param in module.parameters():
            param.requires_grad = False

    optimizer = torch.optim.AdamW(property_predictor.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.MSELoss(reduction="none")

    best_val_loss = float("inf")
    best_state = None
    patience, patience_counter = 10, 0

    for epoch in range(num_epochs):
        property_predictor.train()
        train_loss = 0.0
        num_batches = 0

        for batch in train_dataloader:
            images = batch["image"].to(device)
            text = batch["text"]
            properties = batch["properties"].to(device)
            targets = batch["target"].to(device).float()
            sample_id = batch["id"]

            # Disable multimodal conditioning when a batch lacks images or text
            if not batch["has_image"].any() or not batch["has_text"].any():
                model.use_conditioning = False

            props_norm = (properties - properties_mean) / properties_std
            targets_norm = (targets - targets_mean) / targets_std

            if torch.is_tensor(sample_id):
                sample_ids = sample_id.detach().cpu().tolist()
            else:
                sample_ids = sample_id

            embeddings = sc.get_tabular_embeddings(model, images, props_norm, text)
            predictions = property_predictor(embeddings)
            loss_mse = criterion(predictions.squeeze(-1), targets_norm)

            w = torch.tensor(
                [float(weights_by_id_dict.get(sid, 1.0)) for sid in sample_ids],
                device=device,
                dtype=torch.float32,
            )
            loss = loss_mse.mean() + lambda_boost * ((w - 1.0).clamp(min=0.0) * loss_mse).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(property_predictor.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        scheduler.step()

        # Validation (unweighted MSE on denormalized targets)
        property_predictor.eval()
        model.eval()
        val_loss, all_preds, all_targets = 0.0, [], []

        with torch.no_grad():
            for batch in val_dataloader:
                images = batch["image"].to(device)
                text = batch["text"]
                properties = batch["properties"].to(device)
                targets = batch["target"].to(device).float()

                if not batch.get("has_image", torch.tensor([True])).any():
                    model.use_conditioning = False

                props_norm = (properties - properties_mean) / properties_std
                embeddings = sc.get_tabular_embeddings(model, images, props_norm, text)
                preds_norm = property_predictor(embeddings).squeeze(-1)
                preds = preds_norm * targets_std + targets_mean

                batch_loss = criterion(preds, targets).mean()
                val_loss += batch_loss.item()
                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        val_r2 = r2_score(all_targets, all_preds)
        val_rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
        avg_val_loss = val_loss / len(val_dataloader)

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"train_loss={train_loss / num_batches:.4f} | "
            f"val_loss={avg_val_loss:.4f} | val_R2={val_r2:.4f} | val_RMSE={val_rmse:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in property_predictor.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    if best_state is not None:
        property_predictor.load_state_dict(best_state)
    return property_predictor


def parse_args():
    parser = argparse.ArgumentParser(description="Geodesic-weighted fine-tuning for composite property prediction")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=15, help="Fine-tuning epochs per target")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--jepa-checkpoint",
        type=str,
        default=str(sc.JEPA_CHECKPOINT),
        help="Path to pretrained JEPA checkpoint from scripts_pretrain.py",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=sc.DEFAULT_TARGETS,
        help="Target properties to fine-tune (one head per target)",
    )
    parser.add_argument(
        "--recompute-weights",
        action="store_true",
        help="Recompute geodesic weights even if a cached file exists",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = sc.get_device()
    print(f"Using device: {device}")

    # Load pretrained JEPA encoder
    model, _, _ = sc.build_jepa_model(device=device)
    sc.load_jepa_checkpoint(model, path=args.jepa_checkpoint, device=device)

    # Build dataloaders once to compute geodesic weights (target-agnostic)
    train_loader, _, test_loader = sc.generate_ood_qh1_data(
        batch_size=args.batch_size, target_name=args.targets[0]
    )
    properties_mean, properties_std, _, _ = sc.get_properties_mean_std(train_loader, device)

    if sc.GEODESIC_WEIGHTS_PATH.exists() and not args.recompute_weights:
        print(f"Loading cached geodesic weights from {sc.GEODESIC_WEIGHTS_PATH}")
        weights_by_id = sc.load_geodesic_weights()
    else:
        print("Computing geodesic weights in JEPA latent space...")
        weights_by_id = sc.calculate_geodesic_weights(
            model, train_loader, test_loader, properties_mean, properties_std
        )
        sc.save_geodesic_weights(weights_by_id)

    # Fine-tune one property head per target
    for target in args.targets:
        print("\n" + "=" * 80)
        print(f"Fine-tuning target: {target}")
        print("=" * 80)

        train_loader, val_loader, _ = sc.generate_ood_qh1_data(
            batch_size=args.batch_size, target_name=target
        )
        props_mean, props_std, tgt_mean, tgt_std = sc.get_properties_mean_std(train_loader, device)
        sc.save_normalization_stats(target, props_mean, props_std, tgt_mean, tgt_std)

        predictor = SupervisedPropertyPredictor(
            vit_hidden_dim=128,
            num_properties=1,
            hidden_dim=64,
            dropout=0.2,
        ).to(device)

        predictor = fine_tune_geodesic(
            model=model,
            property_predictor=predictor,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            properties_mean=props_mean,
            properties_std=props_std,
            targets_mean=tgt_mean,
            targets_std=tgt_std,
            weights_by_id_dict=weights_by_id,
            num_epochs=args.epochs,
            lr=args.lr,
            device=device,
        )

        ckpt_path = sc.predictor_checkpoint_path(target)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(predictor.state_dict(), ckpt_path)
        print(f"Property predictor saved: {ckpt_path}")


if __name__ == "__main__":
    main()
