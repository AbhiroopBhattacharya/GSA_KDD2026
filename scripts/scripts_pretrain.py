"""
Phase 1 — Unsupervised JEPA pretraining.

Trains the multimodal JEPA backbone (tabular transformer + image + MatSciBERT text)
on in-domain composite data with masked-column prediction. Saves the best checkpoint
to Results/Checkpoint/jepa_pretrain_best.pth.

Usage:
    python scripts_pretrain.py
    python scripts_pretrain.py --epochs 25 --batch-size 128 --lr 5e-5
"""

import argparse

import torch
import torch.optim as optim

import scripts_common as sc


def train_unsupervised(model, train_dataloader, num_epochs=25, lr=5e-5, save_path=None):
    """
    Unsupervised JEPA pretraining loop.

    Optimizes cosine alignment and masked-column reconstruction losses.
    The target encoder is updated via exponential moving average (EMA) after
    each step when the model supports it.
    """
    print("\n" + "=" * 80)
    print("PHASE 1: UNSUPERVISED JEPA PRETRAINING")
    print("=" * 80)

    trainable_params = model.get_trainable_parameters()
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)

    # Linear warmup for the first 5% of steps, then constant LR
    total_steps = num_epochs * len(train_dataloader)
    warmup_steps = int(0.05 * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_loss = float("inf")
    best_epoch = 0
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        last_losses = {}

        for batch in train_dataloader:
            optimizer.zero_grad()

            images = batch["image"].to(model.device)
            text = batch["text"]
            properties = batch["properties"].to(model.device)

            forward_output = model(images, properties, text, apply_masking=True)
            losses = model.compute_jepa_losses(forward_output)
            total_loss = losses["total_weighted_loss"]

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            if hasattr(model, "update_target_encoder_ema"):
                model.update_target_encoder_ema()

            epoch_loss += total_loss.item()
            last_losses = losses

        avg_loss = epoch_loss / len(train_dataloader)
        print(f"\nEpoch {epoch + 1}/{num_epochs} | LR: {scheduler.get_last_lr()[0]:.2e}")
        print(f"  Cosine loss:         {last_losses['loss_cosine']:.6f}")
        print(f"  Reconstruction loss: {last_losses['reconstruction_loss']:.6f}")
        print(f"  Total loss:          {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Restore best weights and persist checkpoint
    model.load_state_dict(best_state)
    sc.save_jepa_checkpoint(model, best_epoch, best_loss, path=save_path)
    print(f"\nPretraining complete. Best epoch: {best_epoch}, loss: {best_loss:.6f}")
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Unsupervised JEPA pretraining for composite OOD pipeline")
    parser.add_argument("--epochs", type=int, default=25, help="Number of pretraining epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="AdamW learning rate")
    parser.add_argument(
        "--target",
        type=str,
        default="Young_modulus",
        help="Target column used only to build dataloaders (pretraining is unsupervised)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(sc.JEPA_CHECKPOINT),
        help="Output path for the best JEPA checkpoint",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = sc.get_device()
    print(f"Using device: {device}")
    print(f"Data root:    {sc.DATA_ROOT}")

    train_loader, _, _ = sc.generate_ood_qh1_data(batch_size=args.batch_size, target_name=args.target)

    model, _ = sc.build_jepa_model(device=device)
    train_unsupervised(
        model,
        train_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        save_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()
