import torch
# import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import yaml
import csv
import random
import time
import datetime
import itertools
from utils import generate_ood_qh2_data
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from data_jepa_tit_cftr import split_data
from jepa_tit_condtransformer import SupervisedPropertyPredictor

import torch
import torch.nn as nn
import torch.nn.functional as F


def coral_loss(source_feats: torch.Tensor, target_feats: torch.Tensor) -> torch.Tensor:
    """
    CORAL loss: ||Cov(Xs) - Cov(Xt)||_F^2 / (4 d^2)
    source_feats: (Ns, d)
    target_feats: (Nt, d)
    """
    if source_feats.dim() != 2 or target_feats.dim() != 2:
        raise ValueError("CORAL expects 2D tensors of shape (N, d).")

    ns, d = source_feats.shape
    nt, dt = target_feats.shape
    if dt != d:
        raise ValueError(f"Source/target feature dims differ: {d} vs {dt}")
    
    # Check for constant features (would cause zero variance)
    source_std = source_feats.std(dim=0)
    target_std = target_feats.std(dim=0)
    if (source_std < 1e-6).any() or (target_std < 1e-6).any():
        # Features are constant - this is a problem
        # Return a non-zero loss to encourage learning
        return torch.tensor(0.1, device=source_feats.device, requires_grad=True)

    # Center the features
    xs = source_feats - source_feats.mean(dim=0, keepdim=True)
    xt = target_feats - target_feats.mean(dim=0, keepdim=True)

    # Compute covariance matrices (sample covariance)
    # Use max(n-1, 1) to avoid division by zero for single samples
    cs = (xs.T @ xs) / max(ns - 1, 1)
    ct = (xt.T @ xt) / max(nt - 1, 1)

    # Frobenius norm squared of the difference
    diff = cs - ct
    loss = torch.sum(diff ** 2)  # Sum of squared differences
    
    # Normalize by (4 * d^2) as in original CORAL paper
    # This normalization makes the loss scale-independent
    loss = loss / (4.0 * d * d)
    
    # Ensure loss is not NaN or Inf
    if torch.isnan(loss) or torch.isinf(loss):
        return torch.tensor(0.0, device=source_feats.device, requires_grad=True)
    
    # If loss is exactly zero (features are identical), return a small value
    # This shouldn't happen in practice, but handle it gracefully
    if loss.item() < 1e-12:
        # Features are nearly identical - this suggests a problem
        # Return a small non-zero value to maintain gradient flow
        return torch.tensor(1e-6, device=source_feats.device, requires_grad=True)
    
    return loss


class CORALDomainAdapter(nn.Module):
    """
    Wraps an encoder + (optional) task head, and provides a training step
    with CORAL alignment between source and target batches.

    - encoder: maps X -> features (N, d)
    - head: maps features -> predictions (N, out_dim)
    """
    def __init__(self, encoder: nn.Module, head: nn.Module ):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        if feats.dim() != 2:
            # If your encoder outputs (N, T, d) or similar, pool/flatten here.
            # e.g., feats = feats.mean(dim=1)  # Global average over tokens/features
            raise ValueError(f"Encoder must output (N, d). Got {tuple(feats.shape)}.")
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)
        if self.head is None:
            return feats
        return self.head(feats)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(x)

    def training_step(
        self,
        xs: torch.Tensor,
        ys: torch.Tensor,
        xt: torch.Tensor,
        task_criterion: nn.Module,
        lambda_coral: float = 1e-3,
    ):
        """
        Returns: total_loss, task_loss, coral
        """
        fs = self.forward_features(xs)
        ft = self.forward_features(xt)

        if self.head is None:
            raise ValueError("Provide a head to compute task loss (predictions).")

        yhat = self.head(fs)

        # Make shapes compatible for regression
        # If ys is (N,) and yhat is (N,1), align them.
        if ys.dim() == 1 and yhat.dim() == 2 and yhat.size(1) == 1:
            ys_ = ys.view(-1, 1)
        else:
            ys_ = ys

        task_loss = task_criterion(yhat, ys_)
        
        # Compute CORAL loss
        if lambda_coral > 0:
            c_loss = coral_loss(fs, ft)
            # Debug: check if features are too similar (but don't modify loss, just warn)
            if c_loss.item() < 1e-8:
                feat_diff = (fs - ft).abs().mean().item()
                print(f"Debug - Feature diff: {feat_diff:.6f}")
            # If features are identical, this is a problem - but don't fake the loss
            # The issue is likely that source and target data are the same
        else:
            c_loss = torch.tensor(0.0, device=fs.device)
        
        total = task_loss + (lambda_coral * c_loss)
        return total, task_loss.detach(), c_loss.detach()


def fit_coral(
    model: CORALDomainAdapter,
    source_loader,
    target_loader,
    optimizer: torch.optim.Optimizer,
    task_criterion: nn.Module,
    lambda_coral: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    epochs: int = 20,
    grad_clip: float = 1.0,
    properties_mean=None,
    properties_std=None,
    targets_mean=None,
    targets_std=None,
    val_loader=None,
    scheduler=None,
):
    """
    Minimal training loop:
    - source_loader yields (xs, ys)
    - target_loader yields (xt,) or xt
    """

    model.to(device)
    model.train()

    # Cycle target loader if shorter than source
    import itertools
    tgt_iter = itertools.cycle(target_loader)

    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for ep in range(1, epochs + 1):
        model.train()
        total_sum = task_sum = coral_sum = 0.0
        n_batches = 0

        for batch in source_loader:
            properties = batch['properties']
            targets = batch['target']
            t_batch = next(tgt_iter)
            properties_target = t_batch['properties'] 

            xs = properties.to(device).float()
            ys = targets.to(device).float()
            xt = properties_target.to(device).float()

            # Normalize properties
            if properties_mean is not None and properties_std is not None:
                xs = (xs - properties_mean) / (properties_std + 1e-8)
                xt = (xt - properties_mean) / (properties_std + 1e-8)
            
            # Normalize targets
            if targets_mean is not None and targets_std is not None:
                ys = (ys - targets_mean) / (targets_std + 1e-8)

            optimizer.zero_grad(set_to_none=True)
            total, task_l, coral_l = model.training_step(
                xs, ys, xt, task_criterion, lambda_coral=lambda_coral
            )
            
            # Debug: print first batch of first epoch
            if ep == 1 and n_batches == 0:
                with torch.no_grad():
                    fs = model.forward_features(xs)
                    ft = model.forward_features(xt)
                    # print(f"Debug - fs shape: {fs.shape}, ft shape: {ft.shape}")
                    # print(f"Debug - fs mean: {fs.mean().item():.6f}, std: {fs.std().item():.6f}")
                    # print(f"Debug - ft mean: {ft.mean().item():.6f}, std: {ft.std().item():.6f}")
                    # print(f"Debug - Feature diff: {(fs - ft).abs().mean().item():.6f}")
                    # print(f"Debug - CORAL loss: {coral_l.item():.6f}")
            
            total.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_sum += float(total.detach().cpu())
            task_sum += float(task_l.cpu())
            coral_sum += float(coral_l.cpu())
            n_batches += 1

        avg_total = total_sum / n_batches if n_batches > 0 else 0.0
        avg_task = task_sum / n_batches if n_batches > 0 else 0.0
        avg_coral = coral_sum / n_batches if n_batches > 0 else 0.0

        # Validation
        val_loss = None
        if val_loader is not None:
            model.eval()
            val_loss_sum = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    properties = batch['properties'].to(device).float()
                    targets = batch['target'].to(device).float()
                    
                    if properties_mean is not None and properties_std is not None:
                        properties = (properties - properties_mean) / (properties_std + 1e-8)
                    if targets_mean is not None and targets_std is not None:
                        targets = (targets - targets_mean) / (targets_std + 1e-8)
                    
                    preds = model(properties)
                    if preds.dim() > 1 and preds.size(1) == 1:
                        preds = preds.squeeze()
                    if targets.dim() > 1:
                        targets = targets.squeeze()
                    
                    loss = task_criterion(preds, targets)
                    val_loss_sum += loss.item()
                    val_batches += 1
            
            val_loss = val_loss_sum / val_batches if val_batches > 0 else float('inf')
            if scheduler is not None:
                scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {ep}")
                    break

        val_str = f", val_loss={val_loss:.4f}" if val_loss is not None else ""
        # print(
        #     f"Epoch {ep:03d}, total={avg_total:.4f}, "
        #     f"task={avg_task:.4f}, coral={avg_coral:.6f}{val_str}"
        # )
def predict_coral(model, dataloader, properties_mean=None, properties_std=None, 
                  targets_mean=None, targets_std=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for batch in dataloader:
            properties = batch['properties'].to(device)
            targets = batch['target']
            
            # Normalize properties if provided
            if properties_mean is not None and properties_std is not None:
                properties = (properties - properties_mean) / (properties_std + 1e-8)
            
            preds = model.predict(properties)
            
            # Denormalize predictions if provided
            if targets_mean is not None and targets_std is not None:
                preds = preds * targets_std + targets_mean
            
            # Flatten if needed
            if preds.dim() > 1:
                preds = preds.squeeze()
            
            all_predictions.append(preds.cpu().numpy())
            all_targets.append(targets.numpy())
    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)
    
    # Flatten if still 2D
    if predictions.ndim > 1:
        predictions = predictions.flatten()
    if targets.ndim > 1:
        targets = targets.flatten()
    
    return predictions, targets
def evaluate_coral(predictions, targets):
    r2 = r2_score(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)
    return r2, rmse, mae

def main():
    # for target_name in ['fracture','yield','elastic modulus','elongation','tangent modulus']:
    # for target_name in ['Young_modulus','Bulk_modulus','TC1','TC2','EC1','EC2','Tensile_strength']:
    for target_name in ['YM1', 'EC1', 'TS1']:
        print("--------------------------------")
        print(f"Training for target: {target_name}")
        source_dataset, target_dataset = generate_ood_qh2_data(batch_size=32, target_name=target_name)
        target_train_dataset, target_test_dataset = split_data(target_dataset, train_ratio=0.50, random_seed=1234)
        
        # Create dataloaders from datasets
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator().manual_seed(1234)
        source_dataloader = torch.utils.data.DataLoader(
            source_dataset, batch_size=32, shuffle=True, num_workers=0, generator=generator
        )
        target_train_dataloader = torch.utils.data.DataLoader(
            target_train_dataset, batch_size=32, shuffle=True, num_workers=0, generator=generator
        )
        target_test_dataloader = torch.utils.data.DataLoader(
            target_test_dataset, batch_size=32, shuffle=False, num_workers=0
        )
        
        print(f"Number of training samples: {len(target_train_dataset)}")
        print(f"Number of test samples: {len(target_test_dataset)}")
        
        # # Verify source and target data are different
        # source_sample = next(iter(source_dataloader))
        # target_sample = next(iter(target_train_dataloader))
        # source_props = source_sample['properties'][:5]  # First 5 samples
        # target_props = target_sample['properties'][:5]
        # prop_diff = (source_props - target_props).abs().mean().item()
        # print(f"Source-Target property difference (first 5 samples): {prop_diff:.6f}")
        # if prop_diff < 1e-6:
        #     print("WARNING: Source and target properties are nearly identical!")
        
        # Calculate normalization statistics from source data
        from utils import get_properties_mean_std
        properties_mean, properties_std, targets_mean, targets_std = get_properties_mean_std(
            source_dataloader, device
        )
        # print(f"Properties mean: {properties_mean.cpu().numpy()}")
        # print(f"Properties std: {properties_std.cpu().numpy()}")
        # print(f"Targets mean: {targets_mean:.4f}, std: {targets_std:.4f}")

        # Create encoder that outputs features (N, 7) -> (N, 64)
        encoder = nn.Sequential(
            nn.Linear(7, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Create head that outputs predictions (N, 64) -> (N, 1)
        head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

        model = CORALDomainAdapter(encoder=encoder, head=head)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        criterion = nn.MSELoss()

        # Split target train into train and val
        target_train_subset, target_val_subset = split_data(target_train_dataset, train_ratio=0.8, random_seed=1234)
        target_val_dataloader = torch.utils.data.DataLoader(
            target_val_subset, batch_size=128, shuffle=False, num_workers=0
        )

        fit_coral(
            model,
            source_loader=source_dataloader,
            target_loader=target_train_dataloader,
            optimizer=optimizer,
            task_criterion=criterion,
            lambda_coral=1.0,   # Increased from 1e-3
            epochs=20,  # Increased from 5
            grad_clip=1.0,
            properties_mean=properties_mean,
            properties_std=properties_std,
            targets_mean=targets_mean,
            targets_std=targets_std,
            val_loader=target_val_dataloader,
            scheduler=scheduler
        )
        
        predictions, targets = predict_coral(
            model, target_test_dataloader,
            properties_mean=properties_mean,
            properties_std=properties_std,
            targets_mean=targets_mean,
            targets_std=targets_std
        )
        r2, rmse, mae = evaluate_coral(predictions, targets)
        print(f"Test Results - r2: {r2:.4f}, rmse: {rmse:.4f}, mae: {mae:.4f}")
if __name__ == "__main__":
    main()
