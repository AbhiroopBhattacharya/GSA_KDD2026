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


class GradientReversalLayer(torch.autograd.Function):
    """
    Gradient Reversal Layer for DANN.
    During forward pass, acts as identity. During backward pass, multiplies gradients by -lambda.
    """
    @staticmethod
    def forward(ctx, x, lambda_grl):
        ctx.lambda_grl = lambda_grl
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_grl, None


class DANNDomainAdapter(nn.Module):
    """
    Domain Adversarial Neural Network (DANN) adapter.
    
    - encoder: maps X -> features (N, d)
    - head: maps features -> predictions (N, out_dim)
    - domain_discriminator: maps features -> domain prediction (N, 1) [0=source, 1=target]
    """
    def __init__(self, encoder: nn.Module, head: nn.Module, feature_dim: int):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.feature_dim = feature_dim
        
        # Domain discriminator: predicts if features come from source (0) or target (1)
        self.domain_discriminator = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output probability of being target domain
        )

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
        lambda_dann: float = 1.0,
    ):
        """
        DANN training step:
        1. Extract features from source and target
        2. Compute task loss on source
        3. Compute domain adversarial loss (with gradient reversal)
        
        Returns: total_loss, task_loss, domain_loss
        """
        # Extract features
        fs = self.forward_features(xs)  # Source features
        ft = self.forward_features(xt)  # Target features

        if self.head is None:
            raise ValueError("Provide a head to compute task loss (predictions).")

        # Task loss on source domain
        yhat = self.head(fs)
        if ys.dim() == 1 and yhat.dim() == 2 and yhat.size(1) == 1:
            ys_ = ys.view(-1, 1)
        else:
            ys_ = ys
        task_loss = task_criterion(yhat, ys_)
        
        # Domain adversarial loss
        # Concatenate source and target features
        all_features = torch.cat([fs, ft], dim=0)  # [Ns + Nt, d]
        
        # Create domain labels: 0 for source, 1 for target
        ns = fs.size(0)
        nt = ft.size(0)
        domain_labels = torch.cat([
            torch.zeros(ns, 1, device=fs.device),  # Source = 0
            torch.ones(nt, 1, device=ft.device)     # Target = 1
        ], dim=0)
        
        # Apply gradient reversal layer to features before domain discriminator
        # This makes the encoder try to fool the discriminator
        reversed_features = GradientReversalLayer.apply(all_features, lambda_dann)
        
        # Domain predictions
        domain_preds = self.domain_discriminator(reversed_features)
        
        # Binary cross-entropy loss for domain classification
        domain_criterion = nn.BCELoss()
        domain_loss = domain_criterion(domain_preds, domain_labels)
        
        # Total loss: task loss + domain adversarial loss
        # lambda_dann is applied in GradientReversalLayer to control gradient scaling
        total = task_loss + domain_loss
        
        return total, task_loss.detach(), domain_loss.detach()


def fit_dann(
    model: DANNDomainAdapter,
    source_loader,
    target_loader,
    optimizer: torch.optim.Optimizer,
    domain_optimizer: torch.optim.Optimizer,
    task_criterion: nn.Module,
    lambda_dann: float =0.1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    epochs: int = 50,
    grad_clip: float = 1.0,
    properties_mean=None,
    properties_std=None,
    targets_mean=None,
    targets_std=None,
    val_loader=None,
    scheduler=None,
):
    """
    DANN training loop:
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
        total_sum = task_sum = domain_sum = 0.0
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

            # Extract features for domain loss calculation
            fs = model.forward_features(xs)
            ft = model.forward_features(xt)
            
            # Task loss on source domain
            yhat = model.head(fs)
            if ys.dim() == 1 and yhat.dim() == 2 and yhat.size(1) == 1:
                ys_ = ys.view(-1, 1)
            else:
                ys_ = ys
            task_loss = task_criterion(yhat, ys_)
            
            # Domain adversarial loss - need to compute separately for encoder and discriminator
            all_features = torch.cat([fs, ft], dim=0)
            ns = fs.size(0)
            nt = ft.size(0)
            domain_labels = torch.cat([
                torch.zeros(ns, 1, device=fs.device),
                torch.ones(nt, 1, device=ft.device)
            ], dim=0)
            
            # For encoder: use gradient reversal (to fool discriminator)
            reversed_features = GradientReversalLayer.apply(all_features, lambda_dann)
            domain_preds_reversed = model.domain_discriminator(reversed_features)
            domain_criterion = nn.BCELoss()
            domain_loss_encoder = domain_criterion(domain_preds_reversed, domain_labels)
            
            # Total loss for encoder: task_loss + domain_loss (with gradient reversal)
            total_loss = task_loss + domain_loss_encoder
            
            # Optimize encoder + task head (with gradient reversal for domain)
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.encoder.parameters(), grad_clip)
                nn.utils.clip_grad_norm_(model.head.parameters(), grad_clip)
            optimizer.step()
            
            # For discriminator: recalculate with detached features (no gradient reversal)
            # This creates a fresh computation graph for the discriminator
            all_features_detached = torch.cat([fs.detach(), ft.detach()], dim=0)
            domain_preds_normal = model.domain_discriminator(all_features_detached)
            domain_loss_discriminator = domain_criterion(domain_preds_normal, domain_labels)
            
            # Optimize domain discriminator (normal gradients, no reversal)
            domain_optimizer.zero_grad(set_to_none=True)
            domain_loss_discriminator.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.domain_discriminator.parameters(), grad_clip)
            domain_optimizer.step()
            
            # For logging
            domain_l = domain_loss_encoder.detach()
            
            # Debug: print first batch of first epoch
            if ep == 1 and n_batches == 0:
                with torch.no_grad():
                    fs = model.forward_features(xs)
                    ft = model.forward_features(xt)
                    # print(f"Debug - fs shape: {fs.shape}, ft shape: {ft.shape}")
                    # print(f"Debug - Domain loss: {domain_l.item():.6f}")

            total_sum += float(total_loss.detach().cpu())
            task_sum += float(task_loss.detach().cpu())
            domain_sum += float(domain_l.cpu())
            n_batches += 1

        avg_total = total_sum / n_batches if n_batches > 0 else 0.0
        avg_task = task_sum / n_batches if n_batches > 0 else 0.0
        avg_domain = domain_sum / n_batches if n_batches > 0 else 0.0

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
        #     f"task={avg_task:.4f}, domain={avg_domain:.6f}{val_str}"
        # )
def predict_dann(model, dataloader, properties_mean=None, properties_std=None, 
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
def evaluate_dann(predictions, targets):
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

        model = DANNDomainAdapter(encoder=encoder, head=head, feature_dim=64)

        # Separate optimizers for encoder+head vs domain discriminator
        optimizer = torch.optim.Adam(
            list(model.encoder.parameters()) + list(model.head.parameters()),
            lr=1e-4, weight_decay=1e-2
        )
        domain_optimizer = torch.optim.Adam(
            model.domain_discriminator.parameters(),
            lr=1e-5, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        criterion = nn.MSELoss()

        # Split target train into train and val
        target_train_subset, target_val_subset = split_data(target_train_dataset, train_ratio=0.8, random_seed=1234)
        target_val_dataloader = torch.utils.data.DataLoader(
            target_val_subset, batch_size=32, shuffle=False, num_workers=0
        )

        fit_dann(
            model,
            source_loader=source_dataloader,
            target_loader=target_train_dataloader,
            optimizer=optimizer,
            domain_optimizer=domain_optimizer,
            task_criterion=criterion,
            lambda_dann=0.01,
            epochs=50,
            grad_clip=1.0,
            properties_mean=properties_mean,
            properties_std=properties_std,
            targets_mean=targets_mean,
            targets_std=targets_std,
            val_loader=target_val_dataloader,
            scheduler=scheduler
        )
        
        predictions, targets = predict_dann(
            model, target_test_dataloader,
            properties_mean=properties_mean,
            properties_std=properties_std,
            targets_mean=targets_mean,
            targets_std=targets_std
        )
        r2, rmse, mae = evaluate_dann(predictions, targets)
        print(f"Test Results - r2: {r2:.4f}, rmse: {rmse:.4f}, mae: {mae:.4f}")
if __name__ == "__main__":
    main()

class GradientReversalLayer(torch.autograd.Function):
    """
    Gradient Reversal Layer for DANN.
    During forward pass, acts as identity. During backward pass, multiplies gradients by -lambda.
    """
    @staticmethod
    def forward(ctx, x, lambda_grl):
        ctx.lambda_grl = lambda_grl
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_grl, None


class DANNDomainAdapter(nn.Module):
    """
    Domain Adversarial Neural Network (DANN) adapter.
    
    - encoder: maps X -> features (N, d)
    - head: maps features -> predictions (N, out_dim)
    - domain_discriminator: maps features -> domain prediction (N, 1) [0=source, 1=target]
    """
    def __init__(self, encoder: nn.Module, head: nn.Module, feature_dim: int):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.feature_dim = feature_dim
        
        # Domain discriminator: predicts if features come from source (0) or target (1)
        self.domain_discriminator = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output probability of being target domain
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        if feats.dim() != 2:
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
        lambda_dann: float = 1.0,
    ):
        """
        DANN training step:
        1. Extract features from source and target
        2. Compute task loss on source
        3. Compute domain adversarial loss (with gradient reversal)
        
        Returns: total_loss, task_loss, domain_loss
        """
        # Extract features
        fs = self.forward_features(xs)  # Source features
        ft = self.forward_features(xt)  # Target features

        if self.head is None:
            raise ValueError("Provide a head to compute task loss (predictions).")

        # Task loss on source domain
        yhat = self.head(fs)
        if ys.dim() == 1 and yhat.dim() == 2 and yhat.size(1) == 1:
            ys_ = ys.view(-1, 1)
        else:
            ys_ = ys
        task_loss = task_criterion(yhat, ys_)
        
        # Domain adversarial loss
        # Concatenate source and target features
        all_features = torch.cat([fs, ft], dim=0)  # [Ns + Nt, d]
        
        # Create domain labels: 0 for source, 1 for target
        ns = fs.size(0)
        nt = ft.size(0)
        domain_labels = torch.cat([
            torch.zeros(ns, 1, device=fs.device),  # Source = 0
            torch.ones(nt, 1, device=ft.device)     # Target = 1
        ], dim=0)
        
        # Apply gradient reversal layer to features before domain discriminator
        # This makes the encoder try to fool the discriminator
        reversed_features = GradientReversalLayer.apply(all_features, lambda_dann)
        
        # Domain predictions
        domain_preds = self.domain_discriminator(reversed_features)
        
        # Binary cross-entropy loss for domain classification
        domain_criterion = nn.BCELoss()
        domain_loss = domain_criterion(domain_preds, domain_labels)
        
        # Total loss: task loss + domain adversarial loss
        # lambda_dann is applied in GradientReversalLayer to control gradient scaling
        total = task_loss + domain_loss
        
        return total, task_loss.detach(), domain_loss.detach()

