import datetime
import itertools
import torch.nn as nn
from transformers import AdamW, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset, random_split
from jepa_tit_condtransformer import MAEStyleJEPACompositeModel, SupervisedPropertyPredictor, generate_counterfactuals
import torch.optim as optim
import torch
import pandas as pd
import os
import numpy as np
import random
import yaml
import csv
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import logging
import sys
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from utils import AttentionStore, aggregate_attention, plot_feature_attention, register_attention_hooks
from transformers import AutoTokenizer, AutoModel
from body_combined_jepa_tit_qh import load_cf_predictor
from utils import attention_features, FeatureGraphPredictor, topk_sparsify, get_properties_mean_std, get_loaders
from viz_geodesics import calc_weights_euclid_to_anchor, get_regimes, compute_split_metrics, compute_CGD, find_id_anchor_via_geodesic
# ============================================================================
# DATASET POUR IMAGES + DONNÉES TABULAIRES
# ============================================================================

class CompositeImageTextDataset(Dataset):
        
    
    def __init__(self, csv_file, image_folder, text_folder, target_column, image_size=224, num_augmentations=60, mode='random', 
                 add_gaussian_noise=True, noise_std=0.01 ,pretrain=False):
       
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.text_folder = text_folder
        self.image_size = image_size
        self.num_augmentations = num_augmentations
        self.mode = mode
        self.add_gaussian_noise = add_gaussian_noise
        self.noise_std = noise_std  # Standard deviation for Gaussian noise
        self.property_columns = ['f', 'c', 'v', 'r', 't', 'w', 'dir']
        self.target_column = target_column
        self.pretrain = pretrain

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Pour mode='all', on crée un mapping pour accéder aux données
        if self.mode == 'all':
            self.sample_mapping = []
            for row_idx in range(len(self.data)):
                for aug_idx in range(self.num_augmentations):
                    self.sample_mapping.append((row_idx, aug_idx))
        
       
    def __len__(self):
        if self.mode == 'all':
            # Chaque échantillon a 60 images, donc 60x plus d'échantillons
            return len(self.data) * self.num_augmentations
        else:
            return len(self.data)
    
    def __getitem__(self, idx):
        # Pour mode='all', utiliser le mapping pour obtenir row_idx et aug_idx
        if self.mode == 'all':
            row_idx, aug_idx = self.sample_mapping[idx]
            row = self.data.iloc[row_idx]
            sample_id = int(row['ID'])
            
            # Charger l'image spécifique
            image_folder = os.path.join(self.image_folder, f"{sample_id}")
            image_path = os.path.join(image_folder, f"{sample_id}_{aug_idx}.jpg")
            
            if not os.path.exists(image_path):
                print(f"Warning: Image {image_path} does not exist! Using zero tensor.")
                image = torch.zeros(3, self.image_size, self.image_size)
                has_image = False
            else:
                try:
                    image = Image.open(image_path).convert('RGB')
                    image = self.transform(image)
                    has_image = True
                except Exception as e:
                    print(f"Error loading image {sample_id}_{aug_idx}: {e}")
                    image = torch.zeros(3, self.image_size, self.image_size)
                    has_image = False
        else:
            # Pour les autres modes, utiliser l'index directement
            row = self.data.iloc[idx]
            sample_id = int(row['ID'])
            
            if self.mode == 'random':
                random.seed(1234)
                # MODE: Sélectionne aléatoirement une des 60 images
                aug_idx = random.randint(0, self.num_augmentations - 1)
                image_folder = os.path.join(self.image_folder, f"{sample_id}")
                image_path = os.path.join(image_folder, f"{sample_id}_{aug_idx}.jpg")
                
                if not os.path.exists(image_path):
                    print(f"Image {image_path} does not exist!")
                    raise FileNotFoundError(f"Image {image_path} does not exist!")
                    has_image = False
                try:
                    image = Image.open(image_path).convert('RGB')
                    image = self.transform(image)
                    has_image = True
                except Exception as e:
                    print(f"Error loading image {sample_id}_{aug_idx}: {e}")
                    image = torch.zeros(3, self.image_size, self.image_size)
                    has_image = False
            elif self.mode == 'mean':
                # MODE: Utilise toutes les 60 images et fait la moyenne
                image_folder = os.path.join(self.image_folder, f"{sample_id}")
                all_images = []
                for aug_idx in range(self.num_augmentations):
                    image_path = os.path.join(image_folder, f"{sample_id}_{aug_idx}.jpg")
                    if os.path.exists(image_path):
                        try:
                            img = Image.open(image_path).convert('RGB')
                            img = self.transform(img)
                            all_images.append(img)
                        except Exception as e:
                            print(f"Error loading {image_path}: {e}")
                
                if len(all_images) > 0:
                    image = torch.stack(all_images).mean(dim=0)
                    if len(all_images) < self.num_augmentations and idx == 0:
                        print(f"Warning: Sample {sample_id} has only {len(all_images)}/{self.num_augmentations} images")
                    has_image = True
                else:
                    print(f"Error: No images found for sample {sample_id}")
                    image = torch.zeros(3, self.image_size, self.image_size)
                    has_image = False
            elif self.mode == 'QH_random':
                sample_id_fmt = f"{int(sample_id):03d}"
                self.property_columns = ['NumFibers','MMA','Vf','A11','A12','A13','A22']
            
                image_path = os.path.join(self.image_folder, f"{sample_id_fmt}.png")
                
                if not os.path.exists(image_path):
                    print(f"Image {image_path} does not exist!")
                    # raise FileNotFoundError(f"Image {image_path} does not exist!")
                    image = torch.zeros(3, self.image_size, self.image_size)
                    has_image = False
                
                else:
                    try:
                        image = Image.open(image_path).convert('RGB')
                        image = self.transform(image)
                        has_image = True
                    except Exception as e:
                        print(f"Error loading image {sample_id}: {e}")
                        image = torch.zeros(3, self.image_size, self.image_size)
                        has_image = False
            
            elif self.mode == 'QH_random_1':
                sample_id_fmt = f"{int(sample_id):03d}"
                self.property_columns = ['f_cf','f_cnt','f_gra','f_cu','f_ni','f_resin','w_cf']
            
                image_path = os.path.join(self.image_folder, f"{sample_id_fmt}.png")
                
                if not os.path.exists(image_path):
                    # print(f"Image {image_path} does not exist!")
                    image = torch.zeros(3, self.image_size, self.image_size)
                    # raise FileNotFoundError(f"Image {image_path} does not exist!")
                    has_image = False
                else:
                    try:
                        image = Image.open(image_path).convert('RGB')
                        image = self.transform(image)
                        has_image = True
                    except Exception as e:
                        print(f"Error loading image {sample_id}: {e}")
                        image = torch.zeros(3, self.image_size, self.image_size)
                        has_image = False
            else:  # mode == 'first' ou autre
                # Utilise seulement la première image (ID_0.jpg)
                image_folder = os.path.join(self.image_folder, f"{sample_id}")
                image_path = os.path.join(image_folder, f"{sample_id}_0.jpg")
                
                if not os.path.exists(image_path):
                    print(f"Image {image_path} does not exist!")
                    has_image = False
                    raise FileNotFoundError(f"Image {image_path} does not exist!")
                
                try:
                    image = Image.open(image_path).convert('RGB')
                    image = self.transform(image)
                    has_image = True
                except Exception as e:
                    print(f"Error loading image {sample_id}_0: {e}")
                    image = torch.zeros(3, self.image_size, self.image_size)
                    has_image = False
        # Les données tabulaires sont les mêmes pour toutes les images du même échantillon
        text_folder = os.path.join(self.text_folder, f"{sample_id}_report.txt")
        if os.path.exists(text_folder):
            try:
                with open(text_folder, 'r') as file:
                    text = file.read()
                has_text = True
            except Exception as e:
                print(f"Error loading text {sample_id}: {e}")
                text = ""  # Texte vide
                has_text = False
        else:
            text = ""  # Texte vide par défaut
            # print(f"Text file {text_folder} does not exist!")
            has_text = False
        # Extract properties from row
        properties = torch.tensor(
            [row[col] for col in self.property_columns], 
            dtype=torch.float32
        )
        
        # Add small Gaussian noise to properties for data augmentation
        if self.add_gaussian_noise:
            col_std = properties.std(dim=0, keepdim=True)
            noise = torch.randn_like(properties) * col_std * self.noise_std
            properties_noisy = properties + noise

        # if self.generate_synthetic_data:
            #stiffness = synthetic_stiffness_rule_of_mixtures(properties[0])
           # strength = synthetic_strength_optimal_vf(properties[0])
            #E_transverse = synthetic_E_transverse_halpin_tsai(properties[0])
           
        if self.pretrain:
            target = torch.tensor(0.0, dtype=torch.float32)
        else:
            target = torch.tensor(row[self.target_column], dtype=torch.float32)
        return {
            'image': image,
            'text': text,
            'properties': properties,
            'target': target,
            'id': sample_id,
            'has_image': has_image,
            'has_text': has_text
        }


def split_data(dataset, train_ratio=0.8, random_seed=1234):
    """
    Divise le dataset en ensembles d'entraînement et de test
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset

def get_matscibert_embeddings(tokenizer, matscibert, text_data, device):
        """
        Get MatSciBERT embeddings and pool to single vector per sample
        
        Args:
            text_data: List of strings or tokenized dict
        
        Returns:
            pooled_embeddings: [batch, matscibert_hidden_dim] - pooled text embeddings
        """
    
        if isinstance(text_data, (list, tuple)):
            inputs = tokenizer(text_data, return_tensors="pt", padding=True, truncation=True, max_length=512)
        else:
            inputs = text_data  # Assume already tokenized
        
        inputs = {k: v.to(device=device) for k, v in inputs.items()}
        matscibert_output = matscibert(**inputs)
        text_embeddings = matscibert_output.last_hidden_state  # [batch, seq_len, matscibert_hidden_dim]
        
        # Pool to single vector per sample (mean pooling)
        pooled_embeddings = text_embeddings.mean(dim=1).to(device=device)  # [batch, matscibert_hidden_dim]
        return pooled_embeddings

def get_combined_embeddings(model, properties, text, tokenizer, matscibert):

    matscibert_embeddings = get_matscibert_embeddings(tokenizer, matscibert, text,device=model.device) 
    matscibert_embeddings_pooled = model.image_proj(matscibert_embeddings)
    tabular_embeddings = model.tabular_encoder(properties)
    combined_embeddings = torch.cat([matscibert_embeddings_pooled, tabular_embeddings], dim=-1)

    return combined_embeddings


def train_unsupervised(model, train_dataloader, num_epochs=20, lr=5e-5, 
                                   use_improved_alignment=False, save_dir='./Results/Checkpoint',
                                  reduce_vic_contrastive=False):

    print("\n" + "="*80)
    print("UNSUPERVISED TRAINING (JEPA  + PHYSICS-INFORMED LOSS)")
    print("="*80)
   

    trainable_params = model.get_trainable_parameters()
    total_trainable_params = sum(p.numel() for p in trainable_params)
    print(f"Trainable params: {total_trainable_params:,}")
    

    context_encoder_params = [p for p in model.context_encoder.parameters() if p.requires_grad]
    predictor_params = [p for p in model.masked_column_predictor.parameters() if p.requires_grad]
    print(f"Trainable context encoder params: {sum(p.numel() for p in context_encoder_params):,}")
    print(f"Trainable predictor params: {sum(p.numel() for p in predictor_params):,}")
   
  
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
 
    # # Scheduler
    total_steps = num_epochs * len(train_dataloader)
    warmup_steps = int(0.05 * total_steps)
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        #return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    model.train()
    best_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    for epoch in range(num_epochs):
        total_loss = 0
        loss_components = {}
        
        for batch in train_dataloader:
            optimizer.zero_grad()
            
            images = batch['image'].to(model.device)
            text = batch['text']
            properties = batch['properties'].to(model.device)
            targets = batch.get('target', None)  # Optionnel pour l'entraînement non supervisé
            
            # Forward pass - jepa_tit uses tabular JEPA, so we pass text_data directly
            forward_output = model(images, properties, text, apply_masking=True)
            
            # Calcul des pertes JEP
            losses = model.compute_jepa_losses(forward_output)
            total_loss_value = losses['total_weighted_loss']


            # Backward
            total_loss_value.backward()
          
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            # Update target encoder with EMA (if using MIM)
            if hasattr(model, 'update_target_encoder_ema'):
                model.update_target_encoder_ema()
            
            total_loss += total_loss_value.item()
        
        # Affichage
        print(f"\n Epoch {epoch + 1}/{num_epochs}")
        print(f"Scheduler step: {scheduler.get_last_lr()}")
    
        avg_total = total_loss / len(train_dataloader)
        print(f"  Loss Cosine: {losses['loss_cosine']:.6f}")
        print(f"  Reconstruction Loss: {losses['reconstruction_loss']:.6f}")
        print(f"  Average total loss: {avg_total:.6f}")
        print("-" * 80)
    
        if avg_total < best_loss:
            best_loss = avg_total
            best_model_state = model.state_dict().copy()
            best_epoch = epoch + 1
    
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"checkpoint_composite_jepa_tit_cftr_{best_epoch}_resnet.pth")
    torch.save({
        'epoch': best_epoch,
        'model_state_dict': best_model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': best_loss
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


def load_unsupervised_model(checkpoint_path, model, device='cuda'):
    """
    Charge un modèle non supervisé pré-entraîné depuis un checkpoint
    
    Args:
        checkpoint_path: Chemin vers le fichier checkpoint (.pth)
        model: Instance du modèle à charger
        device: Device sur lequel charger le modèle
    """
    print(f"\n📂 Loading unsupervised model from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Charger les poids du modèle
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    #model.train()  # Mode train pour pouvoir continuer l'entraînement
    
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', 0.0)
    
    print(f"  Model loaded successfully!")
    print(f" Checkpoint info:")
    print(f"      - Epoch: {epoch}")
    print(f"      - Loss: {loss:.6f}")
    
    if 'optimizer_state_dict' in checkpoint:
        print(f"  Optimizer state: Available (not loaded, will use new optimizer for supervised training)")
    else:
        print(f" Optimizer state: Not available")
    
    return model, epoch, loss

def get_tabular_embeddings(model, images, properties, text):
        """Extract tabular embeddings from the model"""
        with torch.no_grad():
            model.eval()
            # Get tabular embeddings without masking
            forward_output = model(images, properties, text, apply_masking=False)
            tabular_embeddings = forward_output['original_embeddings']['tabular']  # [batch, hidden_dim]
        return tabular_embeddings

def get_interaction_features(properties):
    # properties: [B, 7] - batch of feature vectors
    B, k = properties.shape  # k = 7 (number of features)
    interaction_features = []
    
    for i in range(k):
        for j in range(i+1, k):
            # Element-wise multiplication for each sample in batch
            interaction_features.append(properties[:, i] * properties[:, j])  # [B]
    
    # Stack along new dimension: [21, B] -> transpose to [B, 21]
    interaction_features = torch.stack(interaction_features, dim=1)  # [B, 21]
    return interaction_features  

class Interaction_featuresModel(nn.Module):
    def __init__(self, D, M):
        super().__init__()
        # Gating mechanism to learn which interactions are useful
        self.interaction_gate = nn.Sequential(
            nn.Linear(M, M),
            nn.Sigmoid()  # Outputs weights in [0, 1] for each interaction
        )
        self.head = nn.Sequential(
            nn.LayerNorm(D + M),
            nn.Linear(D + M, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, model, images, properties, text):
        z = get_tabular_embeddings(model, images, properties, text)   # [B, D]
        phi = get_interaction_features(properties)             # [B, M]
        
        z = F.layer_norm(z, (z.size(-1),))
        phi = F.layer_norm(phi, (phi.size(-1),))
        
        # Learn which interactions are useful
        gate_weights = self.interaction_gate(phi)  # [B, M]
        phi_gated = phi * gate_weights  # [B, M] - zero out unimportant interactions
        
        u = torch.cat([z, phi_gated], dim=-1)  # [B, D+M]
        return u

def fine_tune_supervised_property_predictor(model, property_predictor, train_dataloader, val_dataloader,
                                           properties_mean, properties_std, targets_mean, targets_std,
                                           num_epochs=10, lr=1e-4, device='cuda:0'):
   
    hidden_dim = 64
    for p in model.context_encoder.parameters():
        p.requires_grad = False
    for p in model.target_encoder.parameters():
        p.requires_grad = False
    for p in model.masked_column_predictor.parameters():
        p.requires_grad = False
    for p in property_predictor.parameters():
        p.requires_grad = True
    # Get tabular embeddings from the base model
 
    def get_augmented_properties(model, images, properties, text, vit_hidden_dim=64, device='cuda:0'):
        D = vit_hidden_dim*7
        M = 21 # 7 properties, 7 interaction features
        interaction_features_model = Interaction_featuresModel(D, M).to(device)
        interaction_features_model.train()
        u = interaction_features_model(model, images, properties, text)
        return u
    
    # encoder_params = list(model.context_encoder.parameters())
    # head_params    = list(property_predictor.parameters())

    # optimizer = torch.optim.AdamW(
    # [
    #     {"params": head_params, "lr": 1e-4},
    #     {"params": encoder_params, "lr": 1e-5},  # 10x smaller for encoder
    # ],
    # weight_decay=1e-2,
    # )
    trainable_params = list(property_predictor.parameters())
    # #trainable_params = encoder_params + head_params
    # print(f"Trainable parameters: {sum(p.numel() for p in trainable_params)}")
    
    # # # Optimizer and loss
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        weight_decay=1e-2
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.MSELoss()

    model.train()
    property_predictor.train()
    # Training loop
    best_val_loss = float('inf')
    best_r2 = -float('inf')
    train_losses = []
    val_losses = []
    val_r2_scores = []
    targets_list = []
    patience = 10
    patience_counter = 0
    for epoch in range(num_epochs):
        # Training
        train_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_dataloader):
            images = batch['image'].to(device)
            text = batch['text']
            properties = batch['properties'].to(device)
            targets = batch['target'].to(device).float()
            
            properties_normalized = (properties - properties_mean) / properties_std
            targets_normalized = (targets - targets_mean) / targets_std
            
            tabular_embeddings = get_tabular_embeddings(model,images, properties_normalized, text)
            #augmented_properties = get_augmented_properties(model, images, properties_normalized, text)
            predictions = property_predictor(tabular_embeddings)  # [batch, 1]
            loss = criterion(predictions.squeeze(-1), targets_normalized)
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(property_predictor.parameters(), max_norm=1.0)
            # torch.nn.utils.clip_grad_norm_(model.context_encoder.parameters(), max_norm=1.0)
            optimizer.step()
        
            train_loss += loss.item()
            num_batches += 1
            # if num_batches>2:
            #     break
        scheduler.step()
        # print(f"Scheduler step: {scheduler.get_last_lr()}")
        avg_train_loss = train_loss / num_batches if num_batches > 0 else 0.0
        train_losses.append(avg_train_loss)
        # print(f"Epoch {epoch+1}/{num_epochs}")
        # print(f"Train Loss: {avg_train_loss:.6f}")
    
        # Validation
        property_predictor.eval()
        val_loss = 0.0
        all_predictions = []
        all_targets = []
            
        with torch.no_grad():
            model.eval()
            for batch in val_dataloader:
                images = batch['image'].to(device)
                text = batch['text']
                properties = batch['properties'].to(device)
                # for col_idx in range(properties.size(1)):
                #     properties[:, col_idx] = (properties[:, col_idx] - properties[:, col_idx].mean()) / properties[:, col_idx].std()
                targets = batch['target'].to(device).float()
                properties_normalized = (properties - properties_mean) / properties_std
                targets_normalized = (targets - targets_mean) / targets_std
                tabular_embeddings = get_tabular_embeddings(model,images, properties_normalized, text)
                #augmented_properties = get_augmented_properties(model, images, properties_normalized, text)
                # Predict properties
                predictions_normalized = property_predictor(tabular_embeddings) 
                predictions = predictions_normalized*targets_std + targets_mean
                
                loss = criterion(predictions.squeeze(-1), targets_normalized)
                val_loss += loss.item()
                
                all_predictions.append(predictions.squeeze(-1).cpu().numpy())
                all_targets.append(targets.cpu().numpy())
    
        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)
        val_r2 = r2_score(all_targets, all_predictions)
        val_rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
        val_mae = mean_absolute_error(all_targets, all_predictions)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_r2 = val_r2
            best_rmse = val_rmse
            best_mae = val_mae
            patience_counter = 0
            #torch.save(property_predictor.state_dict(), f'./Results/Checkpoint/property_predictor_tit_{epoch+1}.pth')
            # print(f"Best Val Loss: {best_val_loss:.6f} | Best R²: {best_r2:.4f} | Best RMSE: {best_rmse:.4f} | Best MAE: {best_mae:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    return model, property_predictor

def evaluate_supervised_property_predictor( model, property_predictor, test_dataloader,properties_mean, properties_std, targets_mean,targets_std, device):
    print("\n" + "="*80)
    print("FINAL EVALUATION ON TEST SET")
    print("="*80)
    
    def get_augmented_properties(model, images, properties, text, vit_hidden_dim=64, device='cuda:0'):
        D = vit_hidden_dim*7
        M = 21 # 7 properties, 7 interaction features
        interaction_features_model = Interaction_featuresModel(D, M).to(device)
        interaction_features_model.eval()
        with torch.no_grad():
            u = interaction_features_model(model, images, properties, text)
        return u

    model.eval()
    property_predictor.eval()
    test_predictions = []
    test_targets = []
    with torch.no_grad():
        for batch in test_dataloader:
            images = batch['image'].to(device)  # Not used but kept for compatibility
            text = batch['text']
            properties = batch['properties'].to(device)
            targets = batch['target'].to(device).float()
            properties_normalized = (properties - properties_mean) / properties_std
            # # Get tabular embeddings from the model
            forward_output = model(images, properties_normalized, text, apply_masking=False)
            tabular_embeddings = forward_output['original_embeddings']['tabular']  # [batch, hidden_dim]
            #augmented_properties = get_augmented_properties(model, images, properties_normalized, text)
            predictions_normalized = property_predictor(tabular_embeddings).squeeze(-1)
            predictions = (predictions_normalized * targets_std) + targets_mean
            test_predictions.append(predictions.cpu().numpy())
            test_targets.append(targets.cpu().numpy())
    
    test_predictions = np.concatenate(test_predictions)
    test_targets = np.concatenate(test_targets)
    
    # S'assurer que les deux arrays ont la même forme (1D)
    test_predictions = test_predictions.flatten()
    test_targets = test_targets.flatten()
    
    test_r2 = r2_score(test_targets, test_predictions)
    test_rmse = np.sqrt(mean_squared_error(test_targets, test_predictions))
    test_mae = mean_absolute_error(test_targets, test_predictions)
    
    print(f"Test R²: {test_r2:.4f} | RMSE: {test_rmse:.4f} | MAE: {test_mae:.4f}")
    return test_predictions

def viz_attention(
    model,
    dataloader,
    properties_mean,
    properties_std,
    feature_names,
    attention_context_store,
    attention_target_store,
    num_batches=10,
):
    model.eval()

    # Move stats to correct device
    properties_mean = properties_mean.to(model.device)
    properties_std  = properties_std.to(model.device)

    attn_context_list = []
    attn_target_list = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # if batch_idx >= num_batches:
            #     break            
            attention_context_store.clear()
            attention_target_store.clear()

            properties = batch['properties'].to(model.device)
            properties_normalized = (properties - properties_mean) / (properties_std + 1e-8)
            
            # Context encoder attention
            _ = model.context_encoder(properties_normalized)
            attn_context = aggregate_attention(attention_context_store)
            attn_context_list.append(attn_context)
            # Target encoder attention
            #attention_target_store = register_attention_hooks(model.target_encoder)
            _ = model.target_encoder(properties_normalized)
            attn_target = aggregate_attention(attention_target_store)
            attn_target_list.append(attn_target)
        attn_context_array = np.stack(attn_context_list)
        attn_target_array = np.stack(attn_target_list)
        attn_context_array = attn_context_array.mean(axis=0)
        attn_target_array = attn_target_array.mean(axis=0)
        plot_feature_attention(attn_context_array, feature_names, title=f"Context Encoder Attention_mean_test_(batch {batch_idx})")
        plot_feature_attention(attn_target_array, feature_names, title=f"Target Encoder Attention_mean_test_(batch {batch_idx})")

def fine_tune_supervised_property_predictor_with_weights(model, property_predictor, train_dataloader, val_dataloader,
                                           properties_mean, properties_std, targets_mean, targets_std, attention_context_store,
                                           num_epochs=10, lr=1e-4, device='cuda:0'):
   
    trainable_params = list(property_predictor.parameters())
    #trainable_params = encoder_params + head_params
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params)}")
    def get_attention_context(properties_normalized, device):
        with torch.no_grad():
            model.eval()
            attention_context_store.clear()
            _ = model.context_encoder(properties_normalized)
            attention_context = aggregate_attention(attention_context_store)
            attention_context = topk_sparsify(attention_context, k=3).float().to(device)
            return attention_context
    # # Optimizer and loss
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.MSELoss()

    model.train()
    property_predictor.train()  # Set property predictor to train mode
    # Training loop
    best_val_loss = float('inf')
    best_r2 = -float('inf')
    train_losses = []
    val_losses = []
    val_r2_scores = []
    targets_list = []
    patience = 15
    patience_counter = 0
   
    for epoch in range(num_epochs):
        # Training
        property_predictor.train()  # Ensure train mode
        train_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_dataloader):
            # Zero gradients before each batch
            optimizer.zero_grad()
            
            images = batch['image'].to(device)
            text = batch['text']
            properties = batch['properties'].to(device)
            targets = batch['target'].to(device).float()
            targets_normalized = (targets - targets_mean) / targets_std
            properties_normalized = (properties - properties_mean) / properties_std
            attention_context = get_attention_context(properties_normalized, device)
            #attention_context= torch.eye(properties_normalized.size(1)).unsqueeze(0).to(device)
            # Reshape tabular_embeddings: [B, F * projected_dim] -> [B, F, projected_dim]
            batch_size = properties_normalized.size(0)
            num_features = properties_normalized.size(1)
            predictions = property_predictor(properties_normalized, attention_context)  # [batch, 1]
            
            # In training loop, add L1 penalty on interaction features
            # interaction_weights = property_predictor.property_head[0].weight[:, D:] 
            # l1_penalty = 0.01 * torch.sum(torch.abs(interaction_weights))
            loss = criterion(predictions.squeeze(-1), targets_normalized) 
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(property_predictor.parameters(), max_norm=1.0)
            # torch.nn.utils.clip_grad_norm_(model.context_encoder.parameters(), max_norm=1.0)
            optimizer.step()
        
            train_loss += loss.item()
            num_batches += 1
            # if num_batches>2:
            #     break
        scheduler.step()
        # print(f"Scheduler step: {scheduler.get_last_lr()}")
        avg_train_loss = train_loss / num_batches if num_batches > 0 else 0.0
        train_losses.append(avg_train_loss)
        # print(f"Epoch {epoch+1}/{num_epochs}")
        # print(f"Train Loss: {avg_train_loss:.6f}")
    
        # Validation
        property_predictor.eval()
        val_loss = 0.0
        all_predictions = []
        all_targets = []
            
        with torch.no_grad():
            model.eval()
            for batch in val_dataloader:
                # Clear attention stores before each forward pass to only process current batch
                images = batch['image'].to(device)
                text = batch['text']
                properties = batch['properties'].to(device)
                targets = batch['target'].to(device).float()
                targets_normalized = (targets - targets_mean) / targets_std
                properties_normalized = (properties - properties_mean) / properties_std
                # Clear attention stores before each forward pass to only process current batch
                attention_context = get_attention_context(properties_normalized, device)
                # attent_context= torch.eye(properties_normalized.size(1)).unsqueeze(0).to(device)
                # Reshape tabular_embeddings: [B, F * projected_dim] -> [B, F, projected_dim]
                batch_size = properties_normalized.size(0)
                num_features = properties_normalized.size(1)
                # projected_dim = tabular_embeddings.size(1) // num_features
                # tabular_embeddings = tabular_embeddings.view(batch_size, num_features, projected_dim)
                # Predict properties
                predictions_normalized = property_predictor(properties_normalized, attention_context)
                predictions = predictions_normalized*targets_std + targets_mean
                
                loss = criterion(predictions.squeeze(-1), targets_normalized)
                val_loss += loss.item()
                
                all_predictions.append(predictions.squeeze(-1).cpu().numpy())
                all_targets.append(targets.cpu().numpy())
    
        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)
        val_r2 = r2_score(all_targets, all_predictions)
        val_rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
        val_mae = mean_absolute_error(all_targets, all_predictions)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_r2 = val_r2
            best_rmse = val_rmse
            best_mae = val_mae
            patience_counter = 0
            #torch.save(property_predictor.state_dict(), f'./Results/Checkpoint/property_predictor_tit_{epoch+1}.pth')
            print(f"Best Val Loss: {best_val_loss:.6f} | Best R²: {best_r2:.4f} | Best RMSE: {best_rmse:.4f} | Best MAE: {best_mae:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
    return model, property_predictor


def evaluate_supervised_property_predictor_with_weights( model, property_predictor, test_dataloader,properties_mean,properties_std, targets_mean,targets_std, attention_context_store, device):
    print("\n" + "="*80)
    print("FINAL EVALUATION ON TEST SET")
    print("="*80)
    property_predictor.eval()
    test_predictions = []
    test_targets = []
    with torch.no_grad():
        for batch in test_dataloader:
            attention_context_store.clear()
            images = batch['image'].to(device)  # Not used but kept for compatibility
            text = batch['text']
            properties = batch['properties'].to(device)
            targets = batch['target'].to(device).float()
            properties_normalized = (properties - properties_mean) / properties_std
            targets_normalized = (targets - targets_mean) / targets_std
            # Use get_attention_context helper function
            def get_attention_context_eval(properties_normalized, device):
                with torch.no_grad():
                    model.eval()
                    attention_context_store.clear()
                    _ = model.context_encoder(properties_normalized)
                    attention_context = aggregate_attention(attention_context_store)
                    attention_context = topk_sparsify(attention_context, k=3).float().to(device)
                    return attention_context
            
            attention_context = get_attention_context_eval(properties_normalized, device)
            # attention_context= torch.eye(properties_normalized.size(1)).unsqueeze(0).to(device)
            # Reshape tabular_embeddings: [B, F * projected_dim] -> [B, F, projected_dim]
            batch_size = properties_normalized.size(0)
            num_features = properties_normalized.size(1)
            #tabular_embeddings = tabular_embeddings.view(batch_size, num_features, projected_dim)
            predictions_normalized = property_predictor(properties_normalized, attention_context).squeeze(-1)
            predictions = (predictions_normalized * targets_std) + targets_mean
            test_predictions.append(predictions.cpu().numpy())
            test_targets.append(targets.cpu().numpy())
    
    
    test_predictions = np.concatenate(test_predictions)
    test_targets = np.concatenate(test_targets)
    
    # S'assurer que les deux arrays ont la même forme (1D)
    test_predictions = test_predictions.flatten()
    test_targets = test_targets.flatten()
    
    test_r2 = r2_score(test_targets, test_predictions)
    test_rmse = np.sqrt(mean_squared_error(test_targets, test_predictions))
    test_mae = mean_absolute_error(test_targets, test_predictions)
    
    print(f"Test R²: {test_r2:.4f} | RMSE: {test_rmse:.4f} | MAE: {test_mae:.4f}")
    return test_predictions

def in_domain_evaluation(unsupervised_model, device):

    batch_size = 128
    num_epochs_supervised = 10
    
    csv_file_train = "/home/abhibhatt/Pycharm_Projects/Data_composites/composite_data_train.csv"
    csv_file_test = "/home/abhibhatt/Pycharm_Projects/Data_composites/composite_data_test.csv"
    image_folder = "/home/abhibhatt/Pycharm_Projects/Data_composites/microstructure_images"  
    text_folder_train = "/home/abhibhatt/Pycharm_Projects/Data_composites/text_reports"
    text_folder_test = "/home/abhibhatt/Pycharm_Projects/Data_composites/text_reports_test"

    for target_name in ['fracture', 'yield', 'elastic modulus', 'elongation', 'tangent modulus']: 
        property_predictor = SupervisedPropertyPredictor(
        vit_hidden_dim=64,  # Use projected_dim (128) from tabular JEPA model
        num_properties=1, 
        hidden_dim=64, 
        dropout=0.4
        )
        print(f"Training for target: {target_name}")
        # Training dataset with Gaussian noise augmentation
        full_train_dataset = CompositeImageTextDataset(
            csv_file_train, image_folder, text_folder_train, target_name,
            num_augmentations=5, mode='all',
            add_gaussian_noise=True, noise_std=0.01
        )
        # Test dataset without noise (for evaluation)
        test_dataset = CompositeImageTextDataset(
            csv_file_test, image_folder, text_folder_test, target_name,
            mode='random',
            add_gaussian_noise=False
        )

        train_dataset, val_dataset = split_data(full_train_dataset, train_ratio=0.80, random_seed=1234)
        # train_dataset, test_dataset = split_data(dataset, train_ratio=0.8, random_seed=42)
        generator = torch.Generator().manual_seed(1234)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, generator=generator)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, generator=generator)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, generator=generator)
    
        targets_list = []
        properties_list = []
        i=0
        for b in train_dataloader:
            properties_list.append(b['properties'].float())
            targets_list.append(b['target'].float())
        
        properties_list = np.concatenate(properties_list)
        properties_mean = torch.tensor(properties_list.mean(axis=0), dtype=torch.float32)  # ✅ [7] - mean per column
        properties_std = torch.tensor(properties_list.std(axis=0), dtype=torch.float32)    # ✅ [7] - std per column
        properties_std = torch.clamp(properties_std, min=1e-8)
        properties_mean = properties_mean.to(unsupervised_model.device)
        properties_std = properties_std.to(unsupervised_model.device)
        targets_list = np.concatenate(targets_list)
        targets_mean = targets_list.mean()
        targets_std = targets_list.std()
    
        # feature_names = ['f', 'c', 'v', 'r', 't', 'w', 'dir']
        # attention_context_store = AttentionStore()
        # attention_target_store = AttentionStore()
        # attention_context_store = register_attention_hooks(unsupervised_model.context_encoder)
        # attention_target_store = register_attention_hooks(unsupervised_model.target_encoder)
        # # viz_attention(unsupervised_model, 
        # test_dataloader, properties_mean, properties_std, feature_names, attention_context_store, attention_target_store) 
        unsup_model, property_predictor = fine_tune_supervised_property_predictor(
            model=unsupervised_model,
            property_predictor = property_predictor,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,  
            properties_mean=properties_mean,
            properties_std=properties_std,
            targets_mean=targets_mean,
            targets_std=targets_std,
            num_epochs=num_epochs_supervised,
            lr=1e-4,
            device=unsupervised_model.device
        )
        # torch.save(property_predictor.state_dict(), f'./Results/Checkpoint/property_predictor_combined_jepa_tit_no_cf_{target_name}.pth')
        # # property_predictor = load_cf_predictor(f'./Results/Checkpoint/property_predictor_combined_jepa_tit_no_cf_{target_name}.pth', property_predictor, unsupervised_model.device)
        evaluate_supervised_property_predictor(unsupervised_model, property_predictor, test_dataloader,properties_mean,properties_std, targets_mean, targets_std, unsupervised_model.device)
        

def ood_evaluation(unsupervised_model, device):
    txt_file_QH = "/home/abhibhatt/Pycharm_Projects/Data_composites/text_reports_QH"
    csv_file_QH = "/home/abhibhatt/Pycharm_Projects/Data_composites/composite_data_QH.csv"
    image_folder_QH = "/home/abhibhatt/Pycharm_Projects/Data_composites/microstructure_images_QH"
    feature_columns = ['ID','NumFibers','MMA','Vf','A11','A12','A13','A22','A23','A33']
    num_epochs_supervised = 15
    batch_size = 128
    for target_name in ['yield', 'elongation']:
        property_predictor = SupervisedPropertyPredictor(
        vit_hidden_dim=128,  # Use projected_dim (128) from tabular JEPA model
        num_properties=1, 
        hidden_dim=128, 
        dropout=0.2
        )
        print(f"Training for target: {target_name}")
        # Training dataset with Gaussian noise augmentation
        QH_dataset = CompositeImageTextDataset(
            csv_file_QH, image_folder_QH, txt_file_QH, target_name,
            num_augmentations=1, mode='QH_random',
            add_gaussian_noise=False, noise_std=0.01
        )
        fulltrain_dataset, test_dataset = split_data(QH_dataset, train_ratio=0.8, random_seed=42)
        train_dataset, val_dataset = split_data(fulltrain_dataset, train_ratio=0.80, random_seed=1234)
        
        generator = torch.Generator().manual_seed(1234)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, generator=generator)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, generator=generator)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, generator=generator)

        targets_list = []
        properties_list = []
        i=0
        for b in train_dataloader:
            properties_list.append(b['properties'].float())
            targets_list.append(b['target'].float())
        
        properties_list = np.concatenate(properties_list)
        properties_mean = torch.tensor(properties_list.mean(axis=0), dtype=torch.float32)  # ✅ [7] - mean per column
        properties_std = torch.tensor(properties_list.std(axis=0), dtype=torch.float32)    # ✅ [7] - std per column
        properties_std = torch.clamp(properties_std, min=1e-8)
        properties_mean = properties_mean.to(unsupervised_model.device)
        properties_std = properties_std.to(unsupervised_model.device)
        targets_list = np.concatenate(targets_list)
        targets_mean = targets_list.mean()
        targets_std = targets_list.std()
    
        # feature_names = ['f', 'c', 'v', 'r', 't', 'w', 'dir']
        # attention_context_store = AttentionStore()
        # attention_target_store = AttentionStore()
        # attention_context_store = register_attention_hooks(unsupervised_model.context_encoder)
        # attention_target_store = register_attention_hooks(unsupervised_model.target_encoder)
        # # viz_attention(unsupervised_model, 
        # test_dataloader, properties_mean, properties_std, feature_names, attention_context_store, attention_target_store) 
        unsup_model, property_predictor = fine_tune_supervised_property_predictor(
            model=unsupervised_model,
            property_predictor = property_predictor,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,  
            properties_mean=properties_mean,
            properties_std=properties_std,
            targets_mean=targets_mean,
            targets_std=targets_std,
            num_epochs=num_epochs_supervised,
            lr=1e-4,
            device=unsupervised_model.device
        )
        # torch.save(property_predictor.state_dict(), f'./Results/Checkpoint/property_predictor_combined_jepa_tit_no_cf_{target_name}.pth')
        # # property_predictor = load_cf_predictor(f'./Results/Checkpoint/property_predictor_combined_jepa_tit_no_cf_{target_name}.pth', property_predictor, unsupervised_model.device)
        evaluate_supervised_property_predictor(unsupervised_model, property_predictor, test_dataloader,properties_mean,properties_std, targets_mean, targets_std, unsupervised_model.device)

def in_domain_graph_evaluation(unsupervised_model, device):

    batch_size = 128
    num_epochs_supervised = 30
    csv_file_train = "/home/abhibhatt/Pycharm_Projects/Data_composites/composite_data_train.csv"
    csv_file_test = "/home/abhibhatt/Pycharm_Projects/Data_composites/composite_data_test.csv"
    image_folder = "/home/abhibhatt/Pycharm_Projects/Data_composites/microstructure_images"  
    text_folder_train = "/home/abhibhatt/Pycharm_Projects/Data_composites/text_reports"
    text_folder_test = "/home/abhibhatt/Pycharm_Projects/Data_composites/text_reports_test"
    # Adjacency matrix for the feature graph - should be [7, 7] for 7 features
    hidden_dim = 128
    out_dim = 1
    attention_context_store = AttentionStore()
    attention_context_store = register_attention_hooks(unsupervised_model.context_encoder)
    #for target_name in ['yield', 'elongation', 'tangent modulus', 'elastic modulus', 'fracture']:
    for target_name in ['yield']:
        feature_graph_predictor = FeatureGraphPredictor(
            in_dim=7, hidden_dim=hidden_dim, out_dim=out_dim, 
            num_layers=1, dropout=0.1, device=unsupervised_model.device
        )
        print(f"Training for target: {target_name}")
        # Training dataset with Gaussian noise augmentation
        full_train_dataset = CompositeImageTextDataset(
            csv_file_train, image_folder, text_folder_train, target_name,
            num_augmentations=5, mode='all',
            add_gaussian_noise=True, noise_std=0.01
        )
        # Test dataset without noise (for evaluation)
        test_dataset = CompositeImageTextDataset(
            csv_file_test, image_folder, text_folder_test, target_name,
            mode='random',
            add_gaussian_noise=False
        )

        train_dataset, val_dataset = split_data(full_train_dataset, train_ratio=0.80, random_seed=1234)
        # train_dataset, test_dataset = split_data(dataset, train_ratio=0.8, random_seed=42)
        generator = torch.Generator().manual_seed(1234)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, generator=generator)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, generator=generator)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, generator=generator)
    
        targets_list = []
        properties_list = []
        i=0
        for b in train_dataloader:
            properties_list.append(b['properties'].float())
            targets_list.append(b['target'].float())
        
        properties_list = np.concatenate(properties_list)
        properties_mean = torch.tensor(properties_list.mean(axis=0), dtype=torch.float32)  # ✅ [7] - mean per column
        properties_std = torch.tensor(properties_list.std(axis=0), dtype=torch.float32)    # ✅ [7] - std per column
        properties_std = torch.clamp(properties_std, min=1e-8)
        properties_mean = properties_mean.to(unsupervised_model.device)
        properties_std = properties_std.to(unsupervised_model.device)
        targets_list = np.concatenate(targets_list)
        targets_mean = targets_list.mean()
        targets_std = targets_list.std()

        unsup_model, property_predictor = fine_tune_supervised_property_predictor_with_weights(
            model=unsupervised_model,
            property_predictor = feature_graph_predictor,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,  
            properties_mean=properties_mean,
            properties_std=properties_std,
            targets_mean=targets_mean,
            targets_std=targets_std,
            attention_context_store=attention_context_store,
            num_epochs=num_epochs_supervised,
            lr=1e-3,
            device=unsupervised_model.device
        )

        evaluate_supervised_property_predictor_with_weights(unsupervised_model, feature_graph_predictor, test_dataloader,properties_mean,properties_std, targets_mean, targets_std, attention_context_store, unsupervised_model.device)


# Ensemble evaluation function
def ensemble_predict(model, predictors, dataloader, properties_mean, properties_std, 
                    targets_mean, targets_std, ensemble_method, device):
    """Make ensemble predictions"""
    all_predictions = []
    all_targets = []
    
    model.eval()
    for pred in predictors:
        pred.eval()
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            text = batch['text']
            properties = batch['properties'].to(device)
            targets = batch['target'].to(device).float()
            
            properties_normalized = (properties - properties_mean) / properties_std
            targets_normalized = (targets - targets_mean) / targets_std
            
            # Get embeddings once
            forward_output = model(images, properties_normalized, text, apply_masking=False)
            tabular_embeddings = forward_output['original_embeddings']['tabular']
            
            # Get predictions from all bootstrap models
            bootstrap_preds = []
            for predictor in predictors:
                pred = predictor(tabular_embeddings)
                bootstrap_preds.append(pred)
            
            # Stack: [num_bootstrap, batch, 1]
            bootstrap_preds = torch.stack(bootstrap_preds, dim=0)
            
            # Ensemble: mean or median
            if ensemble_method == 'mean':
                ensemble_pred = bootstrap_preds.mean(dim=0)  # [batch, 1]
            elif ensemble_method == 'median':
                ensemble_pred = bootstrap_preds.median(dim=0)[0]  # [batch, 1]
            
            # Denormalize
            predictions = ensemble_pred * targets_std + targets_mean
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    
    return all_predictions, all_targets



def bootstrap_fine_tune_supervised_property_predictor(
    model, property_predictor_class, train_dataset, val_dataloader, test_dataloader,
    properties_mean, properties_std, targets_mean, targets_std,
    num_bootstrap=5, num_epochs=10, lr=1e-4, device='cuda:0',
    bootstrap_ratio=0.8,  # Sample size relative to original (1.0 = same size)
    ensemble_method='mean'):  # 'mean' or 'median'):

    from torch.utils.data import DataLoader, SubsetRandomSampler
    import numpy as np
    trained_predictors = []
    n_train = len(train_dataset)
    bootstrap_size = int(n_train * bootstrap_ratio)


    print(f"Bootstrap fine-tuning: {num_bootstrap} models, {bootstrap_size} samples each")
    
    for b in range(num_bootstrap):
        print(f"\n{'='*60}")
        print(f"Training Bootstrap Model {b+1}/{num_bootstrap}")
        print(f"{'='*60}")
        
        # Create bootstrap sample (with replacement)
        bootstrap_indices = np.random.choice(n_train, size=bootstrap_size, replace=True)
        bootstrap_sampler = SubsetRandomSampler(bootstrap_indices)
        bootstrap_dataloader = DataLoader(
            train_dataset, 
            batch_size=128, 
            sampler=bootstrap_sampler,
            num_workers=0
        )
        # Create new property predictor for this bootstrap
        property_predictor = SupervisedPropertyPredictor(
        vit_hidden_dim=128,  # Use projected_dim (128) from tabular JEPA model
        num_properties=1, 
        hidden_dim=64, 
        dropout=0.4
        )
        property_predictor = property_predictor.to(device)
       
       
        
        # Fine-tune on bootstrap sample
        _, trained_predictor = fine_tune_supervised_property_predictor(
            model=model,
            property_predictor=property_predictor,
            train_dataloader=bootstrap_dataloader,
            val_dataloader=val_dataloader,
            properties_mean=properties_mean,
            properties_std=properties_std,
            targets_mean=targets_mean,
            targets_std=targets_std,
            num_epochs=num_epochs,
            lr=lr,
            device=device
        )
        
        trained_predictors.append(trained_predictor)    
    # Evaluate ensemble on test set
    print(f"\n{'='*60}")
    print(f"Ensemble Evaluation ({ensemble_method})")
    print(f"{'='*60}")
    
    test_predictions, test_targets = ensemble_predict(
        model, trained_predictors, test_dataloader,
        properties_mean, properties_std, targets_mean, targets_std, ensemble_method, device
    )
    
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    test_r2 = r2_score(test_targets, test_predictions)
    test_rmse = np.sqrt(mean_squared_error(test_targets, test_predictions))
    test_mae = mean_absolute_error(test_targets, test_predictions)
    
    print(f"Ensemble Test R²: {test_r2:.4f} | RMSE: {test_rmse:.4f} | MAE: {test_mae:.4f}")
    
    return trained_predictors, test_predictions, test_targets   

def run_bootstrap_ood(unsupervised_model, device):
    # txt_file_QH = "/home/abhibhatt/Pycharm_Projects/Data_composites/text_reports_QH"
    txt_file_QH = "/home/abhibhatt/Pycharm_Projects/Data_composites/text"
    # csv_file_QH = "/home/abhibhatt/Pycharm_Projects/Data_composites/composite_data_QH.csv"
    csv_file_QH = "/home/abhibhatt/Pycharm_Projects/Data_composites/Composite_Data_QH1.csv"
    # image_folder_QH = "/home/abhibhatt/Pycharm_Projects/Data_composites/microstructure_images_QH"
    image_folder_QH = "/home/abhibhatt/Pycharm_Projects/Data_composites/microstructure"
    # feature_columns = ['ID','NumFibers','MMA','Vf','A11','A12','A13','A22','A23','A33']
    num_epochs_supervised = 15
    batch_size = 128


    # for target_name in ['yield', 'elongation']:
    for target_name in ['Young_modulus']:
        property_predictor = SupervisedPropertyPredictor(
        vit_hidden_dim=128,  # Use projected_dim (128) from tabular JEPA model
        num_properties=1, 
        hidden_dim=128, 
        dropout=0.2
        )
        print(f"Training for target: {target_name}")
        # Training dataset with Gaussian noise augmentation
        QH_dataset = CompositeImageTextDataset(
            csv_file_QH, image_folder_QH, txt_file_QH, target_name,
            num_augmentations=1, mode='QH_random_1',
            add_gaussian_noise=False, noise_std=0.01
        )
        fulltrain_dataset, test_dataset = split_data(QH_dataset, train_ratio=0.8, random_seed=42)
        train_dataset, val_dataset = split_data(fulltrain_dataset, train_ratio=0.80, random_seed=1234)
        
        generator = torch.Generator().manual_seed(1234)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, generator=generator)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, generator=generator)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, generator=generator)

        targets_list = []
        properties_list = []
        i=0
        for b in train_dataloader:
            properties_list.append(b['properties'].float())
            targets_list.append(b['target'].float())
        
        properties_list = np.concatenate(properties_list)
        properties_mean = torch.tensor(properties_list.mean(axis=0), dtype=torch.float32)  # ✅ [7] - mean per column
        properties_std = torch.tensor(properties_list.std(axis=0), dtype=torch.float32)    # ✅ [7] - std per column
        properties_std = torch.clamp(properties_std, min=1e-8)
        properties_mean = properties_mean.to(unsupervised_model.device)
        properties_std = properties_std.to(unsupervised_model.device)
        targets_list = np.concatenate(targets_list)
        targets_mean = targets_list.mean()
        targets_std = targets_list.std()

        bootstrap_predictors, test_preds, test_targets = bootstrap_fine_tune_supervised_property_predictor(
        model=unsupervised_model,
        property_predictor_class=SupervisedPropertyPredictor,
        train_dataset=train_dataset,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        properties_mean=properties_mean,
        properties_std=properties_std,
        targets_mean=targets_mean,
        targets_std=targets_std,
        num_bootstrap=1,  # Train 5 models
        num_epochs=10,
        lr=1e-4,
        device=unsupervised_model.device,
        bootstrap_ratio=0.8,  # Same size as original
        ensemble_method='median'  # or 'median'
        )
    return bootstrap_predictors, test_preds, test_targets

def run_bootstrap_indomain(unsupervised_model, device):

    batch_size = 128
    csv_file_train = "/home/abhibhatt/Pycharm_Projects/Data_composites/composite_data_train.csv"
    csv_file_test = "/home/abhibhatt/Pycharm_Projects/Data_composites/composite_data_test.csv"
    image_folder = "/home/abhibhatt/Pycharm_Projects/Data_composites/microstructure_images"  
    text_folder_train = "/home/abhibhatt/Pycharm_Projects/Data_composites/text_reports"
    text_folder_test = "/home/abhibhatt/Pycharm_Projects/Data_composites/text_reports_test"

    for target_name in ['fracture', 'yield', 'elastic modulus', 'elongation', 'tangent modulus']: 
    # for target_name in ['elongation']: 
        full_train_dataset = CompositeImageTextDataset(
            csv_file_train, image_folder, text_folder_train, target_name,
            num_augmentations=5, mode='all',
            add_gaussian_noise=True, noise_std=0.01
        )
    
        test_dataset = CompositeImageTextDataset(
        csv_file_test, image_folder, text_folder_test, target_name,
        mode='random',
        add_gaussian_noise=False
        )
        train_dataset, val_dataset = split_data(full_train_dataset, train_ratio=0.80, random_seed=1234)
        generator = torch.Generator().manual_seed(1234)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, generator=generator)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, generator=generator)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, generator=generator)

        
        targets_list = []
        properties_list = []
        
        for b in train_dataloader:
            properties_list.append(b['properties'].float())
            targets_list.append(b['target'].float())
        
        properties_list = np.concatenate(properties_list)
        properties_mean = torch.tensor(properties_list.mean(axis=0), dtype=torch.float32)  # ✅ [7] - mean per column
        properties_std = torch.tensor(properties_list.std(axis=0), dtype=torch.float32)    # ✅ [7] - std per column
        properties_std = torch.clamp(properties_std, min=1e-8)
        properties_mean = properties_mean.to(unsupervised_model.device)
        properties_std = properties_std.to(unsupervised_model.device)
        targets_list = np.concatenate(targets_list)
        targets_mean = targets_list.mean()
        targets_std = targets_list.std()

        bootstrap_predictors, test_preds, test_targets = bootstrap_fine_tune_supervised_property_predictor(
        model=unsupervised_model,
        property_predictor_class=SupervisedPropertyPredictor,
        train_dataset=train_dataset,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        properties_mean=properties_mean,
        properties_std=properties_std,
        targets_mean=targets_mean,
        targets_std=targets_std,
        num_bootstrap=5,  # Train 5 models
        num_epochs=15,
        lr=1e-4,
        device=unsupervised_model.device,
        bootstrap_ratio=0.8,  # Same size as original
        ensemble_method='mean'  # or 'median'
        )
    return bootstrap_predictors, test_preds, test_targets

def plot_geodesic(unsupervised_model, csv_file, image_folder, text_folder, properties_mean, properties_std, s, t, title, mode = 'random'):
    batch_size = 64
    target_name = 'elongation'
    dataset = CompositeImageTextDataset(
        csv_file, image_folder, text_folder, target_name,
        num_augmentations=1, mode=mode,
        add_gaussian_noise=False, noise_std=0.01
    )

    generator = torch.Generator().manual_seed(1234)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, generator=generator)
    z_embeddings_list = []
    unsupervised_model.eval()
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(unsupervised_model.device)  # Not used but kept for compatibility
            text = batch['text']
            properties = batch['properties'].to(unsupervised_model.device)
            targets = batch['target'].to(unsupervised_model.device).float()
            properties_normalized = (properties - properties_mean) / properties_std
            # # Get tabular embeddings from the model
            forward_output = unsupervised_model(images, properties_normalized, text, apply_masking=False)
            tabular_embeddings = forward_output['original_embeddings']['tabular']  # [batch, hidden_dim]
            z_embeddings = tabular_embeddings.detach().cpu().numpy()
            z_embeddings_list.append(z_embeddings)
    print(f"Number of z_embeddings: {len(z_embeddings_list)}")
    z_embeddings = np.concatenate(z_embeddings_list, axis=0)
    print(f"Number of z_embeddings: {z_embeddings.shape}")
    visualize_geodesic_between_regimes(z_embeddings, 0, 1, k=12, mutual=True, z_s=s, z_t=t, title=title)

def fetch_regimes(unsupervised_model, dataloader, properties_mean, properties_std):
    z_embeddings_list = []
    unsupervised_model.eval()
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(unsupervised_model.device)  # Not used but kept for compatibility
            text = batch['text']
            properties = batch['properties'].to(unsupervised_model.device)
            targets = batch['target'].to(unsupervised_model.device).float()
            properties_normalized = (properties - properties_mean) / properties_std
            # # Get tabular embeddings from the model
            forward_output = unsupervised_model(images, properties_normalized, text, apply_masking=False)
            tabular_embeddings = forward_output['original_embeddings']['tabular']  # [batch, hidden_dim]
            z_embeddings = tabular_embeddings.detach().cpu().numpy()
            z_embeddings_list.append(z_embeddings)
    z_embeddings = np.concatenate(z_embeddings_list, axis=0)
    # s,t = get_regimes(z_embeddings)
    return z_embeddings

def fine_tune_supervised_property_predictor_geodesic(model, property_predictor, train_dataloader, val_dataloader,
                                           properties_mean, properties_std, targets_mean, targets_std, weights_by_id, residuals,
                                           num_epochs=10, lr=1e-4, device='cuda:0'):
   
    hidden_dim = 64
    for p in model.context_encoder.parameters():
        p.requires_grad = False
    for p in model.target_encoder.parameters():
        p.requires_grad = False
    for p in model.masked_column_predictor.parameters():
        p.requires_grad = False
    for p in property_predictor.parameters():
        p.requires_grad = True
    # Get tabular embeddings from the base model
 
    trainable_params = list(property_predictor.parameters())
  
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        weight_decay=1e-2
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.MSELoss()
    beta_r = 0.1
    model.train()
    property_predictor.train()
    # Training loop
    best_val_loss = float('inf')
    best_r2 = -float('inf')
    train_losses = []
    val_losses = []
    val_r2_scores = []
    targets_list = []
    patience = 10
    patience_counter = 0
    for epoch in range(num_epochs):
        # Training
        train_loss = 0.0
        num_batches = 0
        w_normalized_min_list = []
        w_normalized_med_list = []
        w_normalized_max_list = []
        w_max_list = []
        ess_list = []
        for batch_idx, batch in enumerate(train_dataloader):

            images = batch['image'].to(device)
            text = batch['text']
            properties = batch['properties'].to(device)
            targets = batch['target'].to(device).float()
            sample_id = batch['id']
            has_image = batch['has_image']
            has_text = batch['has_text']
            if not has_image.any() or not has_text.any():
                print(f"Warning: Sample has no image or text")
                model.use_conditioning = False
            properties_normalized = (properties - properties_mean) / properties_std
            targets_normalized = (targets - targets_mean) / targets_std
            if torch.is_tensor(sample_id):
                sample_id_list = sample_id.detach().cpu().tolist()
            else:
                sample_id_list = sample_id
            
            sample_id_list = [min(int(i), len(weights_by_id)-1) for i in sample_id_list]

            
            tabular_embeddings = get_tabular_embeddings(model,images, properties_normalized, text)
            tabular_embeddings = tabular_embeddings.to(device)
            #augmented_properties = get_augmented_properties(model, images, properties_normalized, text)
            r = residuals[sample_id_list]
            r_normalized = beta_r*(r - r.mean(dim=0)) / (r.std(dim=0)+1e-12)
            r_normalized = r_normalized.to(device)
            features = torch.cat([tabular_embeddings, r_normalized], dim=1)
            features = features.to(device)
            predictions = property_predictor(features)  # [batch, 1]
            loss_mse = criterion(predictions.squeeze(-1), targets_normalized)
            if torch.isnan(loss_mse).any() or torch.isinf(loss_mse).any():
                raise ValueError("Warning: loss_mse contains NaN/Inf")
                
              # build weight vector [B]
         
            # w = weights_by_id[sample_id_list]
            w = torch.tensor([float(weights_by_id[i]) for i in sample_id_list],
                              device=device, dtype=torch.float32)  # [B]
            #w_normalized = torch.softmax(w, dim=0)
            w_normalized = w/(w.sum()+1e-12)
            loss = (w_normalized*loss_mse).sum()/(w_normalized.sum()+1e-12)
            # with torch.no_grad():
            #     w_normalized_min_list.append(w_normalized.min().item())
            #     w_normalized_med_list.append(w_normalized.median().item())
            #     w_normalized_max_list.append(w_normalized.max().item())
            #     ess = (w_normalized.sum()**2 / (w_normalized.pow(2).sum() + 1e-12)).item()  # effective sample size'
            #     ess_list.append(ess)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(property_predictor.parameters(), max_norm=1.0)
            # torch.nn.utils.clip_grad_norm_(model.context_encoder.parameters(), max_norm=1.0)
            optimizer.step()
        
            train_loss += loss.item()
            num_batches += 1
            # if num_batches>2:
            #     break

        scheduler.step()
        # print(f"Scheduler step: {scheduler.get_last_lr()}")
        # print("w min/med/max: ", np.mean(w_normalized_min_list), np.mean(w_normalized_med_list), np.mean(w_normalized_max_list))
        # print("ess: ", np.mean(ess_list))
        avg_train_loss = train_loss / num_batches if num_batches > 0 else 0.0
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.6f}")
    
        # Validation
        property_predictor.eval()
        val_loss = 0.0
        all_predictions = []
        all_targets = []
            
        with torch.no_grad():
            model.eval()
            for batch in val_dataloader:
                images = batch['image'].to(device)
                text = batch['text']
                properties = batch['properties'].to(device)
                targets = batch['target'].to(device).float()
                sample_id = batch['id']
                if not has_image.any() or not has_text.any():
                    print(f"Warning: Sample {sample_id} has no image or text in validation set")
                    model.use_conditioning = False
                if torch.is_tensor(sample_id):
                    sample_id_list = sample_id.detach().cpu().tolist()
                else:
                    sample_id_list = sample_id
                
                sample_id_list = [min(int(i), len(weights_by_id)-1) for i in sample_id_list]
                r = residuals[sample_id_list]
                r_normalized = beta_r* (r - r.mean(dim=0)) / (r.std(dim=0)+1e-12)
                r_normalized = r_normalized.to(device)
                
                
                properties_normalized = (properties - properties_mean) / properties_std
                targets_normalized = (targets - targets_mean) / targets_std
                tabular_embeddings = get_tabular_embeddings(model,images, properties_normalized, text)
                features = torch.cat([tabular_embeddings, r_normalized], dim=1)
                
                #augmented_properties = get_augmented_properties(model, images, properties_normalized, text)
                # Predict properties
                predictions_normalized = property_predictor(features) 
                predictions = predictions_normalized*targets_std + targets_mean
                
                loss = criterion(predictions.squeeze(-1), targets_normalized)
                val_loss += loss.item()
                
                all_predictions.append(predictions.squeeze(-1).cpu().numpy())
                all_targets.append(targets.cpu().numpy())
    
        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)
        val_r2 = r2_score(all_targets, all_predictions)
        val_rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
        val_mae = mean_absolute_error(all_targets, all_predictions)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_r2 = val_r2
            best_rmse = val_rmse
            best_mae = val_mae
            patience_counter = 0
            #torch.save(property_predictor.state_dict(), f'./Results/Checkpoint/property_predictor_tit_{epoch+1}.pth')
            print(f"Best Val Loss: {best_val_loss:.6f} | Best R²: {best_r2:.4f} | Best RMSE: {best_rmse:.4f} | Best MAE: {best_mae:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    return model, property_predictor

def evaluate_supervised_property_predictor_geodesic( model, property_predictor, test_dataloader,properties_mean, properties_std, targets_mean,targets_std, weights, residuals, device):
    print("\n" + "="*80)
    print("FINAL EVALUATION ON TEST SET")
    print("="*80)
    beta_r = 0.1

    model.eval()
    property_predictor.eval()
    test_predictions = []
    test_targets = []
    with torch.no_grad():
        for batch in test_dataloader:
            images = batch['image'].to(device)  # Not used but kept for compatibility
            text = batch['text']
            properties = batch['properties'].to(device)
            targets = batch['target'].to(device).float()
            sample_id = batch['id']
            has_image = batch['has_image']
            has_text = batch['has_text']
            if not has_image.any() or not has_text.any():
                print(f"Warning: Sample {sample_id} has no image or text in test set")
                model.use_conditioning = False

            if torch.is_tensor(sample_id):
                sample_id_list = sample_id.detach().cpu().tolist()
            else:
                sample_id_list = sample_id
                
            sample_id_list = [min(int(i), len(weights)-1) for i in sample_id_list]
            r = residuals[sample_id_list]
            r_normalized = beta_r* (r - r.mean(dim=0)) / (r.std(dim=0)+1e-12)
            r_normalized = r_normalized.to(device)
            properties_normalized = (properties - properties_mean) / properties_std
            # # Get tabular embeddings from the model
            forward_output = model(images, properties_normalized, text, apply_masking=False)
            tabular_embeddings = forward_output['original_embeddings']['tabular']  # [batch, hidden_dim]
            features = torch.cat([tabular_embeddings, r_normalized], dim=1)
            features = features.to(device)
            predictions_normalized = property_predictor(features).squeeze(-1)
            predictions = (predictions_normalized * targets_std) + targets_mean
            test_predictions.append(predictions.cpu().numpy())
            test_targets.append(targets.cpu().numpy())
    
    test_predictions = np.concatenate(test_predictions)
    test_targets = np.concatenate(test_targets)
    
    # S'assurer que les deux arrays ont la même forme (1D)
    test_predictions = test_predictions.flatten()
    test_targets = test_targets.flatten()
    
    test_r2 = r2_score(test_targets, test_predictions)
    test_rmse = np.sqrt(mean_squared_error(test_targets, test_predictions))
    test_mae = mean_absolute_error(test_targets, test_predictions)
    
    print(f"Test R²: {test_r2:.4f} | RMSE: {test_rmse:.4f} | MAE: {test_mae:.4f}")
    return test_predictions

def main():
    # log_file = setup_logging()
    # logging.info("="*80)
    # logging.info("CROSS-MODAL PREDICTOR FOR COMPOSITE MATERIALS (PHYSICS-INFORMED)")
    # logging.info("="*80)
    
    # Configuration
    csv_file_train = "/home/abhibhatt/Pycharm_Projects/Data_composites/composite_data_pretrain.csv"
    # csv_file_test = "/home/abhibhatt/Pycharm_Projects/Data_composites/composite_data_test.csv"
    image_folder = "/home/abhibhatt/Pycharm_Projects/Data_composites/microstructure_images"  
    text_folder_train = "/home/abhibhatt/Pycharm_Projects/Data_composites/text_reports_pretrain"
    # text_folder_test = "/home/abhibhatt/Pycharm_Projects/Data_composites/text_reports_test"
    batch_size = 128
    num_epochs_unsupervised = 25
    
    # Loading models
        # Configuration pour charger un modèle pré-entraîné
    load_pretrained_model = False
    pretrained_checkpoint_path = './Results/Checkpoint/checkpoint_composite_jepa_tit_cftr_10_3masked.pth'

    text_model_name = 'm3rg-iitd/matscibert'    
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    matscibert = AutoModel.from_pretrained(text_model_name)
    matscibert_hidden_dim = matscibert.config.hidden_size
    target_name = 'elongation'
    full_train_dataset = CompositeImageTextDataset(
        csv_file_train, image_folder, text_folder_train, target_name,
        num_augmentations=5, mode='all',
        add_gaussian_noise=True, noise_std=0.01,
        pretrain=True
    )
    # train_dataset, val_dataset = split_data(full_train_dataset, train_ratio=0.80, random_seed=1234)
    # train_dataset, test_dataset = split_data(dataset, train_ratio=0.8, random_seed=42)
    generator = torch.Generator().manual_seed(1234)
    pretrain_dataloader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, generator=generator)

    unsupervised_model = MAEStyleJEPACompositeModel(
        tabular_input_dim=7,
        combined_dim=64,
        enable_contrastive=False,
        use_masked_tabular_modeling=True,
        hidden_dim=64,
        num_masked_columns=3,
        tokenizer=tokenizer,
        matscibert=matscibert,
    )
    
    # Phase 1: Unsupervised training with physics (ou chargement d'un modèle pré-entraîné)
    if load_pretrained_model:
        print("\n" + "="*80)
        print("PHASE 1: LOADING PRETRAINED UNSUPERVISED MODEL")
        print("="*80)
        try:
            unsupervised_model, loaded_epoch, loaded_loss = load_unsupervised_model(
                pretrained_checkpoint_path,
                unsupervised_model,
                device=unsupervised_model.device
            )
            print(f" Skipping unsupervised training - using pretrained model from epoch {loaded_epoch}")
            
        except Exception as e:
            print(f"   ❌ Error loading pretrained model: {e}")
            print(f"   → Falling back to training from scratch...")
            load_pretrained_model = False
            exit(0)
    
    if not load_pretrained_model:
        print("Training unsupervised model from scratch")
        target_name = 'elongation'
        full_train_dataset = CompositeImageTextDataset(
            csv_file_train, image_folder, text_folder_train, target_name,
            num_augmentations=5, mode='all',
            add_gaussian_noise=True, noise_std=0.01,
            pretrain=True
        )
        # train_dataset, val_dataset = split_data(full_train_dataset, train_ratio=0.80, random_seed=1234)
        # train_dataset, test_dataset = split_data(dataset, train_ratio=0.8, random_seed=42)
        generator = torch.Generator().manual_seed(1234)
        train_dataloader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, generator=generator)
        # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, generator=generator)
        # val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, generator=generator)
    
        train_unsupervised(
            unsupervised_model, 
            train_dataloader, 
            num_epochs=num_epochs_unsupervised,
            lr=1e-4, 
        )
    print("Unsupervised training completed")
    properties_mean, properties_std, targets_mean, targets_std = get_properties_mean_std(pretrain_dataloader, unsupervised_model.device)
    train_dataloader, val_dataloader, test_dataloader = get_loaders()# batch_size=batch_size, QH_flag=False
    # qh_train_dataloader, qh_val_dataloader, qh_test_dataloader = get_loaders(QH_flag=True)

    bootstrap_predictors, test_preds, test_targets = run_bootstrap_indomain(unsupervised_model, unsupervised_model.device)
    # bootstrap_predictors, test_preds, test_targets = run_bootstrap_ood(unsupervised_model, unsupervised_model.device)
    # s_o,t_o = fetch_regimes(unsupervised_model, train_dataloader, properties_mean, properties_std)
    # print("Plotting geodesic for train data")
    # plot_geodesic(unsupervised_model=unsupervised_model, 
    # csv_file="/home/abhibhatt/Pycharm_Projects/Data_composites/composite_data_train.csv",
    # image_folder="/home/abhibhatt/Pycharm_Projects/Data_composites/microstructure_images",
    # text_folder="/home/abhibhatt/Pycharm_Projects/Data_composites/text_reports",
    # properties_mean=properties_mean,
    # properties_std=properties_std,
    # s=s_o,
    # t=t_o,
    # title='Latent geodesic (graph shortest path) for Training Data')
    # print("Plotting geodesic for test data")
    # plot_geodesic(unsupervised_model=unsupervised_model, 
    # csv_file="/home/abhibhatt/Pycharm_Projects/Data_composites/composite_data_test.csv",
    # image_folder="/home/abhibhatt/Pycharm_Projects/Data_composites/microstructure_images",
    # text_folder="/home/abhibhatt/Pycharm_Projects/Data_composites/text_reports_test",
    # properties_mean=properties_mean,
    # properties_std=properties_std,
    # s=s_o,
    # t=t_o,
    # title='Latent geodesic (graph shortest path) for Test Data1')
    # print("Plotting geodesic for OOD data")
    # plot_geodesic(unsupervised_model=unsupervised_model, 
    # csv_file="/home/abhibhatt/Pycharm_Projects/Data_composites/composite_data_QH.csv",
    # image_folder="/home/abhibhatt/Pycharm_Projects/Data_composites/microstructure_images_QH",
    # text_folder="/home/abhibhatt/Pycharm_Projects/Data_composites/text_reports_QH",
    # properties_mean=properties_mean,
    # properties_std=properties_std,
    # s=s_o,
    # t=t_o,
    # title='Geodesic_jepa_tit_cftr_indomain_ood',
    # mode='QH_random')



    # 
    # #in_domain_evaluation(unsupervised_model, unsupervised_model.device)
        

    # print("="*80)
    # print("In-domain evaluation")
    # print("="*80)
    # in_domain_evaluation(unsupervised_model, unsupervised_model.device) 
    # print("="*80)
    # print("OOD evaluation")
    # print("="*80)
    # ood_evaluation(unsupervised_model, unsupervised_model.device) 
    # property_predictor_cf = SupervisedPropertyPredictor(
    #     vit_hidden_dim=128,  # Use projected_dim (128) from tabular JEPA model
    #     num_properties=1, 
    #     hidden_dim=64, 
    #     dropout=0.2
    # )
# cf_predictor = generate_counterfactuals(unsupervised_model, property_predictor_cf, train_dataloader,val_dataloader, properties_mean, properties_std, targets_mean, targets_std, num_epochs=num_epochs_supervised, lr=1e-4, k=100, device=unsupervised_model.device)
    # # #torch.save(cf_predictor.state_dict(), f'./Results/Checkpoint/cf_predictor_combined_jepa_tit_cf_100_{target_name}.pth')
    #evaluate_supervised_property_predictor(unsupervised_model, cf_predictor, test_dataloader, properties_mean,properties_std, targets_mean,targets_std, unsupervised_model.device)
    # unsup_model, property_predictor = fine_tune_supervised_property_predictor_with_weights(
    #     model=unsupervised_model,
    #     property_predictor=property_predictor,
    #     train_dataloader=train_dataloader,
    #     val_dataloader=val_dataloader,  
    #     properties_mean=properties_mean,
    #     properties_std=properties_std,
    #     targets_mean=targets_mean,
    #     targets_std=targets_std,
    #     attention_context_store=attention_context_store,
    #     attention_target_store=attention_target_store,
    #     num_epochs=num_epochs_supervised,
    #     lr=1e-4,
    #     device=unsupervised_model.device
    # )
    # evaluate_supervised_property_predictor_with_weights(unsup_model, property_predictor, test_dataloader,properties_mean,properties_std, targets_mean, targets_std, attention_context_store, attention_target_store, unsupervised_model.device)
    # #final_predictions = ensemble_prediction(unsupervised_model, property_predictor_cf, test_dataloader,properties_mean,properties_std, targets_mean, targets_std, target_name, unsupervised_model.device)
    # z_embeddings_full = fetch_regimes(unsupervised_model, pretrain_dataloader, properties_mean, properties_std)
    # # # z_train = fetch_regimes(unsupervised_model, train_dataloader, properties_mean, properties_std)
    # # # z_test = fetch_regimes(unsupervised_model, test_dataloader, properties_mean, properties_std)
    # z_ood = fetch_regimes(unsupervised_model, qh_train_dataloader, properties_mean, properties_std)
    # # # print("Computing split metrics for train data")
    # # # train_metrics = compute_split_metrics(z_train, z_embeddings_full, A=5, k_graph=12, mutual=True, seed=1234,
    # # #                                     anchor_method="random", knn_for_local=10, orc_mode="fast")
    # # # print("Computing split metrics for test data")
    # # # test_metrics = compute_split_metrics(z_test, z_embeddings_full, A=5, k_graph=12, mutual=True, seed=1234,
    # # #                                    anchor_method="random", knn_for_local=10, orc_mode="fast")
    # # # print("Computing split metrics for OOD data")
    # # # ood_metrics = compute_split_metrics(z_ood, z_embeddings_full, A=20, k_graph=15, mutual=True, seed=1234,
    # # #                                    anchor_method="farthest", knn_for_local=15, orc_mode="fast")
    # # # # print("TRAIN:", {k:v for k,v in train_metrics.items() if k not in ["graph","anchors_idx","ratios"]})
    # # # # print("TEST: ", {k:v for k,v in test_metrics.items()  if k not in ["graph","anchors_idx","ratios"]})
    # # # print("OOD: ", {k:v for k,v in ood_metrics.items()  if k not in ["graph","anchors_idx","ratios"]})
    # # best_anchor_idx, geodesic_dist = find_id_anchor_via_geodesic(z_embeddings_full, z_ood, A=20, k=15, k_attach=5, seed=1234, anchor_method="random")
    # weights,residuals,_ = calc_weights_euclid_to_anchor(z_embeddings_full, z_ood)
    # qh_property_mean, qh_property_std, qh_targets_mean, qh_targets_std = get_properties_mean_std(qh_train_dataloader, unsupervised_model.device)

    # property_predictor = SupervisedPropertyPredictor(
    #     vit_hidden_dim=128,  # Use projected_dim (128) from tabular JEPA model
    #     num_properties=1, 
    #     hidden_dim=64, 
    #     dropout=0.2
    # )
    # num_epochs_supervised = 15
    # unsup_model, property_predictor = fine_tune_supervised_property_predictor_geodesic(
    #     model=unsupervised_model,
    #     property_predictor=property_predictor,
    #     train_dataloader=qh_train_dataloader,
    #     val_dataloader=qh_val_dataloader,
    #     properties_mean=qh_property_mean,
    #     properties_std=qh_property_std,
    #     targets_mean=qh_targets_mean,
    #     targets_std=qh_targets_std,
    #     weights_by_id=weights,
    #     residuals=residuals,
    #     num_epochs=num_epochs_supervised,
    #     lr=1e-4,
    #     device=unsupervised_model.device
    # )
    # evaluate_supervised_property_predictor_geodesic(unsup_model, property_predictor, qh_test_dataloader,qh_property_mean,qh_property_std, qh_targets_mean, qh_targets_std, weights, residuals, unsupervised_model.device)
    
    # # print("Computing CGD")
    # # cgd = compute_CGD(z_train, z_test, z_embeddings_full, A=5, seed=1234, anchor_method="random",k_graph=12, mutual=True)

 
    # # print("CGD:", cgd)


    


if __name__ == "__main__":
    main()

