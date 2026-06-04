import datetime
import itertools
import torch.nn as nn
from transformers import AdamW, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset, random_split
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
from transformers import AutoTokenizer, AutoModel

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
        has_image = True
        has_text = True
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
                raise FileNotFoundError(f"Image {image_path} does not exist!")
            else:
                try:
                    image = Image.open(image_path).convert('RGB')
                    image = self.transform(image)
                except Exception as e:
                    print(f"Error loading image {sample_id}_{aug_idx}: {e}")
                    image = torch.zeros(3, self.image_size, self.image_size)
        
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
                
                try:
                    image = Image.open(image_path).convert('RGB')
                    image = self.transform(image)
                except Exception as e:
                    print(f"Error loading image {sample_id}_{aug_idx}: {e}")
                    image = torch.zeros(3, self.image_size, self.image_size)
            
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
                else:
                    print(f"Error: No images found for sample {sample_id}")
                    image = torch.zeros(3, self.image_size, self.image_size)
            
            elif self.mode == 'QH_random':
                sample_id_fmt = f"{int(sample_id):03d}"
                self.property_columns = ['NumFibers','MMA','Vf','A11','A12','A13','A22']
            
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
            
            elif self.mode == 'Comp1':
                sample_id_fmt = f"{int(sample_id):03d}"
                self.property_columns = ['f_cf','f_cnt','f_gra','f_cu','f_ni','f_resin','Density']
            
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

            elif self.mode == 'Comp2':
                sample_id_fmt = f"{int(sample_id):03d}"
                self.property_columns = ['f1_T300','f1_T700','f1_T800','f1_T1100','f1_pitch','f_epoxy', 'Density']
            
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
                    raise FileNotFoundError(f"Image {image_path} does not exist!")
                
                try:
                    image = Image.open(image_path).convert('RGB')
                    image = self.transform(image)
                except Exception as e:
                    print(f"Error loading image {sample_id}_0: {e}")
                    image = torch.zeros(3, self.image_size, self.image_size)
        
        # Les données tabulaires sont les mêmes pour toutes les images du même échantillon
        text_folder = os.path.join(self.text_folder, f"{sample_id}_report.txt")
        if os.path.exists(text_folder):
            try:
                with open(text_folder, 'r') as file:
                    text = file.read()
                has_text = True
            except Exception as e:
                print(f"Error loading text {sample_id}: {e}")
                text = ""
                has_text = False
        else:
            text = ""
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

def main():
    pass

if __name__ == "__main__":
    main()

