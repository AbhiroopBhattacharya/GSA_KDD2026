"""
JEPA with MAE-style Masked Image Modeling (MIM) using Simple MLP Predictor
Efficiently encodes only visible patches, uses simple MLP for masked patch prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from fusion_model_lightweight_clean import LightweightJEPACompositeModel
from cnn_cndn_tr import LightweightCNNEncoder
from utils import register_attention_hooks, AttentionStore, ResidualMLPHead


class SupervisedPropertyPredictor(nn.Module):
    """
    Supervised property predictor using pre-trained image encoder
    Simple MLP that takes pooled image embeddings for property prediction
    """
    def __init__(self, vit_hidden_dim=768, num_properties=1, hidden_dim=512, dropout=0.1):
        """
        Args:
            vit_hidden_dim: Hidden dimension of image encoder (default: 768)
            num_properties: Number of properties to predict (default: 1)
            hidden_dim: Hidden dimension for the prediction head
            dropout: Dropout rate
        """
        super().__init__()
        self.vit_hidden_dim = vit_hidden_dim
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        num_old_features = 7
        # num_interaction_features = int(num_old_features*(num_old_features-1)/2)
        # Property prediction head
        self.property_head = nn.Sequential(
            # nn.Linear((self.vit_hidden_dim*num_old_features)+num_interaction_features, hidden_dim),
            nn.Linear((self.vit_hidden_dim*num_old_features), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_properties)
        ).to(self.device)
        # self.property_head = ResidualMLPHead(
        #     in_dim=self.vit_hidden_dim*7,
        #     hidden_dim=hidden_dim,
        #     num_properties=num_properties,
        #     dropout=dropout,
        #     device=self.device
        # ).to(self.device)
      
    def forward(self, embeddings, text_context=None):
    
        # Predict properties
        property_predictions = self.property_head(embeddings)  # [batch, num_properties]
        
        return property_predictions

class AttnReturningEncoderLayer(nn.TransformerEncoderLayer):
    def forward(self, src, src_mask=None, is_causal=False, src_key_padding_mask=None):
        # Mostly copied from PyTorch, but with need_weights=True
        x = src

        # Self-attention block
        attn_output, attn_weights = self.self_attn(
            x, x, x,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            is_causal=is_causal,
            average_attn_weights=False,  # keep [B, num_heads, F, F]
        )
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # Feedforward block
        ff = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(ff)
        x = self.norm2(x)

        # Return x (the usual output) and attn_weights
        # Hooks will see this tuple as (attn_output, attn_weights)
        return x, attn_weights

class FeatureWiseTransformerEncoder(nn.Module):
    """
    Feature-wise transformer encoder for tabular data with conditioning.

    Input:  x  [B, F]  (F = num tabular features)
            condition_embeddings: [B, cond_dim] (optional) - concatenated text + image embeddings
    Output: [B, F * projected_dim]
            (one projected_dim-dim embedding per feature, flattened)
    """
    def __init__(
        self,
        num_features: int,
        projected_dim: int,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 32,
        dropout: float = 0.5,
        device: str = "cuda:0",
        condition_dim: int = None,  # Dimension of condition embeddings (text + image)
    ):
        super().__init__()
        self.num_features = num_features
        self.projected_dim = projected_dim
        self.d_model = d_model
        self.device = device
        self.condition_dim = condition_dim
        self.use_conditioning = condition_dim is not None

        # Each scalar feature → d_model-dim token embedding
        # We apply this to each feature independently: [B, F, 1] → [B, F, d_model]
        self.input_proj = nn.Linear(1, d_model)

        # Learnable "feature position" embeddings (one per column)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_features, d_model))
        
        # Condition embedding projection (text + image → d_model)
        if self.use_conditioning:
            self.condition_proj = nn.Linear(condition_dim, d_model)
            # Learnable scaling factor for condition injection
            self.condition_scale = nn.Parameter(torch.tensor(0.01))
            # self.fuse_mlp = nn.Sequential(
            #     nn.Linear(2 * d_model, d_model),
            #     nn.GELU(),
            #     nn.Dropout(dropout),
            #     nn.Linear(d_model, 1)  # -> (w1, w2)
            # )

        # encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=d_model,
        #     nhead=n_heads,
        #     dim_feedforward=dim_feedforward,
        #     dropout=dropout,
        #     batch_first=True,  # so we can use [B, F, d_model]
        #     norm_first=True,
        #     return_attn_weights=True,
        # )

        encoder_layer = AttnReturningEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Project transformer output per feature → projected_dim
        self.out_proj = nn.Linear(d_model, projected_dim)

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     """
    #     x: [B, F] tabular input
    #     returns: [B, F * projected_dim]
    #     """
    #     # [B, F] → [B, F, 1]
    #     x = x.unsqueeze(-1)

    #     # [B, F, 1] → [B, F, d_model]
    #     x = self.input_proj(x)

    #     # Add feature position embeddings
    #     x = x + self.pos_embed  # [1, F, d_model] broadcasts over batch

    #     # Transformer over features
    #     h = self.encoder(x)  # [B, F, d_model]

    #     # Per-feature projection to projected_dim
    #     h = self.out_proj(h)  # [B, F, projected_dim]

    #     # Flatten all feature embeddings to match your current shape
    #     B = h.size(0)
    #     h = h.view(B, self.num_features * self.projected_dim)  # [B, F * projected_dim]
    #     return h
    def forward(self, x: torch.Tensor, condition_embeddings: torch.Tensor = None) -> torch.Tensor:
        """
        x: [B, F] tabular input
        condition_embeddings: [B, cond_dim] (optional) - concatenated text + image embeddings
        returns: [B, F * projected_dim]
        """
        # [B, F] → [B, F, 1]
        x = x.unsqueeze(-1)

        # [B, F, 1] → [B, F, d_model]
        x = self.input_proj(x)

        # Add feature position embeddings
        x = x + self.pos_embed  # [1, F, d_model] broadcasts over batch
        
        # # Add condition embeddings if provided
        if self.use_conditioning and condition_embeddings is not None:
            # Project condition embeddings: [B, cond_dim] → [B, d_model]
            cond_proj = self.condition_proj(condition_embeddings)  # [B, d_model
            x = x + self.condition_scale * cond_proj.unsqueeze(1)  # conditioning # [B, F, d_model]
            # Broadcast and add to all features: [B, d_model] → [B, 1, d_model] → [B, F, d_model]
            # modality availability_factor = has_image.float().mean() * has_text.float().mean()
            # modality_availability_factor = condition_embeddings.float().mean()
            # print("modality availability_factor: ", modality_availability_factor)
            # if modality_availability_factor == 0:
            #     x = x  # no conditioning   # [B, F, d_model]
            # else:
            #     x = x + self.condition_scale * cond_proj.unsqueeze(1)  # conditioning # [B, F, d_model]
        # if self.use_conditioning and condition_embeddings is not None:
        #     c = self.condition_proj(condition_embeddings)          # [B, d_model]
        #     c = c.unsqueeze(1).expand(-1, x.size(1), -1)          # [B, F, d_model]

        #     z = torch.cat([x, c], dim=-1)                         # [B, F, 2*d_model]
        #     # w = self.fuse_mlp(z)                                  # [B, F, 2]
        #     # w = torch.softmax(w, dim=-1)                          # stable, w1+w2=1
        #     # w1 = w[..., 0:1]                                      # [B, F, 1]
        #     # w2 = w[..., 1:2]                                      # [B, F, 1]

        #     # x = w1 * x + w2 *self.condition_scale * c  
        #     g = torch.sigmoid(self.fuse_mlp(z))                                 # [B, F, d_model]
        #     x = x + g * (self.condition_scale * c)
        # if self.use_conditioning and condition_embeddings is not None:
        #     condition_embeddings = F.layer_norm(condition_embeddings, (condition_embeddings.size(-1),))
        #     c = self.condition_proj(condition_embeddings).unsqueeze(1)          # [B, d_model]          # [B, F, d_model]
        #     c = self.condition_scale * c
        #     h = torch.cat([x, c], dim=1)
        # else:
        #     h = x
        # Transformer over features - need to handle tuple returns
        # Since AttnReturningEncoderLayer returns (x, attn_weights),
        # we need to manually iterate through layers
        h = x
        for layer in self.encoder.layers:
            h, _ = layer(h)  # Extract only the tensor, ignore attn_weights
        
        # Apply final norm if TransformerEncoder has one
        if hasattr(self.encoder, 'norm') and self.encoder.norm is not None:
            h = self.encoder.norm(h)
        
        # if self.use_conditioning and condition_embeddings is not None:
        #     h = h[:, 1:, :]
        # Per-feature projection to projected_dim
        h = self.out_proj(h)  # [B, F, projected_dim]

        # Flatten all feature embeddings to match your current shape
        B = h.size(0)
        h = h.view(B, self.num_features * self.projected_dim)  # [B, F * projected_dim]
        return h

class MAEStyleJEPACompositeModel(LightweightJEPACompositeModel):
    """
    JEPA with MAE-style Masked Tabular Modeling using Simple MLP Predictor
    Masks 2 columns randomly out of 7, encodes visible columns with context encoder,
    encodes masked columns with target encoder (updated via EMA), and predicts masked columns.
    """
    
    def __init__(self, 
                 tabular_input_dim=7, combined_dim=256, enable_contrastive=False,
                 use_masked_tabular_modeling=True,
                 hidden_dim=128, num_masked_columns=2,
                 use_conditioning=True, condition_dim=None, tokenizer=None, matscibert=None):
        super().__init__(
            tabular_input_dim=tabular_input_dim,
            combined_dim=combined_dim,
            enable_contrastive=enable_contrastive
        )
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = tokenizer
        self.matscibert = matscibert
        self.use_masked_tabular_modeling = use_masked_tabular_modeling
        self.tabular_input_dim = tabular_input_dim
        self.num_masked_columns = num_masked_columns
        self.num_visible_columns = tabular_input_dim - num_masked_columns
        self.hidden_dim = hidden_dim
        self.projected_dim = 128 # Projected dimension for embeddings
        self.use_conditioning = use_conditioning
    
        # Determine condition dimension if not provided
        if use_conditioning and condition_dim is None:
            # Image: combined_dim (from image_proj)
            # Text: combined_dim (if using MatSciBERT with projection)
            # Default: assume both are combined_dim, so total is 2 * combined_dim
            # But we'll handle dynamic dimension in _extract_condition_embeddings
            condition_dim = 2 * combined_dim  # Max dimension (image + text)
            
        if use_conditioning and matscibert is not None:
            matscibert_hidden_dim = matscibert.config.hidden_size
            self.text_proj = nn.Linear(matscibert_hidden_dim, self.combined_dim).to(self.device)
        else:
            self.text_proj = None
        
        self.condition_dim = condition_dim if use_conditioning else None
        if use_masked_tabular_modeling:
            # Set projected dimension to 128
            # Context encoder: takes all 7 columns, outputs 7 column embeddings
            # Architecture: 7 -> hidden -> 7 * projected_dim
            self.context_encoder = FeatureWiseTransformerEncoder(
                num_features=self.tabular_input_dim,
                projected_dim=self.projected_dim,
                d_model=64,          # tune
                n_heads=4,           # d_model % n_heads == 0
                n_layers=2,          # 1–4 is usually enough
                dim_feedforward=128, # 2–4x d_model
                dropout=0.1,
                device=self.device,
                condition_dim=self.condition_dim,
            ).to(self.device)

            self.target_encoder = FeatureWiseTransformerEncoder(
                num_features=self.tabular_input_dim,
                projected_dim=self.projected_dim,
                d_model=64,
                n_heads=4,
                n_layers=2,
                dim_feedforward=128,
                dropout=0.1,
                device=self.device,
                condition_dim=self.condition_dim,
            ).to(self.device)

            self.decoder = nn.Sequential(
                nn.Linear(self.num_masked_columns * self.projected_dim, self.projected_dim),
                nn.LayerNorm(self.projected_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.projected_dim, self.projected_dim),
                nn.LayerNorm(self.projected_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.projected_dim, self.num_masked_columns)
                ).to(self.device)
    
            # Disable gradients for target encoder (updated via EMA only)
            for param in self.target_encoder.parameters():
                param.requires_grad = False
            # Predictor: takes 5 visible column embeddings, predicts 2 masked column embeddings
            # Input: 5 * projected_dim, Output: 2 * projected_dim
            self.masked_column_predictor = nn.Sequential(
                nn.Linear(self.num_visible_columns * self.projected_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, self.num_masked_columns * self.projected_dim)
            ).to(self.device)
           
           
            # EMA decay factor
            self.target_encoder_ema_decay = 0.998
    
    def _extract_condition_embeddings(self, images, text_data, tokenizer=None, matscibert=None):
        """
        Extract and concatenate text and image embeddings for conditioning.
        
        Args:
            images: [batch, 3, 224, 224] - microstructure images
            text_data: text data (list of strings or tokenized dict)
            tokenizer: Optional tokenizer for text (if using MatSciBERT)
            matscibert: Optional MatSciBERT model for text encoding
        
        Returns:
            condition_embeddings: [batch, condition_dim] - concatenated text + image embeddings
        """
        if not self.use_conditioning:
            return None
        
        batch_size = images.size(0) if isinstance(images, torch.Tensor) else len(images)
        condition_list = []
        
        # Extract image embeddings
        if images is not None:
            images = images.to(self.device)
            for param in self.vit_encoder.parameters():
                param.requires_grad = False
            vit_output = self.vit_encoder(pixel_values=images)
            image_features = vit_output.last_hidden_state  # [batch, seq_len, vit_hidden_size]
            image_embeddings = self.image_proj(image_features)  # [batch, seq_len, combined_dim]
            # Pool image embeddings (mean pooling)
            image_pooled = image_embeddings.mean(dim=1)  # [batch, combined_dim]
            image_pooled = F.layer_norm(image_pooled, (image_pooled.size(-1),))
            condition_list.append(image_pooled)
        
        # Extract text embeddings using MatSciBERT if provided
        if text_data is not None and tokenizer is not None and matscibert is not None:
            # Tokenize text
            if isinstance(text_data, (list, tuple)):
                inputs = tokenizer(text_data, return_tensors="pt", padding=True, truncation=True, max_length=512)
            else:
                inputs = text_data  # Assume already tokenized
            
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            matscibert_output = matscibert(**inputs)
            text_features = matscibert_output.last_hidden_state  # [batch, seq_len, hidden_dim]
            # Pool text embeddings (mean pooling)
            text_pooled = text_features[:,0,:]
            #text_pooled = text_features.mean(dim=1)  # [batch, hidden_dim]
            # Project to combined_dim if needed
            if text_pooled.shape[1] != self.combined_dim:
                # Create a simple projection if dimension doesn't match     
                text_pooled = self.text_proj(text_pooled)  # [batch, combined_dim]
            text_pooled = F.layer_norm(text_pooled, (text_pooled.size(-1),))
            condition_list.append(text_pooled)
        
        # Concatenate all condition embeddings
        if len(condition_list) > 0:
            condition_embeddings = torch.cat(condition_list, dim=1)  # [batch, actual_condition_dim]
            # If dimension doesn't match expected, pad or project
            if condition_embeddings.shape[1] != self.condition_dim:
                if condition_embeddings.shape[1] < self.condition_dim:
                    # Pad with zeros if smaller
                    padding = torch.zeros(
                        condition_embeddings.shape[0], 
                        self.condition_dim - condition_embeddings.shape[1],
                        device=condition_embeddings.device
                    )
                    condition_embeddings = torch.cat([condition_embeddings, padding], dim=1)
                else:
                    # Project if larger (shouldn't happen, but handle it)
                    if not hasattr(self, 'condition_dim_proj'):
                        self.condition_dim_proj = nn.Linear(condition_embeddings.shape[1], self.condition_dim).to(self.device)
                    condition_embeddings = self.condition_dim_proj(condition_embeddings)
            return condition_embeddings
        else:
            # If no conditions, return None (encoder will work without conditioning)
            print("no conditions")
            return None
    
    def _mask_tabular_columns(self, tabular_data):
        """
        Randomly mask 2 columns out of 7 in tabular data
        
        Args:
            tabular_data: [batch, 7] - tabular input data
        
        Returns:
            visible_columns: [batch, 5] - visible columns
            masked_columns: [batch, 2] - masked columns
            visible_indices: [5] - indices of visible columns
            masked_indices: [2] - indices of masked columns
        """
        batch_size = tabular_data.size(0)
        device = tabular_data.device
        
        # Randomly select 2 columns to mask (different for each sample in batch)
        all_indices = torch.arange(self.tabular_input_dim, device=device)  # [0, 1, 2, 3, 4, 5, 6]
        
        # For each sample, randomly select 2 columns to mask
        masked_indices_list = []
        visible_indices_list = []
        visible_columns_list = []
        masked_columns_list = []
        
        for i in range(batch_size):
            # Random permutation of column indices
            perm = torch.randperm(self.tabular_input_dim, device=device)
            masked_indices = perm[:self.num_masked_columns].sort()[0]  # Sort for consistency
            visible_indices = perm[self.num_masked_columns:].sort()[0]
            
            masked_indices_list.append(masked_indices)
            visible_indices_list.append(visible_indices)
            
            # Extract visible and masked columns
            visible_columns_list.append(tabular_data[i, visible_indices])
            masked_columns_list.append(tabular_data[i, masked_indices])
        
        # Stack to create batch tensors
        visible_columns = torch.stack(visible_columns_list, dim=0)  # [batch, 5]
        masked_columns = torch.stack(masked_columns_list, dim=0)  # [batch, 2]
        
        # Use first sample's indices for consistency (or could use per-sample if needed)
        visible_indices = visible_indices_list[0]
        masked_indices = masked_indices_list[0]
        
        return visible_columns, masked_columns, visible_indices, masked_indices
    
    def forward(self, images, tabular_data, text_data, apply_masking=True, mask_ratio=None):
        """
        Forward pass with masked tabular modeling
        
        Args:
            images: [batch, 3, 224, 224] - microstructure images
            tabular_data: [batch, 7] - tabular input data
            text_data: text data (list of strings or tokenized dict) - text descriptions
            apply_masking: Whether to apply column masking
            mask_ratio: Not used, kept for compatibility
        
        Returns:
            dict with embeddings, predictions, and MTM (Masked Tabular Modeling) outputs
        """
        batch_size = tabular_data.size(0)
        tabular_data = tabular_data.to(self.device)
        
        # Extract condition embeddings (text + image)
        condition_embeddings = None
        if self.use_conditioning:
            # Try to get tokenizer and matscibert from model attributes if available
            tokenizer = getattr(self, 'tokenizer', None)
            matscibert = getattr(self, 'matscibert', None)
            matscibert = matscibert.to(self.device)
            for param in matscibert.parameters():
                param.requires_grad = False
            condition_embeddings = self._extract_condition_embeddings(images, text_data, tokenizer, matscibert)
        
        if apply_masking and self.use_masked_tabular_modeling and self.training:
            # Step 1: Get mask indices (which columns to mask)
            _, _, visible_indices, masked_indices = self._mask_tabular_columns(tabular_data)
            # print("visible indices: ", visible_indices)
            # print("masked indices: ", masked_indices)
            # Step 2: Context encoder - encode all 7 columns
            mod_tabular_data = tabular_data.clone()
            mod_tabular_data[:, masked_indices] = 0.0
            context_all_encoded = self.context_encoder(mod_tabular_data, condition_embeddings)  # [batch, 7 * projected_dim]
            # Reshape to [batch, 7, projected_dim]
            context_all_encoded = context_all_encoded.view(batch_size, self.tabular_input_dim, self.projected_dim)
            
            # Step 3: Keep only visible columns from context encoder
            visible_col_embeddings = context_all_encoded[:, visible_indices, :]  # [batch, 5, projected_dim]
            # Flatten for predictor input
            visible_col_embeddings_flat = visible_col_embeddings.view(batch_size, -1)  # [batch, 5 * projected_dim]
            
            # Step 4: Target encoder - encode all 7 columns
            target_all_encoded = self.target_encoder(tabular_data, condition_embeddings)  # [batch, 7 * projected_dim]
            # Reshape to [batch, 7, projected_dim]
            target_all_encoded = target_all_encoded.view(batch_size, self.tabular_input_dim, self.projected_dim)
            
            # Step 5: Keep only masked columns from target encoder (ground truth)
            masked_gt_encoded = target_all_encoded[:, masked_indices, :]  # [batch, 2, projected_dim]
            
            # Step 6: Predictor - takes 5 visible embeddings, predicts 2 masked embeddings
            masked_predictions_flat = self.masked_column_predictor(visible_col_embeddings_flat)  # [batch, 2 * projected_dim]
            predicted_columns_raw = self.decoder(masked_predictions_flat)
            masked_columns_raw = tabular_data[:, masked_indices]
            masked_columns_raw_normalized = torch.zeros_like(masked_columns_raw)
            for col_idx in range(masked_columns_raw.shape[1]):
                col = masked_columns_raw[:,col_idx]
                masked_columns_raw_normalized[:,col_idx] = (col - col.mean()) / col.std()
            # Reshape to [batch, 2, projected_dim]
            masked_predictions = masked_predictions_flat.view(batch_size, self.num_masked_columns, self.projected_dim)
            
            # Store MTM outputs for loss computation
            mtm_outputs = {
                'masked_predictions': masked_predictions,  # [batch, 2, projected_dim] - Predictions from predictor
                'masked_gt_encoded': masked_gt_encoded,  # [batch, 2, projected_dim] - Target encoded (with stop grad in loss)
                'visible_col_embeddings': visible_col_embeddings,  # [batch, 5, projected_dim] - Visible columns from context encoder
                'context_all_encoded': context_all_encoded,  # [batch, 7, projected_dim] - All columns from context encoder
                'masked_indices': masked_indices,  # Indices of masked columns
                'visible_indices': visible_indices,  # Indices of visible columns
                'masked_columns_raw': masked_columns_raw_normalized,  # Masked columns raw
                'predicted_columns_raw': predicted_columns_raw  # Predicted columns raw

            }
            
            # For downstream tasks, aggregate all 7 columns from context encoder
            # Use mean pooling or learned aggregation
            tabular_embeddings = context_all_encoded.mean(dim=1)  # [batch, projected_dim]
            
        else:
            # No masking - encode all 7 columns for downstream inference using context encoder
            context_all_encoded = self.context_encoder(tabular_data, condition_embeddings)  # [batch, 7 * projected_dim]
            # Reshape to [batch, 7, projected_dim]
            all_col_embeddings = context_all_encoded.view(batch_size, self.tabular_input_dim, self.projected_dim)
            # Aggregate all columns (mean pooling)
            #tabular_embeddings = all_col_embeddings.mean(dim=1)  # [batch, projected_dim]
            tabular_embeddings = all_col_embeddings.view(batch_size, -1)
            mtm_outputs = None
        
        # Return embeddings compatible with parent class interface
        original_embeddings = {
            'image': None,  # Not used in tabular JEPA
            'tabular': tabular_embeddings  # Tabular embeddings
        }
       
        return {
            'original_embeddings': original_embeddings,
            'mim_outputs': mtm_outputs  # MTM outputs for loss computation (renamed for compatibility)
        }
    
    def compute_jepa_losses(self, forward_output):
        """
        Compute JEPA MIM losses
        """
        original = forward_output.get('original_embeddings', {})
        mim_outputs = forward_output.get('mim_outputs')
        
        losses = {}        
        # 2. MASKED TABULAR MODELING LOSS (MTM - MAE-style)
        if mim_outputs is not None and mim_outputs.get('masked_predictions') is not None:
            masked_pred = mim_outputs['masked_predictions']  # [batch, 2, projected_dim]
            masked_gt_encoded = mim_outputs['masked_gt_encoded']  # [batch, 2, projected_dim]
            predicted_columns_raw = mim_outputs['predicted_columns_raw']  # [batch, 2]
            masked_columns_raw = mim_outputs['masked_columns_raw']  # [batch, 2]
            
            # Apply stop gradient on target encoder output (prevents collapse)
            # The predictor must learn to match the target encoder's representation
            masked_gt_encoded = masked_gt_encoded.detach()
            
            # Compute MSE loss: predictor output vs target encoder output
            # Flatten to [batch, 2 * projected_dim] for loss computation
            masked_pred_flat = masked_pred.view(masked_pred.size(0), -1)  # [batch, 2 * projected_dim]
            masked_gt_flat = masked_gt_encoded.view(masked_gt_encoded.size(0), -1)  # [batch, 2 * projected_dim]
            predicted_columns_raw_flat = predicted_columns_raw.view(predicted_columns_raw.size(0), -1)  # [batch, 2]
            masked_columns_raw_flat = masked_columns_raw.view(masked_columns_raw.size(0), -1)  # [batch, 2]
            mtm_loss = F.mse_loss(masked_pred_flat, masked_gt_flat)
            reconstruction_loss = F.mse_loss(predicted_columns_raw_flat, masked_columns_raw_flat)
            # Cosine similarity: compute per masked column, then average
            # masked_pred: [batch, 2, projected_dim], masked_gt_encoded: [batch, 2, projected_dim]
            cos_sim_per_col = F.cosine_similarity(masked_pred, masked_gt_encoded, dim=-1)  # [batch, 2]
            loss_cosine = cos_sim_per_col.mean().item()
            
            losses['loss_cosine'] = loss_cosine
            losses['masked_image_modeling'] = mtm_loss  # Keep same key for compatibility
            losses['masked_tabular_modeling'] = mtm_loss  # New key for clarity
            losses['reconstruction_loss'] = reconstruction_loss
        else:
            losses['masked_image_modeling'] = torch.tensor(0.0, device=self.device)
            losses['masked_tabular_modeling'] = torch.tensor(0.0, device=self.device)
            losses['loss_cosine'] = torch.tensor(0.0, device=self.device)
            losses['reconstruction_loss'] = torch.tensor(0.0, device=self.device)
       
        losses['total_weighted_loss'] = losses['masked_tabular_modeling'] + 0.01 * losses['reconstruction_loss']
        return losses
    
    def get_trainable_parameters(self):
        """
        Get trainable parameters, excluding target encoder (updated via EMA)
        """
        trainable = []
        for name, param in self.named_parameters():
            # Exclude target encoder from trainable parameters
            if 'target_encoder' not in name and param.requires_grad:
                trainable.append(param)
        return trainable
    
    def update_target_encoder_ema(self):
        """
        Update target encoder weights using Exponential Moving Average (EMA)
        The target encoder is updated with the context encoder's weights to prevent collapse.
        This follows the JEPA/MAE approach where the target encoder is a slowly-updating copy.
        
        Both encoders have identical architecture, so all parameters can be updated.
        """
        if not self.use_masked_tabular_modeling:
            return
        
        # Update target encoder with context encoder weights
        # Both have identical architecture (same input/output dimensions)
        with torch.no_grad():
            # Update all parameters with EMA
            for target_param, context_param in zip(
                self.target_encoder.parameters(),
                self.context_encoder.parameters()
            ):
                # EMA update: target = decay * target + (1 - decay) * context
                # This ensures the target encoder slowly tracks the context encoder
                target_param.data.mul_(self.target_encoder_ema_decay).add_(
                    context_param.data, alpha=1 - self.target_encoder_ema_decay
                )


class SupervisedTransformerPredictor(nn.Module):
    """
    Supervised property predictor using pre-trained image encoder
    Simple MLP that takes pooled image embeddings for property prediction
    """
    def __init__(self, vit_hidden_dim=768, num_properties=1, hidden_dim=512, dropout=0.1):
        """
        Args:
            vit_hidden_dim: Hidden dimension of image encoder (default: 768)
            num_properties: Number of properties to predict (default: 1)
            hidden_dim: Hidden dimension for the prediction head
            dropout: Dropout rate
        """
        super().__init__()
        self.vit_hidden_dim = vit_hidden_dim
        self.tabular_input_dim = 7
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.transformer_encoder = FeatureWiseTransformerEncoder(
            num_features=7,
            projected_dim=64,
            d_model=64,
            n_heads=1,
            n_layers=2,
            dim_feedforward=128,
            dropout=0.5,
        )
        # # Property prediction head
        # self.tabular_encoder = nn.Sequential(
        #         nn.Linear(self.tabular_input_dim, hidden_dim),
        #         nn.LayerNorm(hidden_dim),
        #         nn.GELU(),
        #         nn.Dropout(0.3),
        #         nn.Linear(hidden_dim, hidden_dim),
        #         nn.LayerNorm(hidden_dim),
        #         nn.GELU(),
        #         nn.Dropout(0.3),
        #         nn.Linear(hidden_dim, self.tabular_input_dim * self.vit_hidden_dim)
        # ).to(self.device)
        self.property_head = nn.Sequential(
            nn.Linear((self.vit_hidden_dim)*7, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_properties)
        ).to(self.device)
        
    def forward(self, input_data, text_context=None):
    
        # Predict properties
        encoded_embeddings = self.tabular_encoder(input_data)  # [batch, 7 * 64]
        property_predictions = self.property_head(encoded_embeddings)  # [batch, num_properties]
        
        return property_predictions





def decode_masked_columns(model, masked_columns):
    """
    Decode masked columns using the decoder
    """
    model.eval()
    decoder = nn.Sequential(
            nn.Linear(self.num_masked_columns * self.projected_dim, self.projected_dim),
            nn.LayerNorm(self.projected_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.projected_dim, self.projected_dim),
            nn.LayerNorm(self.projected_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.projected_dim, self.num_masked_columns)
            ).to(self.device)
    
    _, _, visible_indices, masked_indices = model._mask_tabular_columns(properties)

    optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr, weight_decay=0.01)
    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(train_dataloader):
            properties = batch['properties'].to(device)  # [batch, 7]
            text = batch['text']
            batch_size = properties.size(0)
            context_all_encoded = model.context_encoder(properties)  # [batch, 7 * projected_dim]
            context_all_encoded = context_all_encoded.view(batch_size, model.tabular_input_dim, model.projected_dim)  # [batch, 7, projected_dim]
            visible_columns = context_all_encoded[:, visible_indices, :]  # [batch, 2, projected_dim]
            predicted_columns_latent = model.masked_column_predictor(visible_columns)  # [batch, 2, projected_dim]
            predicted_columns_raw = decoder(predicted_columns_latent)
            reconstruction_loss = nn.MSELoss()(predicted_columns_raw, properties[:, masked_indices, :])
            optimizer.zero_grad()
            reconstruction_loss.backward()
            optimizer.step()
    return decoder

def generate_counterfactuals(model, property_predictor, train_dataloader, val_dataloader, properties_mean, properties_std, targets_mean, targets_std,
                                                   num_epochs=10, lr=1e-4, k=5, 
                                                   kl_temperature=1.0, device='cuda:0'):
    """
    Generate counterfactuals using pretrained JEPA model and optimize KL divergence.
    
    Process:
    1. For each sample, mask 2 columns randomly
    2. Feed masked data through pretrained JEPA to get embeddings
    3. Use masked_column_predictor to predict masked column embeddings
    4. Reconstruct full 7-column embedding (5 original + 2 predicted)
    5. Repeat k times to get k counterfactuals per sample
    6. Fine-tune property_predictor to minimize L2 loss
    
    Args:
        model: Pretrained MAEStyleJEPACompositeModel
        property_predictor: SupervisedPropertyPredictor to fine-tune
        train_dataloader: DataLoader for training data
        num_epochs: Number of fine-tuning epochs
        lr: Learning rate
        k: Number of counterfactuals to generate per sample
        kl_temperature: Temperature for KL divergence (softens distributions)
        device: Device to use
    
    Returns:
        Fine-tuned property_predictor
    """
    model.train()  
    property_predictor.train()  # Only train property predictor
    for p in model.context_encoder.parameters():
        p.requires_grad = False
    for p in model.target_encoder.parameters():
        p.requires_grad = False
    for p in model.masked_column_predictor.parameters():
        p.requires_grad = False
    for p in property_predictor.parameters():
        p.requires_grad = True
    # Optimizer for property predictor only
    optimizer = torch.optim.AdamW(
        property_predictor.parameters(),
        lr=lr,
        weight_decay=1e-1
    )
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.MSELoss()
    weight_cf_loss =0.5
    patience = 10
    patience_counter = 0
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        total_combined_loss = 0.0
        total_target_loss = 0.0
        total_cf_loss = 0.0
        num_batches = 0
        
        for batch in train_dataloader:
            properties = batch['properties'].to(device)  # [batch, 7]
            text = batch['text']
            batch_size = properties.size(0)

            targets = batch['target'].to(device)  # [batch, 1]
            # for col_idx in range(properties.size(1)):
            #     properties[:, col_idx] = (properties[:, col_idx] - properties[:, col_idx].mean()) / properties[:, col_idx].std()
            properties_normalized = (properties - properties_mean) / properties_std
            targets_normalized = (targets - targets_mean) / targets_std
            # Step 1: Get original predictions (using all 7 columns)
            with torch.no_grad():
                model.eval()
                forward_output_original = model(None, properties_normalized, text, apply_masking=False)
                original_embeddings = forward_output_original['original_embeddings']['tabular']  
            model.train()
            # Original predictions
            original_preds = property_predictor(original_embeddings)# [batch, 1]
            
            # Step 2: Generate k counterfactuals per sample
            all_counterfactual_preds = []
            
            for cf_idx in range(k):
                # Randomly mask 2 columns for each sample
                _, _, visible_indices, masked_indices = model._mask_tabular_columns(properties_normalized)
                
                # Step 1: Context encoder - encode all 7 columns
                target_all_encoded = model.target_encoder(properties_normalized) 
                context_all_encoded =model.context_encoder(properties_normalized) # [batch, 7 * projected_dim]
                context_all_encoded = context_all_encoded.view(batch_size, model.tabular_input_dim, model.projected_dim)  # [batch, 7, projected_dim]
                target_all_encoded = target_all_encoded.view(batch_size, model.tabular_input_dim, model.projected_dim)  # [batch, 7, projected_dim]
                
                # Step 2: Keep only visible columns from context encoder
                visible_col_embeddings = context_all_encoded[:, visible_indices, :]  # [batch, 5, projected_dim]
                #visible_target_embeddings = target_all_encoded[:, visible_indices, :]  # [batch, 2, projected_dim]
                visible_col_embeddings_flat = visible_col_embeddings.view(batch_size, -1)  # [batch, 5 * projected_dim]
                
                # Step 3: Predict masked column embeddings using predictor
                masked_pred_flat = model.masked_column_predictor(visible_col_embeddings_flat)  # [batch, 2 * projected_dim]
                masked_pred_embeddings = masked_pred_flat.view(batch_size, model.num_masked_columns, model.projected_dim)  # [batch, 2, projected_dim]
                # predicted_columns_raw = decoder(masked_pred_embeddings)

                # Step 4: Reconstruct full 7-column embedding
                # Create full embedding tensor with all 7 columns
                all_reconstructed_embeddings = target_all_encoded.clone()
                
                # Fill in visible columns
                #all_reconstructed_embeddings[:, visible_indices, :] = visible_target_embeddings
                # Fill in predicted masked columns
                all_reconstructed_embeddings[:, masked_indices, :] = masked_pred_embeddings
                # for col_idx in range(all_reconstructed_embeddings.size(1)):
                #     all_reconstructed_embeddings[:, col_idx, :] = (all_reconstructed_embeddings[:, col_idx, :] - all_reconstructed_embeddings[:, col_idx, :].mean()) / all_reconstructed_embeddings[:, col_idx, :].std()
                
                # Step 5: Aggregate all 7 columns for downstream prediction
                #counterfactual_embedding = all_reconstructed_embeddings.mean(dim=1)  # [batch, projected_dim]
                counterfactual_embedding = all_reconstructed_embeddings.view(batch_size, -1)  # [batch, 7 * projected_dim]
                # Get counterfactual predictions
                cf_preds = property_predictor(counterfactual_embedding)  # [batch, 1]
                all_counterfactual_preds.append(cf_preds)
            
            # Stack all counterfactual predictions: [batch, k, 1]
            
            all_counterfactual_preds = torch.stack(all_counterfactual_preds, dim=1)  # [batch, k, 1]
          
            # Average counterfactual predictions over k samples
            #avg_cf_preds = all_counterfactual_preds.mean(dim=1)  # [batch, 1]
            
            target_loss = criterion(original_preds.squeeze(-1), targets_normalized)
            # print("Only counterfactual predictions")
            #cf_loss = criterion(original_preds.detach(), avg_cf_preds)
            cf_loss = ((all_counterfactual_preds - original_preds.detach().unsqueeze(1))**2).mean()
             
            #cf_loss = calculate_kl_divergence(predictions=all_counterfactual_preds, targets=original_preds,batch_size= batch_size, kl_temperature=0.8, device=device)
            # print("Only counterfactual predictions")
            #cf_loss = criterion(original_preds.detach(), avg_cf_preds)
            combined_loss = target_loss + weight_cf_loss *cf_loss
            # ensemble_var = all_counterfactual_preds.var(dim=1)   # [batch, 1]
            # entropy_loss = torch.log(ensemble_var + 1e-8).mean()
            # combined_loss = cf_loss + lambda_entropy * entropy_loss 
            #combined_loss = target_loss + weight_cf_loss *cf_loss        
            # Step 4: Backward pass
            optimizer.zero_grad()
            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(property_predictor.parameters(), max_norm=1.0)
            optimizer.step()
            #total_target_loss += target_loss.item()
            #total_cf_loss += cf_loss.item()
            total_combined_loss += combined_loss.item()
            num_batches += 1
        avg_loss = total_combined_loss /num_batches
        #avg_target_loss = total_target_loss /num_batches
        #avg_cf_loss = total_cf_loss /num_batches
        #print(f"  Epoch {epoch+1}/{num_epochs} Avg Loss: {avg_loss:.6f} (Target: {avg_target_loss:.6f}, CF: {avg_cf_loss:.6f})")
        print(f"  Epoch {epoch+1}/{num_epochs} Avg Loss: {avg_loss:.6f} ")
    

        property_predictor.eval()
        val_loss = 0.0
        all_val_predictions = []
        all_val_targets = []
        
        with torch.no_grad():
            model.eval()
            for batch in val_dataloader:
                images = batch['image'].to(device)
                text = batch['text']
                properties = batch['properties'].to(device)
                # for col_idx in range(properties.size(1)):
                #     properties[:, col_idx] = (properties[:, col_idx] - properties[:, col_idx].mean()) / properties[:, col_idx].std()
                properties_normalized = (properties - properties_mean) / properties_std
                targets = batch['target'].to(device).float()
                targets_normalized = (targets - targets_mean) / targets_std
                forward_output = model(None, properties_normalized, text, apply_masking=False)
                tabular_embeddings = forward_output['original_embeddings']['tabular']  # [batch, hidden_dim]
                # Predict properties
                predictions_normalized = property_predictor(tabular_embeddings)
                predictions = predictions_normalized*targets_std + targets_mean
                
                loss = criterion(predictions.squeeze(-1), targets_normalized)
                val_loss += loss.item()
                
                all_val_predictions.append(predictions.cpu().numpy())
                all_val_targets.append(targets.cpu().numpy())
    
            all_val_predictions = np.concatenate(all_val_predictions)
            all_val_targets = np.concatenate(all_val_targets)
            val_r2 = r2_score(all_val_targets, all_val_predictions)
            val_rmse = np.sqrt(mean_squared_error(all_val_targets, all_val_predictions))
            val_mae = mean_absolute_error(all_val_targets, all_val_predictions)
            print(f"Val Loss: {val_loss:.6f} | Val R²: {val_r2:.4f} | Val RMSE: {val_rmse:.4f} | Val MAE: {val_mae:.4f}")
    
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # torch.save(property_predictor.state_dict(), f'./Results/Checkpoint/property_predictor_tit_{epoch+1}.pth')
                print(f"Best Val Loss: {best_val_loss:.6f} | Best R²: {val_r2:.4f} | Best RMSE: {val_rmse:.4f} | Best MAE: {val_mae:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
            # # Compute variance of counterfactual predictions (uncertainty estimate)
            # cf_preds_var = all_counterfactual_preds.var(dim=1)  # [batch, 1]
            # # Add small epsilon to avoid division by zero
            # cf_preds_var = cf_preds_var + 1e-6
            
            # # Option 1: Gaussian KL divergence
            # # KL(N(μ1, σ1) || N(μ2, σ2)) = log(σ2/σ1) + (σ1² + (μ1-μ2)²) / (2σ2²) - 0.5
            # # We'll use fixed variance for original (or compute from batch)
            # original_var = torch.ones_like(original_preds) * 0.1  # Fixed variance for original
            
            # # KL divergence: KL(original || counterfactual)
            # kl_loss_gaussian = (
            #     torch.log(cf_preds_var / original_var) + 
            #     (original_var + (original_preds - avg_cf_preds) ** 2) / (2 * cf_preds_var) - 
            #     0.5
            # ).mean()
            
            # # Option 2: Convert to probability distributions using softmax over a range
            # # Create bins around the predictions
            # num_bins = 10
            # pred_min = min(original_preds.min().item(), avg_cf_preds.min().item())
            # pred_max = max(original_preds.max().item(), avg_cf_preds.max().item())
            # bin_edges = torch.linspace(pred_min - 1.0, pred_max + 1.0, num_bins + 1, device=device)
            
            # # Convert predictions to probability distributions over bins
            # # Using Gaussian kernel density estimation
            # original_dist = torch.zeros(batch_size, num_bins, device=device)
            # cf_dist = torch.zeros(batch_size, num_bins, device=device)
            
            # for i in range(num_bins):
            #     bin_center = (bin_edges[i] + bin_edges[i+1]) / 2
            #     # Gaussian kernel with temperature
            #     original_dist[:, i] = torch.exp(
            #         -0.5 * ((original_preds.squeeze() - bin_center) / kl_temperature) ** 2
            #     )
            #     cf_dist[:, i] = torch.exp(
            #         -0.5 * ((avg_cf_preds.squeeze() - bin_center) / kl_temperature) ** 2
            #     )
            
            # # Normalize to probability distributions
            # original_dist = original_dist / (original_dist.sum(dim=1, keepdim=True) + 1e-8)
            # cf_dist = cf_dist / (cf_dist.sum(dim=1, keepdim=True) + 1e-8)
            
            # # Compute KL divergence: KL(original || counterfactual)
            # kl_loss_categorical = F.kl_div(
            #     F.log_softmax(cf_dist / kl_temperature, dim=1),
            #     original_dist,
            #     reduction='batchmean'
            # )
            
            # Use weighted combination of both losses
            #kl_loss = 0.5 * kl_loss_gaussian + 0.5 * kl_loss_categorical
            
          
        
    print(f"\n{'='*80}")
    print(f"Counterfactual optimization completed!")
    print(f"{'='*80}")
    
    return property_predictor


# Example usage and training function
if __name__ == "__main__":
    print("MAE-style JEPA with Masked Tabular Modeling (Simple MLP Predictor)")
    print("=" * 80)
    
    # Create model
    model = MAEStyleJEPACompositeModel(
        tabular_input_dim=7,
        combined_dim=256,
        enable_contrastive=False,
        use_masked_tabular_modeling=True,
        hidden_dim=512,
        num_masked_columns=2
    )
    
    print(f"\nModel created:")
    print(f"  - Tabular input dim: {model.tabular_input_dim}")
    print(f"  - Combined dim: {model.combined_dim}")
    print(f"  - Hidden dim: {model.hidden_dim}")
    print(f"  - Projected dim: {model.projected_dim}")
    print(f"  - Column embed dim: {model.column_embed_dim}")
    print(f"  - Num masked columns: {model.num_masked_columns}")
    print(f"  - Num visible columns: {model.num_visible_columns}")
    print(f"  - MTM enabled: {model.use_masked_tabular_modeling}")
    
    # Test forward pass
    batch_size = 4
    dummy_images = torch.randn(batch_size, 3, 224, 224).to(model.device)  # Not used but kept for compatibility
    dummy_tabular = torch.randn(batch_size, 7).to(model.device)
    dummy_text = ["This is a composite material with high strength"] * batch_size  # Not used but kept for compatibility
    
    print(f"\nTesting forward pass...")
    with torch.no_grad():
        output = model(dummy_images, dummy_tabular, dummy_text, apply_masking=True)
        
        print(f"  Output keys: {output.keys()}")
        if output['mim_outputs'] is not None:
            print(f"  MTM outputs:")
            print(f"    - Masked predictions shape: {output['mim_outputs']['masked_predictions'].shape}")
            print(f"    - Masked GT shape: {output['mim_outputs']['masked_gt'].shape}")
            print(f"    - Masked GT encoded shape: {output['mim_outputs']['masked_gt_encoded'].shape}")
            print(f"    - Context encoded shape: {output['mim_outputs']['context_encoded'].shape}")
            print(f"    - Num masked columns: {len(output['mim_outputs']['masked_indices'])}")
            print(f"    - Num visible columns: {len(output['mim_outputs']['visible_indices'])}")
            print(f"    - Masked column indices: {output['mim_outputs']['masked_indices']}")
            print(f"    - Visible column indices: {output['mim_outputs']['visible_indices']}")
        
        # Test loss computation
        losses = model.compute_jepa_losses(output)
        print(f"\n  Losses:")
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                print(f"    - {loss_name}: {loss_value.item():.6f}")
            else:
                print(f"    - {loss_name}: {loss_value}")
        
        # Test EMA update
        print(f"\nTesting EMA update...")
        model.update_target_encoder_ema()
        print(f"  ✅ EMA update successful!")
    
    print("\n✅ Model test successful!")


# Example usage for counterfactual generation
if __name__ == "__main__" and False:  # Set to True to test counterfactual generation
    print("\n" + "="*80)
    print("Testing Counterfactual Generation")
    print("="*80)
    
    # Load pretrained model (example)
    # model = MAEStyleJEPACompositeModel(...)
    # model.load_state_dict(torch.load('pretrained_model.pth'))
    # model.eval()
    
    # Create property predictor
    # property_predictor = SupervisedPropertyPredictor(
    #     vit_hidden_dim=128,
    #     num_properties=1,
    #     hidden_dim=128,
    #     dropout=0.2
    # )
    
    # Create dummy dataloader (replace with actual dataloader)
    # train_dataloader = ...
    
    # Generate counterfactuals and optimize KL divergence
    # fine_tuned_predictor = generate_counterfactuals_with_kl_optimization(
    #     model=model,
    #     property_predictor=property_predictor,
    #     train_dataloader=train_dataloader,
    #     num_epochs=10,
    #     lr=1e-4,
    #     k=5,  # Generate 5 counterfactuals per sample
    #     kl_temperature=1.0,
    #     device='cuda:0'
    # )
    
    print("Counterfactual generation example (commented out)")

