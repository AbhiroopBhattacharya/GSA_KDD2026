import torch
import torch.nn as nn
from transformers import AutoModel
from torch_geometric.loader import DataLoader
from training import model_setup
import process
from bidir_attn import MaskedReconstructionLoss,CrossModalAlignmentLoss, BidirectionalConsistencyLoss, VICRegularizer
from bidir_attn import CrossModalAttentionModule 

import torch.nn.functional as F
from cnn_encode import LightweightCNNEncoder
import numpy as np
import math
# from transformers import DeiTModel



class SupervisedModel():
        
        def __init__(self, training_parameters, model_parameters, job_parameters, processing_parameters):
            self.training_parameters = training_parameters
            self.model_parameters = model_parameters
            self.job_parameters = job_parameters
            self.processing_parameters =processing_parameters

        def get_data(self, data_path):
            dataset = process.get_dataset(data_path=data_path, target_index=self.training_parameters["target_index"],
                                          reprocess=True, model_name=self.model_parameters["model"], processing_args= self.processing_parameters)
            print(f"Type of dataset_cif: {type(dataset)}")
            print(f"First item type: {type(dataset[0]) if hasattr(dataset, '__getitem__') else 'Not indexable'}")
            print(f"Dataset length: {len(dataset)}")
            return dataset

        def load_data(self, dataset):
            train_sampler = None
            loader = DataLoader(
                dataset,
                batch_size=self.model_parameters["batch_size"],
                shuffle=(train_sampler is None),
                num_workers=0,
                pin_memory=False,
                sampler=train_sampler,
            )
            return loader

        def load_model(self, dataset, rank='cuda'):
            print(f"Loading model.. {self.model_parameters['model']}")
            ##Set up model
            model = model_setup(
                rank,
                self.model_parameters["model"],
                self.model_parameters,
                dataset,
                False,
                # self.job_parameters["load_model"],
                # self.job_parameters["model_path"],
                "./Results/Models/cgcnn.pth",
                self.model_parameters.get("print_model", False),
            )
    
            return model

class SupervisedPropertyPredictor(nn.Module):
    """
    Supervised wrapper around UnsupervisedCrossModalModel for material property prediction.
    Uses the pretrained cross-modal representations for property prediction.
    """
    
    def __init__(self, unsupervised_model, num_properties=1, hidden_dim=256, dropout=0.1):
        super(SupervisedPropertyPredictor, self).__init__()
        
        # Store the pretrained unsupervised model
        self.unsupervised_model = unsupervised_model
        self.device = unsupervised_model.device
        
        # Property prediction layers - use the actual combined_dim from the model
        combined_dim = unsupervised_model.combined_dim  # Dynamic from lightweight model
        
        # Fusion layer to combine all modalities
        self.modality_fusion = nn.Sequential(
            nn.Linear(combined_dim * 3, hidden_dim),  # 3 modalities * combined_dim
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        ).to(self.device)
        
        # Property-specific prediction heads
        self.property_heads = nn.ModuleDict({
            'formation_energy': nn.Linear(hidden_dim, 1),
            'band_gap': nn.Linear(hidden_dim, 1), 
            'hull_distance': nn.Linear(hidden_dim, 1),
            'thermal_expansion': nn.Linear(hidden_dim, 1)
        }).to(self.device)
        
        # Alternative: single multi-output head
        self.multi_property_head = nn.Linear(hidden_dim, num_properties).to(self.device)
        
    def freeze_unsupervised_backbone(self):
        """Freeze the unsupervised model parameters, only train prediction layers"""
        print("Freezing unsupervised backbone...")
        frozen_params = 0
        
        # for param in self.unsupervised_model.parameters():
        #     param.requires_grad = False
        #     frozen_params += param.numel()
        
        frozen_params = self.unsupervised_model.freeze_transformer()

        print(f"   Frozen: {frozen_params:,} backbone parameters")
        print(f"   Trainable: Only property prediction layers")

    def optimal_jepa_freeze(self):
        """
        OPTIMAL FREEZING STRATEGY: Preserve JEPA learning, enable task adaptation
        
        Based on successful JEPA pretraining, this selectively freezes components:
        - FREEZE: Expensive/well-learned components (transformer, JEPA predictors)
        - UNFREEZE: Task-adaptation components (projections, attention, property head)
        """
        print("\n🎯 OPTIMAL JEPA-to-DOWNSTREAM FREEZING:")
        print("   Strategy: Preserve cross-modal learning + Enable task adaptation")
        
        # Start by freezing everything
        for param in self.parameters():
            param.requires_grad = False
        
        frozen_components = []
        unfrozen_components = []
        
        # === FREEZE: Well-learned expensive components ===
        
        # 1. Transformer (expensive, already well-pretrained)
        if hasattr(self.unsupervised_model, 'transformer'):
            for param in self.unsupervised_model.transformer.parameters():
                param.requires_grad = False
            frozen_components.append("Transformer (109M params)")
        
        # 2. JEPA Predictors (learned cross-modal prediction, not needed for property prediction)
        jepa_predictors = ['text_predictor', 'graph_predictor', 'xrd_predictor']
        for predictor_name in jepa_predictors:
            if hasattr(self.unsupervised_model, predictor_name):
                predictor = getattr(self.unsupervised_model, predictor_name)
                for param in predictor.parameters():
                    param.requires_grad = False
                frozen_components.append(f"JEPA {predictor_name.replace('_', ' ').title()}")
        
        # 3. Auxiliary JEPA components (not needed for downstream)
        aux_components = ['vic_regularizer']
        for aux_name in aux_components:
            if hasattr(self.unsupervised_model, aux_name):
                aux_module = getattr(self.unsupervised_model, aux_name)
                for param in aux_module.parameters():
                    param.requires_grad = False
                frozen_components.append(f"{aux_name.replace('_', ' ').title()}")
        
        # === UNFREEZE: Task adaptation components ===
        
        # 1. Projection layers (critical for cross-modal alignment adaptation)
        projection_layers = ['text_proj', 'graph_proj', 'xrd_proj']
        for proj_name in projection_layers:
            if hasattr(self.unsupervised_model, proj_name):
                proj_layer = getattr(self.unsupervised_model, proj_name)
                for param in proj_layer.parameters():
                    param.requires_grad = True
                unfrozen_components.append(f"{proj_name.replace('_', ' ').title()}")
        
        # 2. Cross-modal attention (fusion needs to adapt for property prediction)
        if hasattr(self.unsupervised_model, 'cross_modal_attention'):
            for param in self.unsupervised_model.cross_modal_attention.parameters():
                param.requires_grad = True
            unfrozen_components.append("Cross-Modal Attention")
        
        # 3. XRD encoder (lightweight, can benefit from task-specific adaptation)
        if hasattr(self.unsupervised_model, 'xrd_encoder'):
            for param in self.unsupervised_model.xrd_encoder.parameters():
                param.requires_grad = True
            unfrozen_components.append("XRD Encoder")
        
        # 4. Property prediction components (obviously need to learn the task)
        for param in self.modality_fusion.parameters():
            param.requires_grad = True
        for param in self.property_heads.parameters():
            param.requires_grad = True
        for param in self.multi_property_head.parameters():
            param.requires_grad = True
        unfrozen_components.extend(["Property Fusion", "Property Heads"])
        
        # === REPORT RESULTS ===
        
        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        
        print(f"\n   ❄️  FROZEN (Preserve JEPA learning):")
        for comp in frozen_components:
            print(f"      • {comp}")
        
        print(f"\n   🔥 UNFROZEN (Task adaptation):")
        for comp in unfrozen_components:
            print(f"      • {comp}")
        
        print(f"\n   📊 Parameter Summary:")
        print(f"      • Trainable: {total_trainable:,} parameters")
        print(f"      • Frozen: {total_frozen:,} parameters") 
        print(f"      • Trainable Ratio: {total_trainable/(total_trainable + total_frozen)*100:.1f}%")
        
        # Recommendations based on trainable parameter count
        if total_trainable < 100000:
            print(f"      • Strategy: Conservative (good for preserving JEPA)")
            recommended_lr = 1e-3
        elif total_trainable < 300000:
            print(f"      • Strategy: Balanced (optimal for most cases)")
            recommended_lr = 5e-4
        else:
            print(f"      • Strategy: Aggressive (risk overwriting JEPA)")
            recommended_lr = 1e-4
        
        print(f"      • Recommended LR: {recommended_lr}")
        
        return total_trainable, recommended_lr
    
    def unfreeze_all(self):
        """Unfreeze everything for end-to-end fine-tuning"""
        print("Unfreezing entire model for end-to-end training...")
        unfrozen_params = 0
        
        for param in self.parameters():
            param.requires_grad = True
            unfrozen_params += param.numel()
            
        print(f"   Unfrozen: {unfrozen_params:,} total parameters")
    
    def get_enhanced_representations(self, text, xrd_pattern, input_for_supervised, tokenizer):
        """Get enhanced cross-modal representations from unsupervised model"""
        # Use unsupervised model without masking for inference
        with torch.set_grad_enabled(self.training):
            forward_output = self.unsupervised_model(
                text, xrd_pattern, input_for_supervised, tokenizer, 
                apply_masking=False  # No masking for property prediction
            )
            
        return forward_output['enhanced_embeddings']
    
    def forward(self, text, xrd_pattern, input_for_supervised, tokenizer, use_individual_heads=False):
        """
        Forward pass for property prediction
        
        Args:
            text: Input text descriptions
            xrd_pattern: XRD patterns
            input_for_supervised: Graph data
            tokenizer: Text tokenizer
            use_individual_heads: If True, use separate heads for each property
            
        Returns:
            Property predictions [batch_size, num_properties]
        """
        # Get enhanced representations from unsupervised model
        enhanced_embeddings = self.get_enhanced_representations(
            text, xrd_pattern, input_for_supervised, tokenizer
        )
        
        # Pool sequences to get fixed-size representations
        text_pooled = enhanced_embeddings['text'].mean(dim=1)  # [batch, combined_dim]
        graph_pooled = enhanced_embeddings['graph'].mean(dim=1)  # [batch, combined_dim] 
        xrd_pooled = enhanced_embeddings['xrd'].mean(dim=1)  # [batch, combined_dim]
        
        # Concatenate all modalities
        fused_representation = torch.cat([text_pooled, graph_pooled, xrd_pooled], dim=1)  # [batch, combined_dim*3]
        
        # Apply fusion layers
        fused_features = self.modality_fusion(fused_representation)  # [batch, hidden_dim]
        
        if use_individual_heads:
            # Use separate heads for each property
            predictions = {}
            for property_name, head in self.property_heads.items():
                predictions[property_name] = head(fused_features)
            
            # Stack into single tensor [batch, num_properties]
            property_order = ['formation_energy', 'band_gap', 'hull_distance', 'thermal_expansion']
            stacked_predictions = torch.cat([predictions[prop] for prop in property_order], dim=1)
            return stacked_predictions
        else:
            # Use single multi-output head
            return self.multi_property_head(fused_features)
    
    def get_attention_weights(self, text, xrd_pattern, input_for_supervised, tokenizer):
        """Get cross-modal attention weights for interpretability"""
        forward_output = self.unsupervised_model(
            text, xrd_pattern, input_for_supervised, tokenizer, 
            apply_masking=False
        )
        return forward_output['attention_maps']


def create_supervised_model_from_pretrained(pretrained_path, transformer_name, supervised_model, 
                                          supervised_dim, xrd_input_size, num_properties=1):
    """
    Create a supervised property predictor from a pretrained unsupervised model
    
    Args:
        pretrained_path: Path to pretrained unsupervised model checkpoint
        transformer_name: Name of transformer model
        supervised_model: Graph model
        supervised_dim: Graph feature dimension
        xrd_input_size: XRD input size
        num_properties: Number of properties to predict
        
    Returns:
        SupervisedPropertyPredictor model
    """
    # Create unsupervised model
    # unsupervised_model = UnsupervisedCrossModalModel(
    #     transformer_name=transformer_name,
    #     supervised_model=supervised_model,
    #     supervised_dim=supervised_dim,
    #     xrd_input_size=xrd_input_size
    # )
    unsupervised_model = LightweightJEPAModel(
        transformer_name=transformer_name,
        supervised_model=supervised_model,  # Used only as feature extractor
        supervised_dim=100,
        xrd_input_size=1000,
        combined_dim=128  # Changed from 256 to 128 to match training configuration
    )
    # Load pretrained weights
    checkpoint = torch.load(pretrained_path, map_location=unsupervised_model.device)
    unsupervised_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded pretrained unsupervised model from {pretrained_path}")
    
    # Create supervised wrapper
    supervised_model = SupervisedPropertyPredictor(
        unsupervised_model=unsupervised_model,
        num_properties=num_properties
    )
    
    return supervised_model


def train_supervised_finetuning(model, train_dataloader, tokenizer, 
                               num_epochs=50, lr=1e-5, freeze_backbone=True, use_optimal_jepa_freeze=True):
    """
    Train the supervised property predictor with optimal JEPA-preserving strategy
    
    Args:
        model: SupervisedPropertyPredictor
        train_dataloader: Training data loader
        tokenizer: Text tokenizer
        num_epochs: Number of training epochs
        lr: Learning rate (will be overridden if use_optimal_jepa_freeze=True)
        freeze_backbone: Whether to freeze unsupervised backbone
        use_optimal_jepa_freeze: Whether to use optimal JEPA-preserving freezing strategy
    """
    import torch.optim as optim
    from torch.nn import MSELoss
    
    # 🎯 OPTIMAL JEPA FREEZING STRATEGY
    if use_optimal_jepa_freeze:
        print("🚀 Using OPTIMAL JEPA-preserving freezing strategy...")
        trainable_count, recommended_lr = model.optimal_jepa_freeze()
        if lr is None or lr == 1e-5:  # Use recommended LR if None or default
            lr = recommended_lr
        params_to_train = [p for p in model.parameters() if p.requires_grad]
        
        print(f"\n✅ JEPA-OPTIMAL Configuration:")
        print(f"   • Learning Rate: {lr} (auto-selected)")
        print(f"   • Trainable Params: {trainable_count:,}")
        print(f"   • Strategy: Preserve cross-modal learning + Enable adaptation")
        
    else:
        # Legacy freezing options
        if freeze_backbone:
            model.freeze_unsupervised_backbone()
            params_to_train = [p for p in model.parameters() if p.requires_grad]
        else:
            model.unfreeze_all()
            params_to_train = model.parameters()
        
        trainable_count = sum(p.numel() for p in params_to_train)
        print(f"\n⚙️  Legacy Configuration:")
        print(f"   • Learning Rate: {lr}")
        print(f"   • Trainable Params: {trainable_count:,}")
        print(f"   • Strategy: {'Backbone frozen' if freeze_backbone else 'All trainable'}")
    
    # 📈 OPTIMIZER with JEPA-optimized settings
    if use_optimal_jepa_freeze:
        # Optimized for preserving JEPA while enabling adaptation
        optimizer = optim.AdamW(params_to_train, lr=lr, weight_decay=0.001, eps=1e-6)
        
        # Conservative scheduler to avoid destroying JEPA representations
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, patience=2, verbose=True, min_lr=lr*0.1
        )
        use_scheduler = True
    else:
        # Standard optimizer for legacy mode
        optimizer = optim.Adam(params_to_train, lr=lr, weight_decay=0.01)
        use_scheduler = False
    
    criterion = MSELoss()
    model.train()
    
    best_loss = float('inf')
    no_improvement_count = 0
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_dataloader):
            texts, xrd_pattern, supervised_inputs, labels = batch
            
            # Forward pass
            predictions = model(texts, xrd_pattern, supervised_inputs, tokenizer)
            
            # Fix tensor shapes
            if predictions.shape != labels.shape:
                if predictions.dim() == 1:
                    predictions = predictions.unsqueeze(1)
                else:
                    labels = labels.unsqueeze(1)
            
            # Compute loss
            loss = criterion(predictions, labels.cuda())
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # 🔧 ADAPTIVE GRADIENT CLIPPING for JEPA preservation
            if use_optimal_jepa_freeze:
                # Gentler clipping to preserve JEPA representations
                torch.nn.utils.clip_grad_norm_(params_to_train, max_norm=0.5)
            else:
                # Standard clipping
                torch.nn.utils.clip_grad_norm_(params_to_train, max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_dataloader)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 📊 ENHANCED MONITORING with JEPA-aware analysis
        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improvement_count = 0
            improvement_indicator = "✅ NEW BEST"
        else:
            no_improvement_count += 1
            improvement_indicator = f"📈 No improvement ({no_improvement_count})"
        
        print(f"Epoch {epoch + 1:2d}/{num_epochs} | Loss: {avg_loss:.6f} | LR: {current_lr:.6f} | {improvement_indicator}")
        
        # Learning rate scheduling (only for optimal strategy)
        if use_scheduler:
            scheduler.step(avg_loss)
        
        # 🛑 EARLY STOPPING with JEPA-aware patience
        patience_limit = 10 if use_optimal_jepa_freeze else 8
        if no_improvement_count >= patience_limit:
            print(f"\n🛑 EARLY STOPPING: No improvement for {no_improvement_count} epochs")
            break
        
        # 🚨 LOSS ANALYSIS with JEPA context
        if epoch == 0:
            if avg_loss > 0.5:
                print(f"   ⚠️  High initial loss ({avg_loss:.3f})")
                if use_optimal_jepa_freeze:
                    print(f"       This may be normal - JEPA features adapting to new task")
                else:
                    print(f"       Consider using optimal JEPA freezing strategy")
    
    print(f"\n🎯 Fine-tuning Results:")
    print(f"   • Best Loss: {best_loss:.6f}")
    print(f"   • Final LR: {optimizer.param_groups[0]['lr']:.8f}")
    print(f"   • Epochs Used: {epoch + 1}/{num_epochs}")
    
    # 💡 RESULTS ANALYSIS with JEPA context
    if use_optimal_jepa_freeze:
        if best_loss < 0.1:
            print(f"\n🎉 EXCELLENT: JEPA adaptation successful!")
        elif best_loss < 0.3:
            print(f"\n✅ GOOD: JEPA representations adapting well")
        else:
            print(f"\n⚠️  HIGH LOSS: Consider longer training or unfreezing more components")
    else:
        # Standard recommendations for legacy mode
        if best_loss > 0.3:
            print(f"\n💡 RECOMMENDATION: Try optimal JEPA freezing strategy")
            print(f"   use_optimal_jepa_freeze=True")
    
    return best_loss


def evaluate_model(model, dataloader, tokenizer, criterion):
    """Evaluate the supervised model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            texts, xrd_pattern, supervised_inputs, labels = batch
            predictions = model(texts, xrd_pattern, supervised_inputs, tokenizer)
            loss = criterion(predictions, labels.cuda())
            total_loss += loss.item()
    
    model.train()
    return total_loss / len(dataloader)


def train_jepa_unsupervised(model, train_dataloader, tokenizer, 
                           num_epochs=100, lr=1e-4, save_every=10, use_pure_jepa=False,
                           use_improved_alignment=False, training_phase='lightweight'):
    """
    Train the JEPA model with improved cross-modal alignment options
    
    Args:
        model: UnsupervisedCrossModalModel with JEPA predictors
        train_dataloader: Training data loader
        tokenizer: Text tokenizer
        num_epochs: Number of training epochs
        lr: Learning rate
        save_every: Save checkpoint every N epochs
        use_pure_jepa: If True, uses only essential JEPA losses (DEPRECATED - use training_phase)
        use_improved_alignment: If True, uses enhanced loss weights (DEPRECATED - use training_phase)
        training_phase: 'lightweight', 'improved_alignment', or 'progressive'
    """
    import torch.optim as optim
    import os
    
    # Get recommended configuration based on training phase
    if hasattr(model, 'get_recommended_training_config'):
        config = model.get_recommended_training_config(training_phase)
        print(f"\nUsing {training_phase.upper()} training configuration:")
        print(f"  Description: {config['loss_description']}")
        print(f"  Epochs: {config['num_epochs']}")
        print(f"  Learning Rate: {config['learning_rate']}")
        print(f"  Improved Alignment: {config.get('use_improved_alignment', False)}")
        
        # Override parameters with recommended config
        if num_epochs == 100:  # Default value, use recommended
            num_epochs = config['num_epochs']
        if lr == 1e-4:  # Default value, use recommended
            lr = config['learning_rate']
        use_improved_alignment = config.get('use_improved_alignment', False)
    else:
        print(f"\nUsing LEGACY training configuration")
        print(f"  Pure JEPA: {use_pure_jepa}")
        print(f"  Improved Alignment: {use_improved_alignment}")
    
    # Setup training
    model.freeze_transformer()  # Start with frozen transformer
    params_to_train = model.get_trainable_parameters()
    
    optimizer = optim.AdamW(params_to_train, lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    model.train()
    
    print(f"\nTraining JEPA model...")
    print(f"  Total epochs: {num_epochs}")
    print(f"  Training phase: {training_phase}")
    
    for epoch in range(num_epochs):
        total_loss = 0
        loss_components = {}
        
        # Progressive unfreezing for progressive training
        if training_phase == 'progressive' and hasattr(model, 'get_recommended_training_config'):
            config = model.get_recommended_training_config('progressive')
            schedule = config.get('unfreezing_schedule', {})
            
            if epoch == 15:  # Phase 2: Unfreeze last 2 layers
                print(f"\nEpoch {epoch + 1}: Entering Phase 2 - Unfreezing last 2 transformer layers")
                model.unfreeze_transformer_layers(layers=[-2, -1])
                params_to_train = model.get_trainable_parameters()
                optimizer = optim.AdamW(params_to_train, lr=5e-6, weight_decay=0.01)
            elif epoch == 35:  # Phase 3: Unfreeze all layers
                print(f"\nEpoch {epoch + 1}: Entering Phase 3 - Unfreezing all transformer layers")
                model.unfreeze_transformer_layers()  # Unfreeze all
                params_to_train = model.get_trainable_parameters()
                optimizer = optim.AdamW(params_to_train, lr=1e-6, weight_decay=0.01)
        
        for batch_idx, batch in enumerate(train_dataloader):
            texts, xrd_pattern, supervised_inputs, _ = batch  # Ignore labels for unsupervised
            
            # Forward pass
            forward_output = model(
                texts, xrd_pattern, supervised_inputs, tokenizer,
                target_modality=None,  # Predict all modalities
                apply_masking=True
            )
            
            # Compute losses with improved alignment if requested
            if hasattr(model, 'compute_lightweight_jepa_losses'):
                losses = model.compute_lightweight_jepa_losses(
                    forward_output, 
                    use_improved_alignment=use_improved_alignment
                )
            else:
                # Fallback to old method
                losses = model.compute_jepa_losses(forward_output, use_pure_jepa=use_pure_jepa)
            
            # Backward pass
            optimizer.zero_grad()
            losses['total_weighted_loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(params_to_train, max_norm=1.0)
            optimizer.step()
            
            # Accumulate losses for reporting
            total_loss += losses['total_weighted_loss'].item()
            for loss_name, loss_value in losses.items():
                if loss_name not in loss_components:
                    loss_components[loss_name] = 0
                loss_components[loss_name] += loss_value.item()
        
        scheduler.step()
        
        # Report epoch results
        avg_loss = total_loss / len(train_dataloader)
        print(f"\nEpoch {epoch + 1}/{num_epochs} ({training_phase})")
        print(f"Total Loss: {avg_loss:.6f}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Report individual loss components
        for loss_name, loss_value in loss_components.items():
            avg_component = loss_value / len(train_dataloader)
            print(f"  {loss_name}: {avg_component:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            checkpoint_dir = 'Results/Checkpoint'
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            checkpoint_path = f'{checkpoint_dir}/checkpoint_lightweight_jepa_{training_phase}_{epoch + 1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'training_phase': training_phase,
                'use_improved_alignment': use_improved_alignment
            }, checkpoint_path)
            
            print(f"Checkpoint saved: {checkpoint_path}")
    
    print(f"\nJEPA training completed!")
    print(f"Training phase: {training_phase}")
    print(f"Final loss: {avg_loss:.6f}")
    
    if training_phase == 'improved_alignment':
        print("\nNEXT STEPS for even better results:")
        print("1. Run cross-modal analysis: python simple_attention_viz.py")
        print("2. If self-similarity > 0.8: Great! Continue to supervised fine-tuning")
        print("3. If self-similarity < 0.8: Consider 'progressive' training phase")
    
    return avg_loss


class LightweightLatentPredictor(nn.Module):
    """
    Lightweight JEPA-style latent predictor using simple MLPs instead of transformers
    Much faster and less memory intensive.
    """
    
    def __init__(self, combined_dim, hidden_dim=128, num_layers=2, dropout=0.1):
        super(LightweightLatentPredictor, self).__init__()
        
        # Simple MLP predictor (much faster than transformer)
        layers = []
        input_dim = combined_dim * 2  # Two input modalities
        
        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, combined_dim))
        
        self.predictor = nn.Sequential(*layers)
    
    def forward(self, modality1_latent, modality2_latent):
        """
        Simple concatenation + MLP prediction 
        Returns pooled prediction to avoid sequence length issues
        
        Args:
            modality1_latent: [batch, seq_len, combined_dim]
            modality2_latent: [batch, seq_len, combined_dim] 
            
        Returns:
            predicted_latent: [batch, 1, combined_dim] - pooled prediction
        """
        # Pool sequences to fixed size (much simpler and more stable)
        mod1_pooled = modality1_latent.mean(dim=1)  # [batch, combined_dim]
        mod2_pooled = modality2_latent.mean(dim=1)  # [batch, combined_dim]
        
        # Concatenate
        combined_input = torch.cat([mod1_pooled, mod2_pooled], dim=-1)  # [batch, combined_dim*2]
        
        # Predict
        predicted_pooled = self.predictor(combined_input)  # [batch, combined_dim]
        
        # Return as single-token sequence to avoid length mismatches
        predicted_latent = predicted_pooled.unsqueeze(1)  # [batch, 1, combined_dim]
        
        return predicted_latent


class ContrastiveCrossModalLoss(nn.Module):
    """
    Contrastive loss for explicit cross-modal alignment
    Uses InfoNCE loss to encourage same-sample cross-modal similarity
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.tau_text_graph = nn.Parameter(torch.tensor(0.07))
        self.tau_text_xrd   = nn.Parameter(torch.tensor(0.07))
        self.tau_graph_xrd  = nn.Parameter(torch.tensor(0.07))

    def info_nce_loss(self, anchor, positive, tau=0.07, hard_k=5):
        logits = torch.mm(anchor, positive.t()) /tau

        # Hard negative mining: keep only top-k highest negative sims
        mask = torch.eye(anchor.size(0), device=anchor.device).bool()
        neg_logits = logits[~mask].view(anchor.size(0), -1)
        hard_neg, _ = torch.topk(neg_logits, k=hard_k, dim=1)
        
        # Rebuild logits with pos + hard negatives
        pos_logits = logits[mask].view(anchor.size(0), 1)
        new_logits = torch.cat([pos_logits, hard_neg], dim=1)

        new_labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)
        return F.cross_entropy(new_logits, new_labels)


    def margin_contrastive(self, anchor, positive, margin=0.05):
        sim_matrix = torch.mm(anchor, positive.t())
        pos = sim_matrix.diag()
        neg = sim_matrix + torch.eye(sim_matrix.size(0), device=anchor.device) * -1e9
        hardest_neg, _ = neg.max(dim=1)
        loss = F.relu(margin + hardest_neg - pos).mean()
        return loss


    def uniformity_loss(self, z, t=2):
        return torch.mean(torch.exp(-t * torch.pdist(z, p=2).pow(2)))

    def forward(self, text_emb, graph_emb, xrd_emb):
        # Normalize embeddings
        text_norm = F.normalize(text_emb.mean(dim=1), dim=-1)
        graph_norm = F.normalize(graph_emb.mean(dim=1), dim=-1)  
        xrd_norm   = F.normalize(xrd_emb.mean(dim=1), dim=-1)

        # Contrastive loss with hard negatives + per-pair tau
        loss = 0
        loss += self.info_nce_loss(text_norm, graph_norm, tau=self.tau_text_graph, hard_k=2)
        loss += self.info_nce_loss(text_norm, xrd_norm, tau=self.tau_text_xrd, hard_k=2)
        loss += self.info_nce_loss(graph_norm, xrd_norm, tau=self.tau_graph_xrd, hard_k=2)
        loss = loss / 3

        # -------- Margin contrastive part --------
        margin_loss = 0
        margin_loss += self.margin_contrastive(text_norm, graph_norm, margin=0.05)
        margin_loss += self.margin_contrastive(text_norm, xrd_norm, margin=0.05)
        margin_loss += self.margin_contrastive(graph_norm, xrd_norm, margin=0.01)
        margin_loss = margin_loss / 3
        # Optional uniformity
        uniformity = self.uniformity_loss(torch.cat([text_norm, graph_norm, xrd_norm], dim=0))

        return loss +  0.5 * margin_loss + 0.05 * uniformity

class LightweightJEPACompositeModel(nn.Module):
    """
    Lightweight JEPA for Composite Materials (2 Modalities)
    - Image (ViT encoder)
    - Tabular Data (MLP encoder)
    
    JEPA + VIC + Contrastive Loss architecture
    Much faster than 3-modality version
    """
    
    def __init__(self, vit_model_name='google/vit-base-patch16-224', 
                 tabular_input_dim=7, combined_dim=256, enable_contrastive=False):
        super(LightweightJEPACompositeModel, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.combined_dim = combined_dim
        self.enable_contrastive = enable_contrastive
        
        # Import ViTModel here
        from transformers import ViTModel
        
        # # Image encoder: Vision Transformer (ViT)
        print(f"Loading ViT model: {vit_model_name}")

        self.vit_encoder = ViTModel.from_pretrained(vit_model_name).to(self.device)
        vit_hidden_size = self.vit_encoder.config.hidden_size  # 768 for ViT-base
        # deit_model_name = 'facebook/deit-small-patch16-224'
        # self.vit_encoder = DeiTModel.from_pretrained(deit_model_name).to(self.device)
        # vit_hidden_size = self.vit_encoder.config.hidden_size  # 768 for ViT-base
        # cnn_hidden_dim = 128
        # self.vit_encoder = LightweightCNNEncoder(
        #     hidden_dim=cnn_hidden_dim, patch_size=16, image_size=224, use_cls_token=True).to(self.device)
        # vit_hidden_size = self.vit_encoder.config.hidden_size  # 768 for ViT-base
        # Tabular encoder: Simple MLP
        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, combined_dim)
        ).to(self.device)
        
        # Projection layers to common space
        self.image_proj = nn.Sequential(
            nn.Linear(vit_hidden_size, combined_dim),
            nn.GELU(),
            nn.LayerNorm(combined_dim)
        ).to(self.device)
        
        self.tabular_proj = nn.Sequential(
            nn.Linear(combined_dim, combined_dim),
            nn.GELU(),
            nn.LayerNorm(combined_dim)
        ).to(self.device)
        
        # Lightweight cross-modal attention for 2 modalities
        self.cross_modal_attention = TwoModalityAttention(combined_dim).to(self.device)
        
        # JEPA predictors (for 2 modalities: predict each from the other)
        self.image_predictor = SingleModalityPredictor(combined_dim, hidden_dim=128).to(self.device)
        self.tabular_predictor = SingleModalityPredictor(combined_dim, hidden_dim=128).to(self.device)
        
        # Regularizers
        self.vic_regularizer = VICRegularizerTwoModality(sim_coeff=0.1, std_coeff=0.5, cov_coeff=0.01).to(self.device)
        
        # Optional contrastive learning
        if enable_contrastive:
            self.contrastive_alignment = TwoModalityContrastiveLoss(temperature=0.1).to(self.device)
            print(f"   Contrastive learning enabled")
        else:
            self.contrastive_alignment = None
            print(f"   Pure lightweight mode (no contrastive)")
    
    def freeze_vit(self):
        """Freeze ViT encoder for faster training"""
        print("Freezing ViT encoder...")
        frozen_params = 0
        for param in self.vit_encoder.parameters():
            param.requires_grad = False
            frozen_params += param.numel()
        print(f"   Frozen: {frozen_params:,} parameters")
        return frozen_params
    
    def unfreeze_vit(self):
        """Unfreeze ViT encoder"""
        print("Unfreezing ViT encoder...")
        for param in self.vit_encoder.parameters():
            param.requires_grad = True
    
    def get_trainable_parameters(self):
        """Get trainable parameters"""
        return [p for p in self.parameters() if p.requires_grad]
    
    def forward(self, images, tabular_data, apply_masking=True):
        """
        Forward pass for 2 modalities
        
        Args:
            images: [batch, 3, 224, 224]
            tabular_data: [batch, tabular_input_dim]
            apply_masking: Kept for compatibility (not used in JEPA)
        
        Returns:
            dict with embeddings and predictions
        """
        batch_size = images.size(0)
        
        # === ENCODE MODALITIES ===
        
        # Image encoding with ViT
        images = images.to(self.device)
        vit_output = self.vit_encoder(pixel_values=images)
        image_features = vit_output.last_hidden_state  # [batch, seq_len, vit_hidden_size]
        image_embeddings = self.image_proj(image_features)  # [batch, seq_len, combined_dim]
        
        # Tabular encoding with MLP
        tabular_data = tabular_data.to(self.device)
        tabular_features = self.tabular_encoder(tabular_data)  # [batch, combined_dim]
        tabular_embeddings = self.tabular_proj(tabular_features).unsqueeze(1)  # [batch, 1, combined_dim]
        
        # === CROSS-MODAL ATTENTION ===
        original_embeddings = {
            'image': image_embeddings.clone(),
            'tabular': tabular_embeddings.clone()
        }
        
        enhanced_embeddings, attention_maps = self.cross_modal_attention(
            image_embeddings, tabular_embeddings
        )
        
        # === JEPA PREDICTION ===
        predictions = {
            'image': self.image_predictor(enhanced_embeddings['tabular']),
            'tabular': self.tabular_predictor(enhanced_embeddings['image'])
        }
        
        return {
            'original_embeddings': original_embeddings,
            'enhanced_embeddings': enhanced_embeddings,
            'latent_predictions': predictions,
            'attention_maps': attention_maps
        }
    
    def compute_jepa_losses(self, forward_output, use_improved_alignment=False):
        """
        Compute JEPA + VIC + Contrastive losses for 2 modalities
        """
        original = forward_output['original_embeddings']
        enhanced = forward_output['enhanced_embeddings']
        predictions = forward_output['latent_predictions']
        
        losses = {}
        
        # 1. JEPA LATENT PREDICTION LOSS
        latent_prediction_loss = 0
        for modality, predicted_latent in predictions.items():
            target_latent = original[modality]
            
            # Handle size mismatches
            if predicted_latent.size() != target_latent.size():
                if target_latent.size(1) > 1:
                    target_latent = target_latent.mean(dim=1, keepdim=True)
            
            loss = F.mse_loss(predicted_latent, target_latent.detach())
            losses[f'{modality}_latent_prediction'] = loss
            latent_prediction_loss += loss
        
        losses['total_latent_prediction'] = latent_prediction_loss / 2  # 2 modalities
        
        # 2. VIC REGULARIZATION
        losses['vic_regularization'] = self.vic_regularizer(
            enhanced['image'], enhanced['tabular']
        )
        
        # 3. CONTRASTIVE ALIGNMENT (optional)
        if self.contrastive_alignment is not None:
            losses['contrastive_alignment'] = self.contrastive_alignment(
                enhanced['image'], enhanced['tabular']
            )
        else:
            losses['contrastive_alignment'] = torch.tensor(0.0, device=self.device)
        
        # 4. TOTAL WEIGHTED LOSS
        if use_improved_alignment and self.contrastive_alignment is not None:
            losses['total_weighted_loss'] = (
                1.0 * losses['total_latent_prediction'] +
                0.2 * losses['vic_regularization'] +
                0.5 * losses['contrastive_alignment']
            )
            mode = "improved_alignment"
        elif self.contrastive_alignment is not None:
            losses['total_weighted_loss'] = (
                1.0 * losses['total_latent_prediction'] +
                0.05 * losses['vic_regularization'] +
                0.1 * losses['contrastive_alignment']
            )
            mode = "lightweight_with_contrastive"
        else:
            losses['total_weighted_loss'] = (
                losses['total_latent_prediction'] +
                0.01 * losses['vic_regularization']
            )
            mode = "pure_lightweight"
        
        losses['training_mode'] = mode
        
        return losses


# Helper classes for 2-modality JEPA

class TwoModalityAttention(nn.Module):
    """Lightweight cross-modal attention for 2 modalities"""
    
    def __init__(self, combined_dim, dropout=0.1, use_attention_pooling=True):
        super(TwoModalityAttention, self).__init__()
        self.combined_dim = combined_dim
        self.layer_norm = nn.LayerNorm(combined_dim)
        self.dropout = nn.Dropout(dropout)
        self.use_attention_pooling = use_attention_pooling
        
        # Attention pooling for images (if sequence length > 1)
        if use_attention_pooling:
            self.image_attention_pool = nn.Sequential(
                nn.Linear(combined_dim, combined_dim),
                nn.Tanh(),
                nn.Linear(combined_dim, 1)
            )
        
        # Projections for each modality
        self.image_proj = nn.Sequential(
            nn.Linear(combined_dim, combined_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.tabular_proj = nn.Sequential(
            nn.Linear(combined_dim, combined_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Cross-modal fusion
        self.image_fusion = nn.Sequential(
            nn.Linear(combined_dim, combined_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(combined_dim * 2, combined_dim)
        )
        self.tabular_fusion = nn.Sequential(
            nn.Linear(combined_dim, combined_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(combined_dim * 2, combined_dim)
        )
        
        self.similarity_temp = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, image_emb, tabular_emb):
        """Cross-modal fusion without data leakage"""
        # Pool images with attention (if sequence length > 1)
        if image_emb.size(1) > 1 and self.use_attention_pooling:
            # Attention pooling: learn which patches are important
            attn_weights = F.softmax(self.image_attention_pool(image_emb), dim=1)  # [batch, seq_len, 1]
            image_pooled = (image_emb * attn_weights).sum(dim=1)  # [batch, dim]
        else:
            # Mean pooling (fallback for seq_len=1 or if attention pooling disabled)
            image_pooled = image_emb.mean(dim=1)
        
        # Tabular is already pooled (size 1) or use mean if needed
        if tabular_emb.size(1) == 1:
            tabular_pooled = tabular_emb.squeeze(1)
        else:
            tabular_pooled = tabular_emb.mean(dim=1)
        
        # Normalize
        image_pooled = self.layer_norm(image_pooled)
        tabular_pooled = self.layer_norm(tabular_pooled)
        
        # Project
        image_proj = self.image_proj(image_pooled)
        tabular_proj = self.tabular_proj(tabular_pooled)
        
        # Cross-modal fusion
        # Image enhanced by tabular (no image)
        image_cross_modal = self.image_fusion(tabular_proj)
        image_fused = image_proj + self.dropout(image_cross_modal)
        
        # Tabular enhanced by image (no tabular)
        tabular_cross_modal = self.tabular_fusion(image_proj)
        tabular_fused = tabular_proj + self.dropout(tabular_cross_modal)
        
        # Return with sequence length 1
        enhanced_embeddings = {
            'image': image_fused.unsqueeze(1),
            'tabular': tabular_fused.unsqueeze(1)
        }
        
        # Compute attention maps
        image_norm = F.normalize(image_fused, dim=-1)
        tabular_norm = F.normalize(tabular_fused, dim=-1)
        
        within_sample_similarity = (image_norm * tabular_norm).sum(dim=-1) /self.similarity_temp.clamp(min=1e-6)
        attention_maps = { 
            'image_to_tabular': within_sample_similarity.unsqueeze(1).unsqueeze(2),  # [batch_size, 1, 1]
            'tabular_to_image': within_sample_similarity.unsqueeze(1).unsqueeze(2)   # [batch_size, 1, 1]
        }

        # attention_maps = {
        #     'image_to_tabular': torch.mm(image_norm, tabular_norm.t()).unsqueeze(1),
        #     'tabular_to_image': torch.mm(tabular_norm, image_norm.t()).unsqueeze(1)
        # }
        
        return enhanced_embeddings, attention_maps


class SingleModalityPredictor(nn.Module):
    """Lightweight predictor for single modality from another"""
    
    def __init__(self, combined_dim, hidden_dim=128, num_layers=2, dropout=0.1):
        super(SingleModalityPredictor, self).__init__()
        
        layers = []
        layers.append(nn.Linear(combined_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_dim, combined_dim))
        
        self.predictor = nn.Sequential(*layers)
    
    def forward(self, modality_latent):
        """Predict from single modality"""
        pooled = modality_latent.mean(dim=1)  # [batch, combined_dim]
        predicted = self.predictor(pooled)
        return predicted.unsqueeze(1)  # [batch, 1, combined_dim]


class VICRegularizerTwoModality(nn.Module):
    """VIC Regularizer adapted for 2 modalities"""
    
    def __init__(self, sim_coeff=0.1, std_coeff=0.5, cov_coeff=0.01):
        super(VICRegularizerTwoModality, self).__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
    
    def forward(self, z1, z2):
        """
        Args:
            z1, z2: [batch, seq_len, dim]
        """
        # Pool sequences
        z1_pooled = z1.mean(dim=1)  # [batch, dim]
        z2_pooled = z2.mean(dim=1)  # [batch, dim]
        
        batch_size = z1_pooled.size(0)
        dim = z1_pooled.size(1)
        
        # Protection: batch_size minimum
        if batch_size < 2:
            return torch.tensor(0.0, device=z1.device, dtype=z1.dtype)
        
        # Variance loss
        std_z1 = torch.sqrt(z1_pooled.var(dim=0) + 1e-4)
        std_z2 = torch.sqrt(z2_pooled.var(dim=0) + 1e-4)
        std_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))
        
        # Covariance loss
        z1_centered = z1_pooled - z1_pooled.mean(dim=0)
        z2_centered = z2_pooled - z2_pooled.mean(dim=0)
        
        # FIX: Protection contre division par zéro
        cov_z1 = (z1_centered.T @ z1_centered) / max(batch_size - 1, 1)
        cov_z2 = (z2_centered.T @ z2_centered) / max(batch_size - 1, 1)
        
        cov_loss = off_diagonal(cov_z1).pow(2).sum() / dim + off_diagonal(cov_z2).pow(2).sum() / dim
        
        # Total VIC loss
        loss = self.std_coeff * std_loss + self.cov_coeff * cov_loss
        
        # Protection: Clip pour éviter explosion
        loss = torch.clamp(loss, max=100.0)
        
        return loss


class TwoModalityContrastiveLoss(nn.Module):
    """Contrastive loss for 2 modalities"""
    
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.tau = nn.Parameter(torch.tensor(0.07))
    
    def info_nce_loss(self, anchor, positive, hard_k=2):
        batch_size = anchor.size(0)
        
        # Protection: batch_size minimum
        if batch_size < 2:
            return torch.tensor(0.0, device=anchor.device, dtype=anchor.dtype)
        
        # Safe normalization
        anchor = F.normalize(anchor + 1e-8, dim=-1)
        positive = F.normalize(positive + 1e-8, dim=-1)
        
        logits = torch.mm(anchor, positive.t()) / (torch.abs(self.tau) + 1e-8)
        
        # Adjust hard_k to batch_size
        effective_hard_k = min(hard_k, batch_size - 1)
        
        if effective_hard_k < 1:
            return F.mse_loss(anchor, positive)
        
        mask = torch.eye(batch_size, device=anchor.device).bool()
        neg_logits = logits[~mask].view(batch_size, -1)
        hard_neg, _ = torch.topk(neg_logits, k=effective_hard_k, dim=1)
        
        pos_logits = logits[mask].view(batch_size, 1)
        new_logits = torch.cat([pos_logits, hard_neg], dim=1)
        
        new_labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)
        loss = F.cross_entropy(new_logits, new_labels)
        
        # Protection: Clip
        return torch.clamp(loss, max=10.0)
    
    def forward(self, image_emb, tabular_emb):
        # Safe normalize
        image_norm = F.normalize(image_emb.mean(dim=1) + 1e-8, dim=-1)
        tabular_norm = F.normalize(tabular_emb.mean(dim=1) + 1e-8, dim=-1)
        
        return self.info_nce_loss(image_norm, tabular_norm, hard_k=2)


def off_diagonal(x):
    """Return off-diagonal elements of a matrix"""
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

