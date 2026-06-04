
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import argparse
from pathlib import Path
import time
from tqdm import tqdm
from typing import Optional, Dict
import yaml
from vl_jepa_model import create_vl_jepa_model
from vl_jepa_mask import create_mask_generator
from vl_jepa_metrics import compute_retrieval_metrics, AverageMeter
from vl_jepa_logger import setup_logger
from vl_jepa_utils import save_checkpoint, load_checkpoint
from utils import generate_ood_qh_data
from transformers import AutoTokenizer

def create_optimizer(model: nn.Module, config: Dict, stage2: bool = False, stage2_lr_factor: float = 0.5) -> torch.optim.Optimizer:
    """Create optimizer from config
    
    Args:
        model: The VL-JEPA model
        config: Configuration dictionary
        stage2: If True, use Stage-2 settings (frozen text encoder, lower LR)
        stage2_lr_factor: Learning rate multiplier for Stage-2
    """
    opt_config = config['training']['optimizer']
    opt_type = opt_config.get('type', 'adamw')
    lr = float(config['training']['learning_rate'])  # Convert to float explicitly
    weight_decay = float(config['training']['weight_decay'])  # Convert to float explicitly
    print(f"Learning rate: {lr} (type: {type(lr)})")
    print(f"Weight decay: {weight_decay} (type: {type(weight_decay)})")
    betas = tuple(opt_config.get('betas', [0.9, 0.999]))
    eps = float(opt_config.get('eps', 1e-8))  # Convert to float explicitly (YAML reads 1e-8 as string)
    print(f"Eps: {eps} (type: {type(eps)})")
    
    # Stage-2: Use reduced learning rate
    if stage2:
        lr = lr * stage2_lr_factor
        print(f"Stage-2: Using reduced LR = {lr:.2e} (factor: {stage2_lr_factor})")
    
    # Get parameter groups (excludes frozen text encoder in Stage-2)
    if hasattr(model, 'get_parameter_groups'):
        param_groups = model.get_parameter_groups(lr, stage2=stage2)
        # Ensure all lr values in param_groups are floats
        for group in param_groups:
            if 'lr' in group:
                group['lr'] = float(group['lr'])
    else:
        # Fallback: only trainable parameters
        param_groups = [p for p in model.parameters() if p.requires_grad]
    
    if opt_type == 'adamw8bit' and HAS_BITSANDBYTES:
        optimizer = bnb.optim.AdamW8bit(
            param_groups,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        print("Using 8-bit AdamW optimizer")
    else:
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        print("Using standard AdamW optimizer")
    
    return optimizer


def create_scheduler(optimizer: torch.optim.Optimizer, config: Dict, steps_per_epoch: int):
    """Create learning rate scheduler"""
    sched_config = config['training']['scheduler']
    sched_type = sched_config.get('type', 'cosine')
    
    num_epochs = config['training']['num_epochs']
    warmup_epochs = config['training'].get('warmup_epochs', 10)
    min_lr = config['training'].get('min_lr', 1e-6)
    
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = num_epochs * steps_per_epoch
    
    if sched_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=min_lr,
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10 * steps_per_epoch,
            gamma=0.1,
        )
    
    return scheduler, warmup_steps


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    mask_generator,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    epoch: int,
    config: Dict,
    logger,
    global_step: int,
    warmup_steps: int,
    stage2: bool = False,
) -> tuple:
    """Train for one epoch"""
    model.train()
    
    # Stage-2 safety check: verify text encoder is still frozen
    if stage2:
        verify_info = model.verify_frozen_text_encoder()
        if not verify_info['all_frozen']:
            raise RuntimeError(f"Stage-2 ERROR: Text encoder unfrozen! {verify_info}")
    
    loss_meter = AverageMeter()
    jepa_loss_meter = AverageMeter()
    contrastive_loss_meter = AverageMeter()
    
    batch_size = config['training']['batch_size']
    grad_accum_steps = config['training']['gradient_accumulation_steps']
    log_every = config['logging'].get('log_every', 100)
    grad_clip = config['training'].get('gradient_clip', 1.0)
    empty_cache_every = config['training'].get('empty_cache_every', 100)
    use_wandb = config['logging'].get('use_wandb', False)
    
    # EMA momentum schedule
    ema_start = config['training'].get('ema_momentum_start', 0.996)
    ema_end = config['training'].get('ema_momentum_end', 1.0)
    
    # Loss weights - LOCKED, do not change during training
    jepa_loss_weight = config['training'].get('jepa_loss_weight', 1.0)
    contrastive_loss_weight = config['training'].get('contrastive_loss_weight', 0.5)
    text_model_name = 'm3rg-iitd/matscibert'    
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        images = batch['image'].cuda()
        text = batch['text']  # This is a list of text strings (one per batch item)
        properties = batch['properties'].cuda()
        
        # Tokenize batch of texts (use tokenizer() not tokenizer.encode() for batches)
        tokenized = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = tokenized['input_ids'].cuda()
        attention_mask = tokenized['attention_mask'].cuda()

        # Generate masks (context = visible, target = to predict)
        context_masks = []
        target_masks = []
        for _ in range(images.shape[0]):
            ctx_mask, tgt_mask = mask_generator()
            context_masks.append(ctx_mask)
            target_masks.append(tgt_mask)
        
        context_mask = torch.stack(context_masks).cuda()
        target_mask = torch.stack(target_masks).cuda()
        
        # Forward pass with mixed precision
        with autocast(enabled=True):
            outputs = model(
                images=images,
                text_input_ids=input_ids,
                text_attention_mask=attention_mask,
                context_mask=context_mask,
                target_mask=target_mask,
                mode="both",
            )
            
            # Compute weighted loss: JEPA + contrastive
            jepa_loss = outputs['jepa_loss']
            contrastive_loss = outputs['contrastive_loss']
            loss = jepa_loss_weight * jepa_loss + contrastive_loss_weight * contrastive_loss
            loss = loss / grad_accum_steps  # Scale loss for gradient accumulation
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Update weights every grad_accum_steps
        if (batch_idx + 1) % grad_accum_steps == 0:
            # Unscale gradients and clip
            scaler.unscale_(optimizer)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Update EMA target encoder
            progress = global_step / (config['training']['num_epochs'] * len(dataloader))
            ema_momentum = ema_start + (ema_end - ema_start) * progress
            model.ema_momentum = ema_momentum
            model.update_target_encoder()
            
            # Learning rate warmup
            if global_step < warmup_steps:
                lr_scale = min(1.0, float(global_step + 1) / warmup_steps)
                base_lr = float(config['training']['learning_rate'])  # Convert to float (YAML reads 1e-4 as string)
                for pg in optimizer.param_groups:
                    pg['lr'] = base_lr * lr_scale
            else:
                scheduler.step()
            
            global_step += 1
        
        # Update meters
        loss_meter.update(loss.item() * grad_accum_steps, images.size(0))
        if 'jepa_loss' in outputs:
            jepa_loss_meter.update(outputs['jepa_loss'].item(), images.size(0))
        if 'contrastive_loss' in outputs:
            contrastive_loss_meter.update(outputs['contrastive_loss'].item(), images.size(0))
        
        # Logging
        if batch_idx % log_every == 0:
            lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f"{loss_meter.avg:.4f}",
                'jepa': f"{jepa_loss_meter.avg:.4f}",
                'contrast': f"{contrastive_loss_meter.avg:.4f}",
                'lr': f"{lr:.2e}",
            })
            
            if use_wandb and wandb.run is not None:
                wandb.log({
                    'train/loss': loss_meter.avg,
                    'train/jepa_loss': jepa_loss_meter.avg,
                    'train/contrastive_loss': contrastive_loss_meter.avg,
                    'train/lr': lr,
                    'train/ema_momentum': model.ema_momentum,
                    'train/epoch': epoch,
                    'train/step': global_step,
                })
        
        # Clear CUDA cache periodically
        if batch_idx % empty_cache_every == 0:
            torch.cuda.empty_cache()
    
    logger.info(f"Epoch {epoch} - Loss: {loss_meter.avg:.4f}, JEPA Loss: {jepa_loss_meter.avg:.4f}, Contrastive Loss: {contrastive_loss_meter.avg:.4f}")
    
    return loss_meter.avg, global_step


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    epoch: int,
    config: Dict,
    logger,
) -> Dict[str, float]:
    """Validate model"""
    model.eval()
    
    loss_meter = AverageMeter()
    
    # Collect embeddings for retrieval
    all_image_embeds = []
    all_text_embeds = []
    
    use_wandb = config['logging'].get('use_wandb', False)
    
    pbar = tqdm(dataloader, desc=f"Validation Epoch {epoch}")
    text_model_name = 'm3rg-iitd/matscibert'    
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)

    for batch in pbar:
        images = batch['image'].cuda()
        text = batch['text']  # This is a list of text strings (one per batch item)
        properties = batch['properties'].cuda()
        tokenized = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = tokenized['input_ids'].cuda()
        attention_mask = tokenized['attention_mask'].cuda()
        
        # Forward pass
        outputs = model(
            images=images,
            text_input_ids=input_ids,
            text_attention_mask=attention_mask,
            mode="contrastive",
        )
        
        # Collect embeddings
        all_image_embeds.append(outputs['vision_embed'].cpu())
        all_text_embeds.append(outputs['text_embed'].cpu())
        
        if 'loss' in outputs:
            loss_meter.update(outputs['loss'].item(), images.size(0))
    
    # Compute retrieval metrics
    image_embeds = torch.cat(all_image_embeds, dim=0)
    text_embeds = torch.cat(all_text_embeds, dim=0)
    
    # Limit to first 1000 samples for faster evaluation
    if image_embeds.shape[0] > 1000:
        image_embeds = image_embeds[:1000]
        text_embeds = text_embeds[:1000]
    
    metrics = compute_retrieval_metrics(image_embeds, text_embeds)
    metrics['val_loss'] = loss_meter.avg
    
    logger.info(f"Validation Epoch {epoch} - Loss: {loss_meter.avg:.4f}")
    logger.info(f"Retrieval Metrics: {metrics}")
    
    if use_wandb and wandb.run is not None:
        wandb.log({f'val/{k}': v for k, v in metrics.items()})
        wandb.log({'val/epoch': epoch})
    
    return metrics


def main():
    num_workers =0
    stage2_lr_factor = 0.1
    stage2_epochs = 1
    # args = parse_args()
    config_path = './vl_jepa_config.yaml'
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # print("Configuration:")
    # print_config(config)
    
    # Setup logger
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logger = setup_logger(
        name="vl_jepa",
        log_file=log_dir / f"train_{time.strftime('%Y%m%d_%H%M%S')}.log",
    )
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = 'cpu'
    
    logger.info(f"Using device: {device}")
    if device == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Initialize wandb
    # if args.wandb and HAS_WANDB and config['logging'].get('use_wandb', False):
    #     wandb.init(
    #         project=config['logging'].get('wandb_project', 'vl-jepa'),
    #         entity=config['logging'].get('wandb_entity', None),
    #         config=config,
    #         name=f"vl_jepa_{time.strftime('%Y%m%d_%H%M%S')}",
    #     )
    
    # Create model
    logger.info("Creating model...")
    model = create_vl_jepa_model(config)
    model = model.to(device)
    
    # Stage-2: Freeze text encoder
    stage2 = False
    resume_checkpoint = False
    if stage2:
        logger.info("=" * 50)
        logger.info("STAGE-2 TRAINING MODE")
        logger.info("=" * 50)
        freeze_info = model.freeze_text_encoder()
        logger.info(f"Frozen text encoder parameters: {freeze_info['frozen_text_params'] / 1e6:.2f}M")
        logger.info(f"Trainable parameters: {freeze_info['trainable_params'] / 1e6:.2f}M")
        logger.info("Text encoder weights are FROZEN - no gradients will flow through DistilBERT")
        # logger.info(f"Stage-2 epochs: {args.stage2_epochs}")
        # logger.info(f"Stage-2 LR factor: {args.stage2_lr_factor}")
        # logger.info(f"Early stopping patience: {args.early_stop_patience} (based on mean_recall)")
        
        # Verify freeze was successful
        verify_info = model.verify_frozen_text_encoder()
        if verify_info['all_frozen']:
            logger.info(f"✓ Verified: All {verify_info['total_count']} text encoder parameters are frozen")
        else:
            logger.error(f"✗ ERROR: Only {verify_info['frozen_count']}/{verify_info['total_count']} text params frozen!")
            raise RuntimeError("Text encoder freeze verification failed!")
        logger.info("=" * 50)
    
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {num_params / 1e6:.2f}M")
    logger.info(f"Trainable parameters: {num_trainable / 1e6:.2f}M")
    
    # Create datasets
    logger.info("Creating datasets...")
    
    # Get tokenizer from model
    tokenizer = model.text_encoder.tokenizer

    
    # Create dataloaders
    train_dataloader, val_dataloader, test_dataloader = generate_ood_qh_data(batch_size=1, target_name= 'elongation')

    
    # Only add these for multiprocessing (num_workers > 0)
    if num_workers > 0:
        train_loader_kwargs['persistent_workers'] = config['data'].get('persistent_workers', True)
        train_loader_kwargs['prefetch_factor'] = config['data'].get('prefetch_factor', 2)
    
    # Create mask generator
    mask_generator = create_mask_generator(config['data'])
    
    # Create optimizer and scheduler (with Stage-2 settings if applicable)
    optimizer = create_optimizer(model, config, stage2=stage2, stage2_lr_factor=stage2_lr_factor)
    scheduler, warmup_steps = create_scheduler(optimizer, config, len(train_dataloader))
    
    # Create gradient scaler for mixed precision
    scaler = GradScaler(enabled=True)
    
    # Load checkpoint if resuming
    start_epoch = 0
    global_step = 0
    best_metric = float('inf') if not stage2 else 0.0  # Stage-2 uses mean_recall (higher is better)
    best_mean_recall = 0.0  # Track best mean recall for Stage-2
    
    if resume_checkpoint:
        checkpoint_info = load_checkpoint(
            resume_checkpoint,
            model,
            optimizer=None if stage2 else optimizer,  # Don't load optimizer state for Stage-2
            scheduler=None if stage2 else scheduler,  # Don't load scheduler state for Stage-2
            device=device,
        )
        if stage2:
            # For Stage-2, start fresh epoch count but keep model weights
            start_epoch = 0
            global_step = 0
            logger.info(f"Stage-2: Loaded model weights from checkpoint (epoch {checkpoint_info['epoch']})")
            logger.info("Stage-2: Reset optimizer and scheduler for fine-tuning")
        else:
            start_epoch = checkpoint_info['epoch'] + 1
            global_step = checkpoint_info['global_step']
            best_metric = checkpoint_info['best_metric']
            logger.info(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    num_epochs = stage2_epochs if stage2 else config['training']['num_epochs']
    checkpoint_dir = Path(config['training'].get('checkpoint_dir', 'checkpoints'))
    checkpoint_dir.mkdir(exist_ok=True)
    save_every = config['training'].get('save_every', 5)
    
    # Early stopping for Stage-2
    early_stop_patience = 10
    epochs_without_improvement = 0
    
    logger.info("Starting training...")
    if stage2:
        logger.info(f"Stage-2: Training for {num_epochs} epochs with early stopping (patience={early_stop_patience})")
        logger.info(f"Stage-2: Loss weights LOCKED at jepa={config['training'].get('jepa_loss_weight', 1.0)}, contrastive={config['training'].get('contrastive_loss_weight', 0.5)}")
    
    for epoch in range(start_epoch, num_epochs):
        # Train
        train_loss, global_step = train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            mask_generator=mask_generator,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            config=config,
            logger=logger,
            global_step=global_step,
            warmup_steps=warmup_steps if not stage2 else 0,  # No warmup for Stage-2
            stage2=stage2,  # Pass stage2 flag for safety checks
        )
        
        # Validate (if val_loader exists)
        val_metrics = {'val_loss': float('inf'), 'mean_recall': 0.0}
        if val_dataloader is not None:
            val_metrics = validate(
                model=model,
                dataloader=val_dataloader,
                epoch=epoch,
                config=config,
                logger=logger,
            )
        
        # Determine if this is the best model
        if stage2:
            # Stage-2: Use mean_recall (higher is better)
            current_mean_recall = val_metrics.get('mean_recall', 0.0)
            is_best = current_mean_recall > best_mean_recall
            if is_best:
                best_mean_recall = current_mean_recall
                epochs_without_improvement = 0
                logger.info(f"Stage-2: New best mean_recall = {best_mean_recall:.2f}%")
            else:
                epochs_without_improvement += 1
                logger.info(f"Stage-2: No improvement for {epochs_without_improvement} epoch(s)")
            
            # Early stopping check
            if epochs_without_improvement >= early_stop_patience:
                logger.info(f"Stage-2: Early stopping triggered (no improvement for {early_stop_patience} epochs)")
                break
        else:
            # Stage-1: Use val_loss (lower is better)
            is_best = val_metrics['val_loss'] < best_metric
            if is_best:
                best_metric = val_metrics['val_loss']

        stage2 = False
        # Save checkpoint
        if stage2:
            # Stage-2: Always save each epoch, mark best
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                best_metric=best_mean_recall,
                config=config,
                save_path=checkpoint_dir / f"stage2_epoch_{epoch}.pth",
                is_best=is_best,
            )
            if is_best:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    global_step=global_step,
                    best_metric=best_mean_recall,
                    config=config,
                    save_path=checkpoint_dir / "stage2_best.pth",
                    is_best=True,
                )
        else:
            if (epoch + 1) % save_every == 0 or is_best:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    global_step=global_step,
                    best_metric=best_metric,
                    config=config,
                    save_path=checkpoint_dir / f"checkpoint_epoch_{epoch}.pth",
                    is_best=is_best,
                )
    
    logger.info("Training completed!")
    if stage2:
        logger.info(f"Stage-2 Best Mean Recall: {best_mean_recall:.2f}%")
    
    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()