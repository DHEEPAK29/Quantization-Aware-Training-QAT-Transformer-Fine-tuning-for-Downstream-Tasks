import torch
from torch.utils.data import DataLoader
from transformers import default_data_collator
from datasets import load_dataset
from tqdm import tqdm
import json
from pathlib import Path
import logging

from model import get_phi2_qat_model, QATConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingConfig:
    """Training hyperparameters."""
    # Data
    dataset_name = "wikitext"
    dataset_config = "wikitext-2-raw-v1"
    max_length = 512
    train_samples = 1000  # Limit for quick experimentation
    
    # Training
    num_epochs = 3
    batch_size = 2
    gradient_accumulation_steps = 4
    learning_rate = 1e-5
    weight_decay = 0.01
    warmup_steps = 100
    max_grad_norm = 1.0
    
    # Logging
    log_interval = 10
    save_interval = 500
    output_dir = Path("./qat_outputs")
    
    # Device
    mixed_precision = True  # Use automatic mixed precision


def prepare_dataset(config: TrainingConfig, tokenizer):
    """
    Load and prepare dataset for training.
    
    Args:
        config: TrainingConfig instance
        tokenizer: HuggingFace tokenizer
        
    Returns:
        DataLoader for training
    """
    logger.info(f"Loading dataset: {config.dataset_name}")
    
    # Load dataset
    dataset = load_dataset(
        config.dataset_name,
        config.dataset_config,
        split="train"
    )
    
    # Limit samples for experimentation
    if config.train_samples:
        dataset = dataset.select(range(min(config.train_samples, len(dataset))))
    
    logger.info(f"Dataset size: {len(dataset)} samples")
    
    # Tokenization function
    def tokenize_function(examples):
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=config.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        # For causal LM, labels are the same as input_ids
        outputs["labels"] = outputs["input_ids"].clone()
        return outputs
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    # Create dataloader
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=default_data_collator,
    )
    
    return dataloader

def log_layer_weights(model, epoch, step, output_dir):
    """
    Log weight distributions from key layers.
    
    Args:
        model: The QAT model
        epoch: Current epoch
        step: Current step
        output_dir: Directory to save weight tensors
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Log weights from first transformer layer
    layer_0 = model.model.layers[0]
    
    # Log MLP weights
    if hasattr(layer_0.mlp, 'fc1'):
        weights = layer_0.mlp.fc1.weight.detach().cpu().flatten()
        torch.save(
            weights,
            output_dir / f"mlp_fc1_epoch{epoch}_step{step}.pt"
        )
    
    # Log attention weights
    if hasattr(layer_0.self_attn, 'q_proj'):
        weights = layer_0.self_attn.q_proj.weight.detach().cpu().flatten()
        torch.save(
            weights,
            output_dir / f"attn_q_epoch{epoch}_step{step}.pt"
        )


def compute_weight_statistics(model):
    """
    Compute statistics about weight clustering.
    
    Args:
        model: The QAT model
        
    Returns:
        dict: Statistics about weight distributions
    """
    stats = {}
    
    layer_0 = model.model.layers[0]
    
    if hasattr(layer_0.mlp, 'fc1'):
        weights = layer_0.mlp.fc1.weight.detach().cpu().flatten()
        stats['mlp_fc1'] = {
            'mean': weights.mean().item(),
            'std': weights.std().item(),
            'min': weights.min().item(),
            'max': weights.max().item(),
            'unique_values': len(torch.unique(weights)),
        }
    
    return stats


def train_qat_model(
    model,
    tokenizer,
    train_config: TrainingConfig = None,
):
    """
    Main training loop for QAT.
    
    Args:
        model: QAT-prepared model
        tokenizer: Tokenizer
        train_config: TrainingConfig instance
    """
    if train_config is None:
        train_config = TrainingConfig()
    
    # Setup output directory
    train_config.output_dir.mkdir(exist_ok=True, parents=True)
    
    # Prepare dataset
    train_dataloader = prepare_dataset(train_config, tokenizer)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    
    # Learning rate scheduler
    from transformers import get_linear_schedule_with_warmup
    total_steps = len(train_dataloader) * train_config.num_epochs // train_config.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=train_config.warmup_steps,
        num_training_steps=total_steps,
    )
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if train_config.mixed_precision else None
    
    # Training metrics
    metrics = {
        'losses': [],
        'weight_stats': [],
    }
    
    logger.info("Starting QAT training...")
    logger.info(f"Total epochs: {train_config.num_epochs}")
    logger.info(f"Steps per epoch: {len(train_dataloader)}")
    logger.info(f"Gradient accumulation: {train_config.gradient_accumulation_steps}")
    
    model.train()
    global_step = 0
    
    for epoch in range(train_config.num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{train_config.num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(model.device) for k, v in batch.items()}
            
            # Forward pass with mixed precision
            if train_config.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
                    loss = outputs.loss / train_config.gradient_accumulation_steps
            else:
                outputs = model(**batch)
                loss = outputs.loss / train_config.gradient_accumulation_steps
            
            # Backward pass
            if train_config.mixed_precision:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            epoch_loss += loss.item()
            
            # Gradient accumulation
            if (step + 1) % train_config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if train_config.mixed_precision:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        train_config.max_grad_norm
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        train_config.max_grad_norm
                    )
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Logging
                if global_step % train_config.log_interval == 0:
                    current_loss = loss.item() * train_config.gradient_accumulation_steps
                    current_lr = scheduler.get_last_lr()[0]
                    
                    metrics['losses'].append({
                        'epoch': epoch,
                        'step': global_step,
                        'loss': current_loss,
                        'lr': current_lr,
                    })
                    
                    progress_bar.set_postfix({
                        'loss': f"{current_loss:.4f}",
                        'lr': f"{current_lr:.2e}",
                    })
                
                # Save weights periodically
                if global_step % train_config.save_interval == 0:
                    log_layer_weights(
                        model, epoch, global_step,
                        train_config.output_dir / "weight_logs"
                    )
                    
                    # Compute and save statistics
                    stats = compute_weight_statistics(model)
                    stats['step'] = global_step
                    stats['epoch'] = epoch
                    metrics['weight_stats'].append(stats)
        
        # End of epoch logging
        avg_loss = epoch_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
        
        # Save epoch weights
        log_layer_weights(
            model, epoch, "final",
            train_config.output_dir / "weight_logs"
        )
    
    # Save final metrics
    with open(train_config.output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info("Training completed!")
    
    return model, metrics


if __name__ == "__main__":
    # Load QAT model
    logger.info("Initializing QAT model...")
    model, tokenizer = get_phi2_qat_model()
    
    # Train
    trained_model, metrics = train_qat_model(model, tokenizer)
    
    # Save final model
    logger.info("Saving final model...")
    trained_model.save_pretrained(TrainingConfig.output_dir / "final_model")
    tokenizer.save_pretrained(TrainingConfig.output_dir / "final_model")
    
    logger.info("All done!")