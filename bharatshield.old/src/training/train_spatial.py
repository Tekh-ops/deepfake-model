# src/training/train_spatial.py

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import mlflow
from pathlib import Path

def train_spatial_model(config: dict):
    mlflow.set_experiment("spatial_detector")
    
    with mlflow.start_run(run_name=config['run_name']):
        mlflow.log_params(config)
        
        model = SpatialDetector(
            model_name=config['backbone'],
            dropout=config['dropout']
        ).cuda()
        
        # Focal loss to handle class imbalance
        criterion = FocalLoss(gamma=2.0, alpha=0.25)
        
        # Optimizer: AdamW with weight decay
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
        
        # Cosine annealing with warm restarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2
        )
        
        scaler = GradScaler()  # mixed precision
        
        best_auc = 0.0
        patience_counter = 0
        
        for epoch in range(config['max_epochs']):
            # Training
            model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                images, labels, weights = batch
                images = images.cuda()
                labels = labels.float().cuda()
                weights = weights.cuda()
                
                optimizer.zero_grad()
                
                with autocast():
                    logits, _ = model(images)
                    loss = criterion(logits, labels, weights)
                
                scaler.scale(loss).backward()
                
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()
            
            scheduler.step()
            
            # Validation
            val_auc = evaluate(model, val_loader)
            
            mlflow.log_metrics({
                "train_loss": train_loss / len(train_loader),
                "val_auc": val_auc,
                "lr": scheduler.get_last_lr()[0]
            }, step=epoch)
            
            # Early stopping
            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                torch.save(model.state_dict(), 
                          f"checkpoints/spatial_best.pt")
                mlflow.log_artifact("checkpoints/spatial_best.pt")
            else:
                patience_counter += 1
                if patience_counter >= config['patience']:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        return best_auc