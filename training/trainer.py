"""
Trainer loop for GeoViT.
"""

import logging
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from pathlib import Path
from tqdm import tqdm
from config import TRAIN_CONFIG, ADAM_CONFIG, MODEL_DIR

class Trainer:
    def __init__(self, model, train_loader, val_loader, device="cuda", lr=TRAIN_CONFIG["learning_rate"], epochs=TRAIN_CONFIG["epochs"], save_every=TRAIN_CONFIG["save_every_n_epochs"]):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.criterion = nn.MSELoss()
        self.optimizer = Adam(
            self.model.parameters(), 
            lr=lr, 
            **ADAM_CONFIG
        )
        self.epochs = epochs
        self.save_every = save_every
        
    def train(self, model_name="geovit"):
        best_val_loss = float('inf')
        
        for epoch in range(1, self.epochs + 1):
            # Training Phase
            self.model.train()
            train_losses = []
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs} [Train]")
            for X_batch, y_batch in pbar:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                self.optimizer.zero_grad()
                preds = self.model(X_batch)
                loss = self.criterion(preds, y_batch)
                
                loss.backward()
                self.optimizer.step()
                
                train_losses.append(loss.item())
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                
            train_loss = np.mean(train_losses)
            
            # Validation Phase
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for X_batch, y_batch in self.val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    preds = self.model(X_batch)
                    loss = self.criterion(preds, y_batch)
                    val_losses.append(loss.item())
            
            val_loss = np.mean(val_losses)
            logging.info(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(f"{model_name}_best.pth")
                
            if epoch % self.save_every == 0:
                self.save_checkpoint(f"{model_name}_epoch_{epoch}.pth")
                
        return best_val_loss
        
    def save_checkpoint(self, filename):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        path = MODEL_DIR / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        logging.info(f"Saved checkpoint to {path}")
