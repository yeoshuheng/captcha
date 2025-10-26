import os
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt


class Trainer(ABC):
    def __init__(
        self,
        model: nn.Module,
        dataset: Dataset,
        criterion_function: Callable[..., torch.Tensor],
        optimizer: type = torch.optim.Adam,
        epochs: int = 30,
        lr: float = 1e-5,
        batch_size: int = 64,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        checkpoint_dir: str = "./checkpoints",
        collate_function: Callable[[List[Any]], Tuple] = None,
        checkpoint_interval: int = 2,
        best_model_checkpoint: bool = True,
        gradient_clip: int = -1
    ):
        self.model = model.to(device)
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.epochs = epochs
        self.best_model_checkpoint = best_model_checkpoint
        self.gradient_clip = gradient_clip
        
        self.criterion = criterion_function
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self.collate_fn = collate_function

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate_fn
        )

        if self.device.type == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()

        self.best_loss = float('inf')

        self.train_losses = []

    @abstractmethod
    def _process_results(self, results: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _save_checkpoint(self, epoch: int) -> None:
        checkpoint_path = os.path.join(self.checkpoint_dir, f"captcha_model_epoch{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    def _save_best_model(self) -> None:
        best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, best_path)
        print(f"Best model checkpoint saved: {best_path}")

    def train(self) -> None:
        self.model.train()
        for epoch in range(self.epochs):
            self.current_epoch = epoch + 1
            total_loss = 0
            progress_bar = tqdm(self.loader, desc=f"Epoch {self.current_epoch}/{self.epochs}")
            
            for batch_idx, (images, labels, target_lengths) in enumerate(progress_bar):
                images = images.to(self.device)
                labels = labels.to(self.device)

                target_lengths = target_lengths.to(self.device)

                self.optimizer.zero_grad()

                if self.device.type == "cuda":
                    with torch.cuda.amp.autocast():
                        logits = self.model(images)
                        logits = self._process_results(logits)
                        T, N, _ = logits.shape
                        input_lengths = torch.full((N, ), T, dtype=torch.long).to(self.device)
                        loss = self.criterion(logits, labels, input_lengths, target_lengths)

                    self.scaler.scale(loss).backward()

                    if self.gradient_clip != -1:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    logits = self.model(images)
                    logits = self._process_results(logits)
                    T, N, _ = logits.shape
                    input_lengths = torch.full((N, ), T, dtype=torch.long).to(self.device)
                    loss = self.criterion(logits, labels, input_lengths, target_lengths)

                    if self.gradient_clip != -1:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip)

                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()
                self.train_losses.append(loss.item())

                progress_bar.set_postfix({"loss": total_loss / (batch_idx + 1)})

            avg_loss = total_loss / len(self.loader)
            print(f"Epoch [{self.current_epoch}/{self.epochs}] - Loss: {avg_loss:.4f}")

            if (epoch + 1) % self.checkpoint_interval == 0:
                self._save_checkpoint(epoch + 1)

            if self.best_model_checkpoint and avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self._save_best_model()

        self._plot_loss()
            
    def _plot_loss(self) -> None:
        """Plot and save training loss"""
        plt.figure(figsize=(12, 5))
    
        plt.subplot(1, 2, 2)
        plt.plot(self.train_losses, 'r-', alpha=0.7, linewidth=1)
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.title('Training Loss per Batch')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.checkpoint_dir, "training_loss.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Loss plot saved to: {save_path}")


