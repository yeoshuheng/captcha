"""
Complete CAPTCHA Recognition System using CTC Loss
Usage:
    python captcha_ctc.py --train_dir ./train --test_dir ./test --epochs 20 --lr 0.001
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import string
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
import argparse
import os
from torch.optim.lr_scheduler import LambdaLR
import cv2
import numpy as np

# ============================================================================
# Character set for CAPTCHA (alphanumeric)
# ============================================================================
CHARSET = string.digits + string.ascii_lowercase + string.ascii_uppercase
CHAR_TO_IDX = {char: idx + 1 for idx, char in enumerate(CHARSET)}  # 0 reserved for CTC blank
IDX_TO_CHAR = {idx + 1: char for idx, char in enumerate(CHARSET)}
NUM_CLASSES = len(CHARSET) + 1  # +1 for CTC blank token

# ============================================================================
# CAPTCHA preprocessing
# ============================================================================
class CaptchaPreprocess:
    def __init__(self, img_height=32, img_width=128):
        self.img_height = img_height
        self.img_width = img_width
        self.normalize = T.Normalize(mean=[0.5], std=[0.5])

    def __call__(self, img):
        img = np.array(img.convert('RGB'))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.medianBlur(img, 3)
        img = cv2.adaptiveThreshold(
            img, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )
        kernel = np.ones((2,2), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        img = cv2.resize(img, (self.img_width, self.img_height))
        img = torch.from_numpy(img).unsqueeze(0).float() / 255.0
        img = self.normalize(img)
        return img

# ============================================================================
# Dataset
# ============================================================================
class CaptchaDataset(Dataset):
    def __init__(self, image_dir, img_height=32, img_width=128):
        self.image_paths = []
        self.labels = []
        for path in Path(image_dir).glob("*.png"):
            self.image_paths.append(str(path))
            label = path.stem.split('-')[0]  
            self.labels.append(label)
        self.transform = CaptchaPreprocess(img_height, img_width)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx])
        img = self.transform(img)

        label = self.labels[idx]
        label_indices = torch.tensor([CHAR_TO_IDX[c] for c in label], dtype=torch.long)

        return img, label_indices

# ============================================================================
# Collate function for variable-length CTC labels
# ============================================================================
def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images, dim=0)
    labels_concat = torch.cat(labels)
    label_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
    return images, labels_concat, label_lengths

# ============================================================================
# Model Architecture
# ============================================================================
import torch
import torch.nn as nn

class SmallCRNNModel(nn.Module):
    """
    Smaller CRNN for CAPTCHA recognition.
    """
    def __init__(self, num_classes, hidden_size=128):
        super().__init__()

        # CNN feature extractor (smaller channels)
        self.cnn = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),  # H/2, W/2

            # Conv Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)),  # H/4, W/2

            # Conv Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)),  # H/8, W/2
        )

        # LSTM expects input shape (seq_len, batch, features)
        self.rnn = nn.LSTM(
            input_size=128*2,  # assuming height after H/8 = 2, channels=128
            hidden_size=hidden_size,
            num_layers=1,  # smaller
            bidirectional=True,
            batch_first=False
        )

        self.fc = nn.Linear(hidden_size*2, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        conv_features = self.cnn(x)  # (batch, channels, H, W)
        batch, channels, height, width = conv_features.size()
        conv_features = conv_features.permute(3, 0, 1, 2)  # (W, batch, C, H)
        conv_features = conv_features.reshape(width, batch, channels*height)
        rnn_out, _ = self.rnn(conv_features)
        output = self.fc(rnn_out)  # (seq_len=W, batch, num_classes)
        return output



# ============================================================================
# Training
# ============================================================================

def train_model(model, train_loader, criterion, optimizer, device, epoch, debug=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, (images, labels, label_lengths) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)
        
        # Forward pass
        outputs = model(images)  # (seq_len, batch, num_classes)
        
        # Debug first batch of first epoch
        if debug and batch_idx == 0:
            print(f"\n[TRAIN DEBUG] Outputs shape: {outputs.shape}")
            print(f"[TRAIN DEBUG] Outputs min/max: {outputs.min():.4f} / {outputs.max():.4f}")
            print(f"[TRAIN DEBUG] Label lengths: {label_lengths.tolist()[:5]}")
            print(f"[TRAIN DEBUG] Sequence length: {outputs.size(0)}")
        
        # Apply log_softmax for CTC
        log_probs = F.log_softmax(outputs, dim=2)
        
        if debug and batch_idx == 0:
            print(f"[TRAIN DEBUG] Log probs min/max: {log_probs.min():.4f} / {log_probs.max():.4f}")
            argmax_preds = log_probs.argmax(dim=2)
            blank_ratio = (argmax_preds == 0).float().mean().item()
            print(f"[TRAIN DEBUG] Blank ratio in predictions: {blank_ratio:.2%}")
        
        # Input lengths (sequence length from model output)
        seq_len = log_probs.size(0)
        input_lengths = torch.full((log_probs.size(1),), seq_len, dtype=torch.long, device=device)
        
        # Compute CTC loss
        loss = criterion(log_probs, labels, input_lengths, label_lengths)
        
        if debug and batch_idx == 0:
            print(f"[TRAIN DEBUG] Loss: {loss.item():.4f}")
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


# ============================================================================
# Decoding
# ============================================================================

def ctc_decode(predictions):
    """
    CTC greedy decoding: remove blanks and repeated characters.
    
    Args:
        predictions: (seq_len, batch, num_classes) tensor
    
    Returns:
        List of decoded strings
    """
    # Get the most likely class at each timestep
    _, max_indices = predictions.max(dim=2)  # (seq_len, batch)
    max_indices = max_indices.transpose(0, 1)  # (batch, seq_len)
    
    decoded_strings = []
    for sequence in max_indices:
        chars = []
        prev_idx = None
        
        for idx in sequence:
            idx = idx.item()
            # Skip blank (0) and repeated characters
            if idx != 0 and idx != prev_idx:
                if idx in IDX_TO_CHAR:
                    chars.append(IDX_TO_CHAR[idx])
            prev_idx = idx
        
        decoded_strings.append(''.join(chars))
    
    return decoded_strings


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_model(model, test_loader, device, debug=False):
    """Evaluate model on test set."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (images, labels_concat, label_lengths) in enumerate(tqdm(test_loader, desc="Evaluating")):
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            log_probs = F.log_softmax(outputs, dim=2)
            
            # Debug first batch
            if debug and batch_idx == 0:
                print(f"\n[DEBUG] Output shape: {outputs.shape}")  # (seq_len, batch, num_classes)
                print(f"[DEBUG] Log probs shape: {log_probs.shape}")
                print(f"[DEBUG] Log probs min/max: {log_probs.min():.4f} / {log_probs.max():.4f}")
                
                # Check argmax predictions
                argmax_preds = log_probs.argmax(dim=2)  # (seq_len, batch)
                print(f"[DEBUG] Argmax shape: {argmax_preds.shape}")
                print(f"[DEBUG] Unique predictions: {torch.unique(argmax_preds).tolist()[:20]}")
                print(f"[DEBUG] Blank ratio: {(argmax_preds == 0).float().mean().item():.2%}")
                print(f"[DEBUG] First sequence (first 20 steps): {argmax_preds[:20, 0].tolist()}")
            
            # Decode predictions
            predictions = ctc_decode(log_probs)
            
            # Reconstruct true labels
            true_labels = []
            start_idx = 0
            for length in label_lengths:
                label_seq = labels_concat[start_idx:start_idx + length]
                label_str = ''.join([IDX_TO_CHAR[idx.item()] for idx in label_seq])
                true_labels.append(label_str)
                start_idx += length
            
            # Debug first batch predictions
            if debug and batch_idx == 0:
                print(f"\n[DEBUG] Sample predictions vs true labels:")
                for i, (pred, true) in enumerate(zip(predictions[:5], true_labels[:5])):
                    print(f"  [{i}] Pred: '{pred}' | True: '{true}' | Match: {pred == true}")
            
            # Calculate accuracy
            for pred, true in zip(predictions, true_labels):
                if pred == true:
                    correct += 1
                total += 1
    
    accuracy = 100 * correct / total
    return accuracy, correct, total

def get_linear_warmup_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps))
        )
    return LambdaLR(optimizer, lr_lambda)

# ============================================================================
# Main Training Script
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train CAPTCHA recognition model with CTC')
    parser.add_argument('--train_dir', type=str, required=True, help='Training data directory')
    parser.add_argument('--test_dir', type=str, required=True, help='Test data directory')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate (default: 0.0005)')
    parser.add_argument('--hidden_size', type=int, default=256, help='RNN hidden size')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Checkpoint directory')
    args = parser.parse_args()
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = CaptchaDataset(args.train_dir)
    test_dataset = CaptchaDataset(args.test_dir)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Number of classes: {NUM_CLASSES} (including blank)")
    
    # Initialize model
    model = CRNNModel(num_classes=NUM_CLASSES, hidden_size=args.hidden_size)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.05 * total_steps)
    scheduler = get_linear_warmup_scheduler(optimizer, warmup_steps, total_steps)
    
    # Training loop
    best_accuracy = 0.0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss = train_model(model, train_loader, criterion, optimizer, device, epoch, debug=(epoch % 10 == 0))
        print(f"Train Loss: {train_loss:.4f}")
        
        # Evaluate
        accuracy, correct, total = evaluate_model(model, test_loader, device, debug=(epoch % 10 == 0))
        print(f"Test Accuracy: {accuracy:.2f}% ({correct}/{total})")
        
        # Learning rate scheduling
        if epoch >= 10:
            scheduler.step(train_loss)
        
        # Save checkpoint
        if epoch % 5 == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'model_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'accuracy': accuracy
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'accuracy': accuracy
            }, best_path)
            print(f"★ New best model saved! Accuracy: {accuracy:.2f}%")
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best accuracy: {best_accuracy:.2f}%")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()