"""
Improved CAPTCHA Recognition System
Key improvements:
1. Better architecture with preserved spatial resolution
2. More stable training with proper initialization
3. Focal CTC loss to handle class imbalance
4. Better data augmentation
5. Comprehensive debugging tools
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
import cv2
import numpy as np

# ============================================================================
# Character set for CAPTCHA
# ============================================================================
CHARSET = string.digits + string.ascii_lowercase + string.ascii_uppercase
CHAR_TO_IDX = {char: idx + 1 for idx, char in enumerate(CHARSET)}
IDX_TO_CHAR = {idx + 1: char for idx, char in enumerate(CHARSET)}
NUM_CLASSES = len(CHARSET) + 1  # +1 for CTC blank token

print(f"Character set size: {len(CHARSET)}")
print(f"Total classes (with blank): {NUM_CLASSES}")

# ============================================================================
# Enhanced Preprocessing with Augmentation
# ============================================================================
class CaptchaPreprocess:
    def __init__(self, img_height=64, img_width=200, augment=False):
        self.img_height = img_height
        self.img_width = img_width
        self.augment = augment
        self.normalize = T.Normalize(mean=[0.5], std=[0.5])

    def __call__(self, img):
        # Convert to grayscale
        img = np.array(img.convert('L'))
        
        # Basic preprocessing
        img = cv2.resize(img, (self.img_width, self.img_height))
        
        # Optional augmentation for training
        if self.augment:
            # Random slight rotation
            if np.random.rand() < 0.3:
                angle = np.random.uniform(-5, 5)
                M = cv2.getRotationMatrix2D((self.img_width/2, self.img_height/2), angle, 1.0)
                img = cv2.warpAffine(img, M, (self.img_width, self.img_height), 
                                    borderMode=cv2.BORDER_REPLICATE)
            
            # Random brightness
            if np.random.rand() < 0.3:
                factor = np.random.uniform(0.8, 1.2)
                img = np.clip(img * factor, 0, 255).astype(np.uint8)
        
        # CLAHE for contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        
        # Convert to tensor
        img = torch.from_numpy(img).unsqueeze(0).float() / 255.0
        img = self.normalize(img)
        return img

# ============================================================================
# Dataset
# ============================================================================
class CaptchaDataset(Dataset):
    def __init__(self, image_dir, img_height=64, img_width=200, max_label_len=10, augment=False):
        self.image_paths = []
        self.labels = []
        self.max_label_len = max_label_len
        
        skipped = 0
        for path in Path(image_dir).glob("*.png"):
            label = path.stem.split('-')[0]
            
            if len(label) > max_label_len or len(label) == 0:
                skipped += 1
                continue
            if not all(c in CHARSET for c in label):
                skipped += 1
                continue
                
            self.image_paths.append(str(path))
            self.labels.append(label)
        
        if skipped > 0:
            print(f"Skipped {skipped} invalid samples")
        
        # Calculate statistics
        label_lengths = [len(label) for label in self.labels]
        print(f"Label length range: {min(label_lengths)} - {max(label_lengths)}")
        print(f"Average label length: {np.mean(label_lengths):.1f}")
        
        self.transform = CaptchaPreprocess(img_height, img_width, augment=augment)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx])
        img = self.transform(img)
        
        label = self.labels[idx]
        label_indices = torch.tensor([CHAR_TO_IDX[c] for c in label], dtype=torch.long)
        
        return img, label_indices, label

def collate_fn(batch):
    images, labels, label_strs = zip(*batch)
    images = torch.stack(images, dim=0)
    labels_concat = torch.cat(labels)
    label_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
    return images, labels_concat, label_lengths, label_strs

# ============================================================================
# Improved CNN Architecture - Preserves More Spatial Info
# ============================================================================
class ImprovedCNN(nn.Module):
    """CNN that maintains good sequence length for CTC"""
    def __init__(self):
        super().__init__()
        
        # Block 1: 64x200 -> 32x100
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 2: 32x100 -> 16x50
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 3: 16x50 -> 8x50 (only pool height)
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )
        
        # Block 4: 8x50 -> 4x50 (only pool height)
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )
        
        # Block 5: Keep spatial resolution
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

# ============================================================================
# Complete CRNN Model
# ============================================================================
class ImprovedCRNN(nn.Module):
    def __init__(self, num_classes, hidden_size=256, img_height=64, img_width=200):
        super().__init__()
        
        self.img_height = img_height
        self.img_width = img_width
        
        # CNN
        self.cnn = ImprovedCNN()
        
        # Calculate dimensions
        self.rnn_input_size = self._get_rnn_input_size()
        self.sequence_length = self._get_sequence_length()
        
        print(f"CNN output: {self.sequence_length} timesteps")
        print(f"RNN input size: {self.rnn_input_size}")
        
        # Bidirectional LSTM with 2 layers
        self.rnn = nn.LSTM(
            input_size=self.rnn_input_size,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            dropout=0.1,
            batch_first=False
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
        self._initialize_weights()
    
    def _get_sequence_length(self):
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.img_height, self.img_width)
            features = self.cnn(dummy)
            return features.size(3)
    
    def _get_rnn_input_size(self):
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.img_height, self.img_width)
            features = self.cnn(dummy)
            return features.size(1) * features.size(2)
    
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
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

    def forward(self, x):
        # CNN features
        conv_features = self.cnn(x)
        batch, channels, height, width = conv_features.size()
        
        # Reshape for RNN: (width, batch, channels*height)
        conv_features = conv_features.permute(3, 0, 1, 2)
        conv_features = conv_features.reshape(width, batch, channels * height)
        
        # RNN
        rnn_out, _ = self.rnn(conv_features)
        
        # Classifier
        output = self.fc(rnn_out)
        return output

# ============================================================================
# Focal CTC Loss - Better than blank penalty
# ============================================================================
class FocalCTCLoss(nn.Module):
    def __init__(self, blank=0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.blank = blank
        self.gamma = gamma
        self.reduction = reduction
        self.ctc_loss = nn.CTCLoss(blank=blank, reduction='none', zero_infinity=True)
    
    def forward(self, log_probs, targets, input_lengths, target_lengths):
        # Standard CTC loss
        losses = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        
        # Apply focal weighting
        p = torch.exp(-losses)
        focal_weight = (1 - p) ** self.gamma
        focal_losses = focal_weight * losses
        
        if self.reduction == 'mean':
            return focal_losses.mean()
        elif self.reduction == 'sum':
            return focal_losses.sum()
        else:
            return focal_losses

# ============================================================================
# Training
# ============================================================================
def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0
    valid_batches = 0
    blank_predictions = []
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for images, labels, label_lengths, label_strs in pbar:
        images = images.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)
        
        # Forward
        outputs = model(images)
        log_probs = F.log_softmax(outputs, dim=2)
        
        # Monitor blank ratio
        with torch.no_grad():
            pred_classes = log_probs.argmax(dim=2)
            blank_ratio = (pred_classes == 0).float().mean().item()
            blank_predictions.append(blank_ratio)
        
        # CTC loss
        seq_len = log_probs.size(0)
        batch_size = log_probs.size(1)
        input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)
        
        # Verify sequence length is sufficient
        if seq_len < label_lengths.max().item():
            print(f"\nWARNING: seq_len {seq_len} < max_label_len {label_lengths.max().item()}")
            continue
        
        try:
            loss = criterion(log_probs, labels, input_lengths, label_lengths)
        except RuntimeError as e:
            print(f"\nCTC Error: {e}")
            continue
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\nSkipping batch: invalid loss")
            continue
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        if scheduler:
            scheduler.step()
        
        total_loss += loss.item()
        valid_batches += 1
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'blank%': f'{blank_ratio*100:.1f}'
        })
    
    avg_loss = total_loss / max(valid_batches, 1)
    avg_blank = np.mean(blank_predictions) if blank_predictions else 0
    
    return avg_loss, avg_blank

# ============================================================================
# CTC Decoding
# ============================================================================
def ctc_decode(predictions):
    _, max_indices = predictions.max(dim=2)
    max_indices = max_indices.transpose(0, 1)
    
    decoded_strings = []
    for sequence in max_indices:
        chars = []
        prev_idx = None
        
        for idx in sequence:
            idx = idx.item()
            if idx != 0 and idx != prev_idx:
                if idx in IDX_TO_CHAR:
                    chars.append(IDX_TO_CHAR[idx])
            prev_idx = idx
        
        decoded_strings.append(''.join(chars))
    
    return decoded_strings

# ============================================================================
# Evaluation
# ============================================================================
def evaluate(model, test_loader, device, show_examples=False):
    model.eval()
    correct = 0
    total = 0
    char_correct = 0
    char_total = 0
    
    predictions_list = []
    labels_list = []
    
    with torch.no_grad():
        for images, _, _, label_strs in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            
            outputs = model(images)
            log_probs = F.log_softmax(outputs, dim=2)
            predictions = ctc_decode(log_probs)
            
            for pred, true in zip(predictions, label_strs):
                predictions_list.append(pred)
                labels_list.append(true)
                
                if pred == true:
                    correct += 1
                total += 1
                
                # Character accuracy
                min_len = min(len(pred), len(true))
                for i in range(min_len):
                    if pred[i] == true[i]:
                        char_correct += 1
                char_total += max(len(pred), len(true))
    
    accuracy = 100 * correct / total if total > 0 else 0
    char_acc = 100 * char_correct / char_total if char_total > 0 else 0
    
    if show_examples:
        print("\n" + "="*70)
        print("Sample Predictions:")
        print("="*70)
        for i in range(min(20, len(predictions_list))):
            match = "✓" if predictions_list[i] == labels_list[i] else "✗"
            print(f"{match} Pred: {predictions_list[i]:12s} | True: {labels_list[i]}")
        print(f"\nSequence Accuracy: {accuracy:.2f}%")
        print(f"Character Accuracy: {char_acc:.2f}%")
    
    return accuracy, char_acc

# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, required=True)
    parser.add_argument('--test_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--img_height', type=int, default=64)
    parser.add_argument('--img_width', type=int, default=200)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    args = parser.parse_args()
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Datasets with augmentation for training
    train_dataset = CaptchaDataset(args.train_dir, args.img_height, args.img_width, augment=True)
    test_dataset = CaptchaDataset(args.test_dir, args.img_height, args.img_width, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                             collate_fn=collate_fn, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=4, pin_memory=True)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}\n")
    
    # Model
    model = ImprovedCRNN(NUM_CLASSES, args.hidden_size, args.img_height, args.img_width)
    model = model.to(device)
    
    # Check if sequence length is sufficient
    max_label_len = max([len(label) for label in train_dataset.labels])
    print(f"Max label length: {max_label_len}")
    print(f"Model sequence length: {model.sequence_length}")
    if model.sequence_length < max_label_len * 2:
        print("⚠️  WARNING: Sequence length might be too short!")
    
    # Focal CTC Loss
    criterion = FocalCTCLoss(blank=0, gamma=2.0)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    best_accuracy = 0.0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*70}")
        
        train_loss, blank_ratio = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch
        )
        print(f"Train Loss: {train_loss:.4f} | Blank Ratio: {blank_ratio*100:.1f}%")
        
        accuracy, char_acc = evaluate(model, test_loader, device, show_examples=(epoch % 10 == 0))
        print(f"Test Accuracy: {accuracy:.2f}% | Char Accuracy: {char_acc:.2f}%")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'accuracy': accuracy,
                'char_accuracy': char_acc
            }, os.path.join(args.checkpoint_dir, 'best_model.pth'))
            print(f"★ New best model saved! Accuracy: {accuracy:.2f}%")
    
    print(f"\n{'='*70}")
    print(f"Training Complete! Best Accuracy: {best_accuracy:.2f}%")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()