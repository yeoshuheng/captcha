"""
Fixed CAPTCHA Recognition System - Prevents CTC Blank Token Collapse
Key improvements:
1. Added spatial attention mechanism
2. Residual connections for deeper CNN
3. Better initialization to avoid saturation
4. Blank penalty to discourage excessive blank predictions
5. Monitoring blank token usage
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
# Character set for CAPTCHA (alphanumeric)
# ============================================================================
CHARSET = string.digits + string.ascii_lowercase + string.ascii_uppercase
CHAR_TO_IDX = {char: idx + 1 for idx, char in enumerate(CHARSET)}
IDX_TO_CHAR = {idx + 1: char for idx, char in enumerate(CHARSET)}
NUM_CLASSES = len(CHARSET) + 1  # +1 for CTC blank token (index 0)

# ============================================================================
# CAPTCHA preprocessing (simplified)
# ============================================================================
class CaptchaPreprocess:
    def __init__(self, img_height=32, img_width=128):
        self.img_height = img_height
        self.img_width = img_width
        self.normalize = T.Normalize(mean=[0.5], std=[0.5])

    def __call__(self, img):
        # Convert to grayscale
        img = np.array(img.convert('L'))
        
        # Light denoising only
        img = cv2.medianBlur(img, 3)
        
        # Resize
        img = cv2.resize(img, (self.img_width, self.img_height))
        
        # Convert to tensor and normalize
        img = torch.from_numpy(img).unsqueeze(0).float() / 255.0
        img = self.normalize(img)
        return img

# ============================================================================
# Dataset with validation
# ============================================================================
class CaptchaDataset(Dataset):
    def __init__(self, image_dir, img_height=32, img_width=128, max_label_len=10):
        self.image_paths = []
        self.labels = []
        self.max_label_len = max_label_len
        
        skipped = 0
        for path in Path(image_dir).glob("*.png"):
            label = path.stem.split('-')[0]
            
            # Validate label
            if len(label) > max_label_len:
                skipped += 1
                continue
            if not all(c in CHARSET for c in label):
                skipped += 1
                continue
                
            self.image_paths.append(str(path))
            self.labels.append(label)
        
        if skipped > 0:
            print(f"Skipped {skipped} samples with invalid labels")
        
        self.transform = CaptchaPreprocess(img_height, img_width)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx])
        img = self.transform(img)
        
        label = self.labels[idx]
        label_indices = torch.tensor([CHAR_TO_IDX[c] for c in label], dtype=torch.long)
        
        return img, label_indices, label

# ============================================================================
# Collate Function
# ============================================================================
def collate_fn(batch):
    images, labels, label_strs = zip(*batch)
    images = torch.stack(images, dim=0)
    labels_concat = torch.cat(labels)
    label_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
    return images, labels_concat, label_lengths, label_strs

# ============================================================================
# Spatial Attention Module
# ============================================================================
class SpatialAttention(nn.Module):
    """Spatial attention to focus on character regions"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        attention = self.conv(x)
        attention = self.sigmoid(attention)
        return x * attention

# ============================================================================
# Residual Block
# ============================================================================
class ResidualBlock(nn.Module):
    """Residual block for deeper CNN without gradient issues"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# ============================================================================
# Enhanced Model with Attention and Residual Connections
# ============================================================================
class EnhancedCRNNModel(nn.Module):
    def __init__(self, num_classes, hidden_size=256, img_height=32, img_width=128):
        super().__init__()
        
        self.img_height = img_height
        self.img_width = img_width
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        
        # Residual blocks for deeper feature extraction
        self.res1 = ResidualBlock(64, 128, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        
        self.res2 = ResidualBlock(128, 256, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        
        self.res3 = ResidualBlock(256, 256, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        
        self.res4 = ResidualBlock(256, 512, stride=1)
        
        # Spatial attention mechanism
        self.attention = SpatialAttention(512)
        
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        
        self.rnn_input_size = self._get_rnn_input_size(img_height, img_width)
        self.sequence_length = self._get_sequence_length(img_height, img_width)
        
        print(f"RNN input size: {self.rnn_input_size}")
        print(f"Output sequence length: {self.sequence_length}")
        
        # Bidirectional LSTM
        self.rnn = nn.LSTM(
            input_size=self.rnn_input_size,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            dropout=0.3,
            batch_first=False
        )
        
        # Final classification layer
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        # Bias blank token to be less likely initially
        with torch.no_grad():
            self.fc.bias[0] = -2.0  # Make blank less likely
    
    def _get_sequence_length(self, img_height, img_width):
        with torch.no_grad():
            dummy = torch.zeros(1, 1, img_height, img_width)
            features = self.forward_cnn(dummy)
            _, _, _, w = features.shape
            return w
    
    def _get_rnn_input_size(self, img_height, img_width):
        with torch.no_grad():
            dummy = torch.zeros(1, 1, img_height, img_width)
            features = self.forward_cnn(dummy)
            _, c, h, w = features.shape
            return c * h
    
    def forward_cnn(self, x):
        """Forward pass through CNN with residual connections and attention"""
        x = self.conv1(x)
        x = self.res1(x)
        x = self.pool1(x)
        x = self.res2(x)
        x = self.pool2(x)
        x = self.res3(x)
        x = self.pool3(x)
        x = self.res4(x)
        x = self.attention(x)  # Apply spatial attention
        x = self.pool4(x)
        return x
    
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
                # Smaller initialization to prevent saturation
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

    def forward(self, x):
        # CNN with residual connections and attention
        conv_features = self.forward_cnn(x)
        batch, channels, height, width = conv_features.size()
        
        # Reshape for RNN
        conv_features = conv_features.permute(3, 0, 1, 2)
        conv_features = conv_features.reshape(width, batch, channels * height)
        
        # RNN
        rnn_out, _ = self.rnn(conv_features)
        
        # Classifier
        output = self.fc(rnn_out)
        return output

# ============================================================================
# CTC Loss with Blank Penalty
# ============================================================================
class CTCLossWithBlankPenalty(nn.Module):
    def __init__(self, blank=0, reduction='mean', zero_infinity=True, blank_penalty=0.0):
        super().__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank, reduction=reduction, zero_infinity=zero_infinity)
        self.blank_penalty = blank_penalty
        self.blank = blank
    
    def forward(self, log_probs, targets, input_lengths, target_lengths):
        # Standard CTC loss
        ctc_loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        
        # Add penalty for predicting blank tokens
        if self.blank_penalty > 0:
            # Get probabilities for blank token
            blank_probs = torch.exp(log_probs[:, :, self.blank])
            blank_penalty = self.blank_penalty * blank_probs.mean()
            return ctc_loss + blank_penalty
        
        return ctc_loss

# ============================================================================
# Training with Blank Token Monitoring
# ============================================================================
def train_model(model, train_loader, criterion, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0
    valid_batches = 0
    total_blank_ratio = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, (images, labels, label_lengths, label_strs) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)
        
        # Forward pass
        outputs = model(images)
        log_probs = F.log_softmax(outputs, dim=2)
        
        # Monitor blank token predictions
        with torch.no_grad():
            predicted_classes = log_probs.argmax(dim=2)
            blank_ratio = (predicted_classes == 0).float().mean().item()
            total_blank_ratio += blank_ratio
        
        # CTC setup
        seq_len = log_probs.size(0)
        batch_size = log_probs.size(1)
        input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)
        
        # Validate
        max_label_len = label_lengths.max().item()
        if seq_len < max_label_len:
            print(f"\nERROR: seq_len ({seq_len}) < max_label_len ({max_label_len})")
            continue
        
        # Compute loss
        try:
            loss = criterion(log_probs, labels, input_lengths, label_lengths)
        except RuntimeError as e:
            print(f"\nCTC Loss Error: {e}")
            continue
        
        # Check for invalid loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\nSkipping batch {batch_idx}: invalid loss")
            continue
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        valid_batches += 1
        current_lr = optimizer.param_groups[0]['lr']
        
        # Show blank ratio in progress bar
        avg_blank = total_blank_ratio / valid_batches
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'blank%': f'{blank_ratio*100:.1f}',
            'lr': f'{current_lr:.6f}'
        })
    
    avg_loss = total_loss / max(valid_batches, 1)
    avg_blank_ratio = total_blank_ratio / max(valid_batches, 1)
    
    # Warning if too many blanks
    if avg_blank_ratio > 0.8:
        print(f"\n⚠️  WARNING: {avg_blank_ratio*100:.1f}% blank predictions - model may be collapsing!")
    
    return avg_loss, avg_blank_ratio

# ============================================================================
# CTC Decoding
# ============================================================================
def ctc_decode(predictions):
    """CTC greedy decoding with blank removal"""
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
def evaluate_model(model, test_loader, device, show_examples=False):
    model.eval()
    correct = 0
    total = 0
    char_correct = 0
    char_total = 0
    total_blank_ratio = 0
    
    all_predictions = []
    all_true_labels = []
    
    with torch.no_grad():
        for batch_idx, (images, labels_concat, label_lengths, label_strs) in enumerate(tqdm(test_loader, desc="Evaluating")):
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            log_probs = F.log_softmax(outputs, dim=2)
            
            # Monitor blanks
            predicted_classes = log_probs.argmax(dim=2)
            blank_ratio = (predicted_classes == 0).float().mean().item()
            total_blank_ratio += blank_ratio
            
            # Decode predictions
            predictions = ctc_decode(log_probs)
            
            # Compare with true labels
            for pred, true in zip(predictions, label_strs):
                all_predictions.append(pred)
                all_true_labels.append(true)
                
                if pred == true:
                    correct += 1
                total += 1
                
                # Character-level accuracy
                for i in range(max(len(pred), len(true))):
                    if i < len(pred) and i < len(true):
                        if pred[i] == true[i]:
                            char_correct += 1
                    char_total += 1
    
    accuracy = 100 * correct / total if total > 0 else 0
    char_accuracy = 100 * char_correct / char_total if char_total > 0 else 0
    avg_blank_ratio = total_blank_ratio / len(test_loader)
    
    # Show examples
    if show_examples:
        print("\n" + "="*60)
        print("Sample Predictions:")
        print("="*60)
        for i in range(min(15, len(all_predictions))):
            match = "✓" if all_predictions[i] == all_true_labels[i] else "✗"
            pred_display = all_predictions[i] if all_predictions[i] else "[BLANK]"
            print(f"{match} Pred: '{pred_display:12s}' | True: '{all_true_labels[i]}'")
        print(f"\nCharacter-level accuracy: {char_accuracy:.2f}%")
        print(f"Blank token ratio: {avg_blank_ratio*100:.1f}%")
    
    return accuracy, correct, total, avg_blank_ratio

# ============================================================================
# Learning Rate Scheduler
# ============================================================================
def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    from torch.optim.lr_scheduler import LambdaLR
    return LambdaLR(optimizer, lr_lambda)

# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, required=True)
    parser.add_argument('--test_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0005)  # Lower LR to prevent collapse
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--img_height', type=int, default=32)
    parser.add_argument('--img_width', type=int, default=128)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--blank_penalty', type=float, default=0.1, help='Penalty for blank tokens (0.1-0.3)')
    args = parser.parse_args()
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = CaptchaDataset(args.train_dir, args.img_height, args.img_width)
    test_dataset = CaptchaDataset(args.test_dir, args.img_height, args.img_width)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Enhanced Model with Attention and Residual Connections
    model = EnhancedCRNNModel(
        num_classes=NUM_CLASSES, 
        hidden_size=args.hidden_size, 
        img_height=args.img_height,
        img_width=args.img_width
    )
    model = model.to(device)
    
    print(f"\nModel: Enhanced CRNN with Attention + Residual Connections")
    print(f"Model Output Sequence Length: {model.sequence_length}")
    print(f"Blank penalty: {args.blank_penalty}")
    
    # Loss with blank penalty
    criterion = CTCLossWithBlankPenalty(
        blank=0, 
        reduction='mean', 
        zero_infinity=True,
        blank_penalty=args.blank_penalty
    )
    
    # Optimizer with lower learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Scheduler with longer warmup
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * 5  # 5 epochs warmup
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    # Training loop
    best_accuracy = 0.0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")
        
        train_loss, train_blank_ratio = train_model(
            model, train_loader, criterion, optimizer, scheduler, device, epoch
        )
        print(f"Train Loss: {train_loss:.4f} | Blank Ratio: {train_blank_ratio*100:.1f}%")
        
        accuracy, correct, total, test_blank_ratio = evaluate_model(
            model, test_loader, device, show_examples=(epoch % 5 == 0)
        )
        print(f"Test Accuracy: {accuracy:.2f}% ({correct}/{total}) | Blank Ratio: {test_blank_ratio*100:.1f}%")
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'accuracy': accuracy,
                'args': vars(args)
            }, os.path.join(args.checkpoint_dir, 'best_model.pth'))
            print(f"★ New best model! Accuracy: {accuracy:.2f}%")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy
            }, os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth'))
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Best accuracy: {best_accuracy:.2f}%")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()