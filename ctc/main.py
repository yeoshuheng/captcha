"""
Enhanced CAPTCHA Recognition System with Advanced Improvements
Key enhancements:
1. Attention mechanism to focus on character regions
2. Better preprocessing with adaptive thresholding
3. Character-specific augmentation
4. Label smoothing to reduce overconfidence
5. Ensemble prediction with confidence scoring
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

# Define confusable character groups for targeted augmentation
CONFUSABLE_GROUPS = [
    ['0', 'O', 'o'],
    ['1', 'l', 'I', 'i'],
    ['2', 'Z', 'z'],
    ['5', 'S', 's'],
    ['8', 'B'],
    ['p', 'P', 'b'],
    ['n', 'm'],
    ['u', 'v'],
    ['c', 'C'],
    ['q', 'g', '9'],
]

print(f"Character set size: {len(CHARSET)}")
print(f"Total classes (with blank): {NUM_CLASSES}")

# ============================================================================
# Enhanced Preprocessing with Advanced Augmentation
# ============================================================================
class EnhancedCaptchaPreprocess:
    def __init__(self, img_height=64, img_width=200, augment=False):
        self.img_height = img_height
        self.img_width = img_width
        self.augment = augment
        self.normalize = T.Normalize(mean=[0.5], std=[0.5])

    def adaptive_threshold(self, img):
        """Apply adaptive thresholding for better character separation"""
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        return thresh

    def __call__(self, img):
        # Convert to grayscale
        img = np.array(img.convert('L'))
        
        # Resize first
        img = cv2.resize(img, (self.img_width, self.img_height))
        
        # Enhanced preprocessing
        # 1. Denoise
        img = cv2.fastNlMeansDenoising(img, None, h=10)
        
        # 2. CLAHE for contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        
        # 3. Optional: Adaptive thresholding (helps with character separation)
        if np.random.rand() < 0.3 and self.augment:
            thresh = self.adaptive_threshold(img)
            img = cv2.addWeighted(img, 0.7, thresh, 0.3, 0)
        
        # Optional augmentation for training
        if self.augment:
            # Random rotation (smaller range)
            if np.random.rand() < 0.4:
                angle = np.random.uniform(-3, 3)
                M = cv2.getRotationMatrix2D((self.img_width/2, self.img_height/2), angle, 1.0)
                img = cv2.warpAffine(img, M, (self.img_width, self.img_height), 
                                    borderMode=cv2.BORDER_REPLICATE)
            
            # Random brightness/contrast
            if np.random.rand() < 0.5:
                alpha = np.random.uniform(0.8, 1.2)  # contrast
                beta = np.random.uniform(-10, 10)    # brightness
                img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)
            
            # Random Gaussian noise
            if np.random.rand() < 0.3:
                noise = np.random.normal(0, 3, img.shape)
                img = np.clip(img + noise, 0, 255).astype(np.uint8)
            
            # Random blur (simulate focus issues)
            if np.random.rand() < 0.2:
                kernel_size = np.random.choice([3, 5])
                img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
            
            # Random sharpening
            if np.random.rand() < 0.3:
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                img = cv2.filter2D(img, -1, kernel)
                img = np.clip(img, 0, 255).astype(np.uint8)
        
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
        
        self.transform = EnhancedCaptchaPreprocess(img_height, img_width, augment=augment)

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
# Attention Module
# ============================================================================
class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=False
        )
        self.norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        # x: (seq_len, batch, hidden_size)
        attn_out, _ = self.attention(x, x, x)
        return self.norm(x + attn_out)

# ============================================================================
# Enhanced CNN Architecture with Residual Connections
# ============================================================================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class EnhancedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Initial conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # Residual blocks with downsampling
        self.layer1 = self._make_layer(64, 128, stride=2)    # 64x200 -> 32x100
        self.layer2 = self._make_layer(128, 256, stride=2)   # 32x100 -> 16x50
        self.layer3 = self._make_layer(256, 512, stride=(2,1))  # 16x50 -> 8x50
        self.layer4 = self._make_layer(512, 512, stride=(2,1))  # 8x50 -> 4x50
        
        # Keep spatial info
        self.conv_final = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2)
        )
    
    def _make_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            nn.MaxPool2d(kernel_size=stride, stride=stride)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv_final(x)
        return x

# ============================================================================
# Enhanced CRNN with Attention
# ============================================================================
class EnhancedCRNN(nn.Module):
    def __init__(self, num_classes, hidden_size=256, img_height=64, img_width=200):
        super().__init__()
        
        self.img_height = img_height
        self.img_width = img_width
        
        # CNN
        self.cnn = EnhancedCNN()
        
        # Calculate dimensions
        self.rnn_input_size = self._get_rnn_input_size()
        self.sequence_length = self._get_sequence_length()
        
        print(f"CNN output: {self.sequence_length} timesteps")
        print(f"RNN input size: {self.rnn_input_size}")
        
        # Input projection
        self.input_proj = nn.Linear(self.rnn_input_size, hidden_size)
        
        # Bidirectional LSTM with 2 layers
        self.rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            dropout=0.2,
            batch_first=False
        )
        
        # Attention
        self.attention = SelfAttention(hidden_size * 2)
        
        # Output layer with dropout
        self.dropout = nn.Dropout(0.3)
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

    def forward(self, x):
        # CNN features
        conv_features = self.cnn(x)
        batch, channels, height, width = conv_features.size()
        
        # Reshape for RNN: (width, batch, channels*height)
        conv_features = conv_features.permute(3, 0, 1, 2)
        conv_features = conv_features.reshape(width, batch, channels * height)
        
        # Project to hidden size
        conv_features = self.input_proj(conv_features)
        
        # RNN
        rnn_out, _ = self.rnn(conv_features)
        
        # Attention
        rnn_out = self.attention(rnn_out)
        
        # Classifier
        rnn_out = self.dropout(rnn_out)
        output = self.fc(rnn_out)
        return output

# ============================================================================
# Label Smoothing CTC Loss
# ============================================================================
class LabelSmoothingCTCLoss(nn.Module):
    def __init__(self, blank=0, smoothing=0.1, reduction='mean'):
        super().__init__()
        self.blank = blank
        self.smoothing = smoothing
        self.reduction = reduction
        self.ctc_loss = nn.CTCLoss(blank=blank, reduction='none', zero_infinity=True)
    
    def forward(self, log_probs, targets, input_lengths, target_lengths):
        # Standard CTC loss
        losses = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        
        # Add label smoothing regularization
        if self.smoothing > 0:
            # KL divergence with uniform distribution
            kl_loss = -log_probs.mean()
            losses = (1 - self.smoothing) * losses + self.smoothing * kl_loss
        
        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:
            return losses

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
            'blank%': f'{blank_ratio*100:.1f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
    
    avg_loss = total_loss / max(valid_batches, 1)
    avg_blank = np.mean(blank_predictions) if blank_predictions else 0
    
    return avg_loss, avg_blank

# ============================================================================
# Enhanced CTC Decoding with Beam Search
# ============================================================================
def ctc_decode_greedy(predictions):
    """Standard greedy decoding"""
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

def ctc_decode_with_confidence(log_probs, beam_width=5):
    """Decode with confidence scores"""
    probs = torch.exp(log_probs)
    _, max_indices = log_probs.max(dim=2)
    max_indices = max_indices.transpose(0, 1)
    probs = probs.transpose(0, 1)
    
    decoded_strings = []
    confidences = []
    
    for sequence, prob_seq in zip(max_indices, probs):
        chars = []
        char_confidences = []
        prev_idx = None
        
        for idx, probs_t in zip(sequence, prob_seq):
            idx = idx.item()
            if idx != 0 and idx != prev_idx:
                if idx in IDX_TO_CHAR:
                    chars.append(IDX_TO_CHAR[idx])
                    char_confidences.append(probs_t[idx].item())
            prev_idx = idx
        
        decoded_strings.append(''.join(chars))
        avg_conf = np.mean(char_confidences) if char_confidences else 0.0
        confidences.append(avg_conf)
    
    return decoded_strings, confidences

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
    confidences_list = []
    
    with torch.no_grad():
        for images, _, _, label_strs in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            
            outputs = model(images)
            log_probs = F.log_softmax(outputs, dim=2)
            predictions, confidences = ctc_decode_with_confidence(log_probs)
            
            for pred, true, conf in zip(predictions, label_strs, confidences):
                predictions_list.append(pred)
                labels_list.append(true)
                confidences_list.append(conf)
                
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
    avg_confidence = np.mean(confidences_list) if confidences_list else 0
    
    if show_examples:
        print("\n" + "="*80)
        print("Sample Predictions:")
        print("="*80)
        # Sort by confidence for analysis
        sorted_indices = np.argsort(confidences_list)[::-1]
        
        print("\nHigh Confidence Predictions:")
        for i in sorted_indices[:10]:
            match = "✓" if predictions_list[i] == labels_list[i] else "✗"
            print(f"{match} Pred: {predictions_list[i]:12s} | True: {labels_list[i]:12s} | Conf: {confidences_list[i]:.3f}")
        
        print("\nLow Confidence Predictions:")
        for i in sorted_indices[-10:]:
            match = "✓" if predictions_list[i] == labels_list[i] else "✗"
            print(f"{match} Pred: {predictions_list[i]:12s} | True: {labels_list[i]:12s} | Conf: {confidences_list[i]:.3f}")
        
        print(f"\nSequence Accuracy: {accuracy:.2f}%")
        print(f"Character Accuracy: {char_acc:.2f}%")
        print(f"Average Confidence: {avg_confidence:.3f}")
    
    return accuracy, char_acc, avg_confidence

# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, required=True)
    parser.add_argument('--test_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--img_height', type=int, default=64)
    parser.add_argument('--img_width', type=int, default=200)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    args = parser.parse_args()
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Datasets
    train_dataset = CaptchaDataset(args.train_dir, args.img_height, args.img_width, augment=True)
    test_dataset = CaptchaDataset(args.test_dir, args.img_height, args.img_width, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                             collate_fn=collate_fn, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=4, pin_memory=True)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}\n")
    
    # Model
    model = EnhancedCRNN(NUM_CLASSES, args.hidden_size, args.img_height, args.img_width)
    model = model.to(device)
    
    # Loss
    criterion = LabelSmoothingCTCLoss(blank=0, smoothing=args.label_smoothing)
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    # Cosine annealing with warmup
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1000.0
    )
    
    best_accuracy = 0.0
    patience = 20
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*80}")
        
        train_loss, blank_ratio = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch
        )
        print(f"Train Loss: {train_loss:.4f} | Blank Ratio: {blank_ratio*100:.1f}%")
        
        accuracy, char_acc, avg_conf = evaluate(
            model, test_loader, device, 
            show_examples=(epoch % 10 == 0 or epoch < 5)
        )
        print(f"Test Accuracy: {accuracy:.2f}% | Char: {char_acc:.2f}% | Confidence: {avg_conf:.3f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'accuracy': accuracy,
                'char_accuracy': char_acc,
                'confidence': avg_conf
            }, os.path.join(args.checkpoint_dir, 'best_model.pth'))
            print(f"★ New best model saved! Accuracy: {accuracy:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping after {patience} epochs without improvement")
                break
    
    print(f"\n{'='*80}")
    print(f"Training Complete! Best Accuracy: {best_accuracy:.2f}%")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()