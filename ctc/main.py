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

CHARSET = string.digits + string.ascii_lowercase + string.ascii_uppercase
CHAR_TO_IDX = {char: idx + 1 for idx, char in enumerate(CHARSET)}  # 0 reserved for CTC blank
IDX_TO_CHAR = {idx + 1: char for idx, char in enumerate(CHARSET)}
NUM_CLASSES = len(CHARSET) + 1  # +1 for CTC blank token


# ============================================================================
# Dataset
# ============================================================================

class CaptchaDataset(Dataset):
    """Dataset for CAPTCHA images with labels extracted from filenames."""
    
    def __init__(self, image_dir, img_height=32, img_width=128):
        self.image_paths = []
        self.labels = []
        
        # Load all PNG images and extract labels from filenames
        for path in Path(image_dir).glob("*.png"):
            self.image_paths.append(str(path))
            # Assuming filename format: "label-xxxxx.png"
            label = path.stem.split('-')[0]
            self.labels.append(label)
        
        self.transform = T.Compose([
            T.Resize((img_height, img_width)),
            T.Grayscale(num_output_channels=1),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img = Image.open(self.image_paths[idx]).convert('RGB')
        img = self.transform(img)
        
        # Convert label to indices
        label = self.labels[idx]
        label_indices = torch.tensor([CHAR_TO_IDX[c] for c in label], dtype=torch.long)
        
        return img, label_indices


def collate_fn(batch):
    """Collate function for DataLoader to handle variable-length labels."""
    images, labels = zip(*batch)
    
    # Stack images
    images = torch.stack(images, dim=0)
    
    # Concatenate all labels
    labels_concat = torch.cat(labels)
    
    # Get label lengths
    label_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
    
    return images, labels_concat, label_lengths


# ============================================================================
# Model Architecture
# ============================================================================

class CRNNModel(nn.Module):
    """
    CRNN architecture for CAPTCHA recognition:
    - CNN for feature extraction
    - RNN for sequence modeling
    - Fully connected layer for classification
    """
    
    def __init__(self, num_classes, hidden_size=256):
        super().__init__()
        
        # CNN layers for feature extraction
        self.cnn = nn.Sequential(
            # Layer 1: 1 -> 64
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # H/2, W/2
            
            # Layer 2: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # H/4, W/4
            
            # Layer 3: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Layer 4: 256 -> 256
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),  # H/8, W/4
            
            # Layer 5: 256 -> 512
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Layer 6: 512 -> 512
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),  # H/16, W/4
        )
        
        # Bidirectional LSTM
        self.rnn = nn.LSTM(
            input_size=512 * 2,  # CNN output channels * height
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            dropout=0.3,
            batch_first=False  # Input shape: (seq_len, batch, features)
        )
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional
    
    def forward(self, x):
        # CNN feature extraction
        # Input: (batch, 1, 32, 128)
        conv_features = self.cnn(x)
        # Output: (batch, 512, 2, 32)
        
        batch, channels, height, width = conv_features.size()
        
        # Reshape for RNN: (batch, channels, height, width) -> (width, batch, channels*height)
        conv_features = conv_features.permute(3, 0, 1, 2)  # (width, batch, channels, height)
        conv_features = conv_features.reshape(width, batch, channels * height)
        
        # RNN sequence modeling
        rnn_out, _ = self.rnn(conv_features)
        # Output: (seq_len, batch, hidden_size*2)
        
        # Fully connected layer
        output = self.fc(rnn_out)
        # Output: (seq_len, batch, num_classes)
        
        return output


# ============================================================================
# Training
# ============================================================================

def train_model(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for images, labels, label_lengths in pbar:
        images = images.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)
        
        # Forward pass
        outputs = model(images)  # (seq_len, batch, num_classes)
        
        # Apply log_softmax for CTC
        log_probs = F.log_softmax(outputs, dim=2)
        
        # Input lengths (sequence length from model output)
        seq_len = log_probs.size(0)
        input_lengths = torch.full((log_probs.size(1),), seq_len, dtype=torch.long, device=device)
        
        # Compute CTC loss
        loss = criterion(log_probs, labels, input_lengths, label_lengths)
        
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

def evaluate_model(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels_concat, label_lengths in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            log_probs = F.log_softmax(outputs, dim=2)
            
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
            
            # Calculate accuracy
            for pred, true in zip(predictions, true_labels):
                if pred == true:
                    correct += 1
                total += 1
    
    accuracy = 100 * correct / total
    return accuracy, correct, total


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train CAPTCHA recognition model with CTC')
    parser.add_argument('--train_dir', type=str, required=True, help='Training data directory')
    parser.add_argument('--test_dir', type=str, required=True, help='Test data directory')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Training loop
    best_accuracy = 0.0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss = train_model(model, train_loader, criterion, optimizer, device, epoch)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Evaluate
        accuracy, correct, total = evaluate_model(model, test_loader, device)
        print(f"Test Accuracy: {accuracy:.2f}% ({correct}/{total})")
        
        # Learning rate scheduling
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