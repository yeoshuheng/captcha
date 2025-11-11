import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from commons import ctc_decode

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
        
        outputs = model(images)
        log_probs = F.log_softmax(outputs, dim=2)
        
        with torch.no_grad():
            pred_classes = log_probs.argmax(dim=2)
            blank_ratio = (pred_classes == 0).float().mean().item()
            blank_predictions.append(blank_ratio)
        
        seq_len = log_probs.size(0)
        batch_size = log_probs.size(1)
        input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)
        
        if seq_len < label_lengths.max().item():
            continue
        
        loss = criterion(log_probs, labels, input_lengths, label_lengths)
       
        
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

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    char_correct = 0
    char_total = 0
    
    predictions_list = []
    labels_list = []
    confidences_list = []
    
    with torch.no_grad():
        for images, _, _, label_strs in tqdm(test_loader, desc="evaluation..."):
            images = images.to(device)
            
            outputs = model(images)
            log_probs = F.log_softmax(outputs, dim=2)
            predictions, confidences = ctc_decode(log_probs)
            
            for pred, true, conf in zip(predictions, label_strs, confidences):
                predictions_list.append(pred)
                labels_list.append(true)
                confidences_list.append(conf)
                
                if pred == true:
                    correct += 1
                total += 1
                
                min_len = min(len(pred), len(true))
                for i in range(min_len):
                    if pred[i] == true[i]:
                        char_correct += 1
                char_total += max(len(pred), len(true))
    
    accuracy = 100 * correct / total if total > 0 else 0
    char_acc = 100 * char_correct / char_total if char_total > 0 else 0
    avg_confidence = np.mean(confidences_list) if confidences_list else 0
        
    return accuracy, char_acc, avg_confidence