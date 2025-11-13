import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import time
import editdistance

from commons import ctc_decode

def train(model, train_loader, criterion, optimizer, scheduler, device, epoch):
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
    
    total_edit_distance = 0
    total_length = 0
    
    substitutions = 0
    insertions = 0
    deletions = 0
    
    length_bins = defaultdict(lambda: {"correct": 0, "total": 0})
    
    predictions_list = []
    labels_list = []
    confidences_list = []
    
    start_time = time.time()
    
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
                
                ed = editdistance.eval(pred, true)
                total_edit_distance += ed
                total_length += len(true)
                 
                dp = np.zeros((len(true)+1, len(pred)+1), dtype=int)
                for i in range(len(true)+1):
                    dp[i][0] = i
                for j in range(len(pred)+1):
                    dp[0][j] = j
                for i in range(1, len(true)+1):
                    for j in range(1, len(pred)+1):
                        cost = 0 if true[i-1] == pred[j-1] else 1
                        dp[i][j] = min(
                            dp[i-1][j] + 1,
                            dp[i][j-1] + 1,
                            dp[i-1][j-1] + cost
                        )
                i, j = len(true), len(pred)
                while i > 0 or j > 0:
                    if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] and true[i-1] == pred[j-1]:
                        i, j = i-1, j-1
                    elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
                        substitutions += 1
                        i, j = i-1, j-1
                    elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
                        deletions += 1
                        i -= 1
                    elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
                        insertions += 1
                        j -= 1
                
                length_bins[len(true)]["total"] += 1
                if pred == true:
                    length_bins[len(true)]["correct"] += 1
    
    inference_time = time.time() - start_time
    avg_inference_time = inference_time / total if total > 0 else 0
    
    exact_match_acc = 100 * correct / total if total > 0 else 0
    char_acc = 100 * char_correct / char_total if char_total > 0 else 0
    norm_edit_distance = total_edit_distance / total_length if total_length > 0 else 0
    
    total_edits = substitutions + insertions + deletions
    sub_rate = 100 * substitutions / total_edits if total_edits > 0 else 0
    ins_rate = 100 * insertions / total_edits if total_edits > 0 else 0
    del_rate = 100 * deletions / total_edits if total_edits > 0 else 0
    
    length_acc_values = [
        100 * v["correct"] / v["total"] for v in length_bins.values() if v["total"] > 0
    ]
    length_acc = np.mean(length_acc_values) if length_acc_values else 0
    
    return {
        "exact_match_acc": exact_match_acc,
        "length_acc": length_acc,
        "char_acc": char_acc,
        "normalized_edit_distance": norm_edit_distance,
        "sub_rate": sub_rate,
        "ins_rate": ins_rate,
        "del_rate": del_rate,
        "avg_inference_time": avg_inference_time,
        "total_samples": total
    }
