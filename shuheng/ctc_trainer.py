import torch
from trainer import Trainer
from dataset import CHARSET
from typing_extensions import override
import torch.nn as nn

def ctc_preprocess(batch):
    images, labels, label_len = zip(*batch)
    
    images = torch.stack(images)
    
    labels_concat = torch.cat(labels)
    target_lengths = torch.tensor(label_len, dtype=torch.long)
    
    return images, labels_concat, target_lengths


def ctc_criterion(log_probs, targets, input_lengths, target_lengths):
    return nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)(
        log_probs, targets, input_lengths, target_lengths
    )

def decode(tensor: torch.Tensor):
    selection = tensor.argmax(dim=2)
    
    decoded_captchas = []
    
    for batch_idx in range(selection.shape[1]):
        seq = selection[:, batch_idx] 
        
        output = []
        prev = -1
        
        for idx in seq:
            idx = idx.item()
            if idx != 0 and idx != prev:  
                output.append(CHARSET[idx - 1])
            prev = idx
        
        decoded_captchas.append("".join(output))
    
    return decoded_captchas

class CTCTrainer(Trainer):

    @override
    def _process_results(self, results: torch.Tensor) -> torch.Tensor:
        return results.log_softmax(2).permute(1, 0, 2) 
    
