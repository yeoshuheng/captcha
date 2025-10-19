import torch
from trainer import Trainer
from dataset import CHARSET
from typing_extensions import override
import torch.nn as nn

def ctc_preprocess(batch):
    images, labels, label_len = zip(*batch)
    
    images = torch.stack(images)
    
    labels_concat = torch.cat(labels)

    input_lengths = torch.full((len(batch),), 128, dtype=torch.long)
    target_lengths = torch.tensor(label_len, dtype=torch.long)
    
    return images, labels_concat, input_lengths, target_lengths


def ctc_criterion(log_probs, targets, input_lengths, target_lengths):
    return nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)(
        log_probs, targets, input_lengths, target_lengths
    )

def decode(tensor: torch.Tensor):
    selection = tensor.argmax(dim = 2)
    decoded_captchas = []

    for sel in selection:
        curr_captcha = []
        prev = 0
        for selected_idx in sel:
            idx = selected_idx.item()
            if idx != prev and idx != 0:
                curr_captcha.append(CHARSET[idx - 1])
            prev = idx
        decoded_captchas.append("".join(curr_captcha))
    
    return decoded_captchas

class CTCTrainer(Trainer):

    @override
    def _process_results(self, results: torch.Tensor) -> torch.Tensor:
        return results.log_softmax(2).permute(1, 0, 2) 
    
