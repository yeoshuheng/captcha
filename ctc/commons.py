import string
import torch
import numpy as np

CHARSET = string.digits + string.ascii_lowercase + string.ascii_uppercase
CHAR_TO_IDX = {char: idx + 1 for idx, char in enumerate(CHARSET)}
IDX_TO_CHAR = {idx + 1: char for idx, char in enumerate(CHARSET)}
NUM_CLASSES = len(CHARSET) + 1  # +1 for CTC blank token

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

def ctc_decode(log_probs):
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