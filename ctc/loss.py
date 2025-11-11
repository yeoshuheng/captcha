import torch.nn as nn

class SmoothCTC(nn.Module):
    def __init__(self, blank=0, smoothing=0.1, reduction='mean'):
        super().__init__()
        self.blank = blank
        self.smoothing = smoothing
        self.reduction = reduction
        self.ctc_loss = nn.CTCLoss(blank=blank, reduction='none', zero_infinity=True)
    
    def forward(self, log_probs, targets, input_lengths, target_lengths):
        losses = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        if self.smoothing > 0:
            kl_loss = -log_probs.mean()
            losses = (1 - self.smoothing) * losses + self.smoothing * kl_loss
        return losses.mean()
