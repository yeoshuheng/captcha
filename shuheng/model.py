import torch
import torch.nn as nn
from typing_extensions import override

class CaptchaModel(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.n_classes = n_classes

        self.conv_layers: nn.Module = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2, dilation=2), 
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=4, dilation=4),  
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1), 
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=(2,1), padding=(1,1)), 
            nn.ReLU()
        )

        self.gru: nn.Module = nn.GRU(512 * 2, 128, bidirectional=True, num_layers=1, batch_first=True)
        self.fc = nn.Linear(256, self.n_classes)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        B, C, H, W = x.size()

        x = x.permute(0, 3, 1, 2)
        x = x.reshape(B, W, C * H)

        x, _ = self.gru(x)

        return self.fc(x)
