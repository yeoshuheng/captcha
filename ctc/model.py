
import torch
import torch.nn as nn

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
        attn_out, _ = self.attention(x, x, x)
        return self.norm(x + attn_out)

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
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        self.layer1 = self._make_layer(64, 128, stride=2)    
        self.layer2 = self._make_layer(128, 256, stride=2)   
        self.layer3 = self._make_layer(256, 512, stride=(2,1)) 
        self.layer4 = self._make_layer(512, 512, stride=(2,1)) 
        
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


class EnhancedCRNN(nn.Module):
    def __init__(self, num_classes, hidden_size=256, img_height=64, img_width=200):
        super().__init__()
        
        self.img_height = img_height
        self.img_width = img_width
        
        self.cnn = EnhancedCNN()
        
        self.rnn_input_size = self._get_rnn_input_size()
        self.sequence_length = self._get_sequence_length()
        
        self.input_proj = nn.Linear(self.rnn_input_size, hidden_size)
        
        self.rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            dropout=0.2,
            batch_first=False
        )
        
        self.attention = SelfAttention(hidden_size * 2)
        
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    
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


    def forward(self, x):
        conv_features = self.cnn(x)
        batch, channels, height, width = conv_features.size()
        conv_features = conv_features.permute(3, 0, 1, 2)
        conv_features = conv_features.reshape(width, batch, channels * height)
        conv_features = self.input_proj(conv_features)
        rnn_out, _ = self.rnn(conv_features)
        rnn_out = self.attention(rnn_out)
        rnn_out = self.dropout(rnn_out)
        output = self.fc(rnn_out)
        return output