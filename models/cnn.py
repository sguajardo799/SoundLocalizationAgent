import torch
from torch import nn

class CNN_min(nn.Module):
    def __init__(self, W, H):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2),
        
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2),
        
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2)
        )

        self.flatten = nn.Flatten()

        self.dense1 = nn.Sequential(
            nn.Linear(in_features=self._calculate_output_size(W, H), out_features = 64),
            nn.ReLU(),

            nn.Linear(in_features=64, out_features = 32),
            nn.ReLU(),
            )
        
        self.out = nn.Linear(in_features=32, out_features=5)

    def _calculate_output_size(self, W, H): #Ws = (We - k_size + 2*pad)/stride #pad=0, stride=1
        # Conv1
        W = (W - 3 + 2*0)//1 + 1
        H = (H - 3 + 2*0)//1 + 1
        W, H = W // 2, H // 2

        # Conv2
        W = W - 3 + 1
        H = H - 3 + 1
        W, H = W // 2, H // 2

        # Conv3
        W = W - 3 + 1
        H = H - 3 + 1
        W, H = W // 2, H // 2

        return int(W * H * 256)

    def forward(self, x:torch.Tensor):
        x = self.convs(x)

        x = self.flatten(x)

        x = self.dense1(x)
        x = self.out(x)       
        return x
