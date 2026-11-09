import torch
from torch import nn

class VanillaLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.0):
        super().__init__()
        d = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=d)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x, sector_id=None):
        _, (h, _) = self.lstm(x)
        return self.head(h[-1]).squeeze(-1)
