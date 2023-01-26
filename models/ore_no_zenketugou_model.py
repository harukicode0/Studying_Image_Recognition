import torch.nn.functional as F
import torch
import torch.nn as nn

# モデルの作成、シンプルな全結合
class Zenketugou(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3072,512)
        self.fc2 = nn.Linear(512,100)
    
    def forward(self, x):
        out = x.view(-1,32*32*3)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out