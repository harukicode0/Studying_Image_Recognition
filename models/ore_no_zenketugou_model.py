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
    
# 全結合ビック
class Zenketugou_big(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3072,4096)
        self.fc2 = nn.Linear(4096,2048)
        self.fc3 = nn.Linear(2048,1024)
        self.fc4 = nn.Linear(1024,100)
    
    def forward(self, x):
        out = x.view(-1,32*32*3)
        out = torch.tanh(self.fc1(out))
        out = torch.tanh(self.fc2(out))
        out = torch.tanh(self.fc3(out))
        out = self.fc4(out)
        return out

# ドロップアウト付き全結合の実装
class DropoutZenketugou(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3072,4096)
        self.fc1_dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(4096,2048)
        self.fc2_dropout = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(2048,1024)
        self.fc3_dropout = nn.Dropout(p=0.3)
        self.fc4 = nn.Linear(1024,100)
    
    def forward(self, x):
        out = x.view(-1,32*32*3)
        out = torch.tanh(self.fc1(out))
        out = self.fc1_dropout(out)
        out = torch.tanh(self.fc2(out))
        out = self.fc2_dropout(out)
        out = torch.tanh(self.fc3(out))
        out = self.fc3_dropout(out)
        out = self.fc4(out)
        return out

# CNN
class CNN(nn.Module):
    def __init__(self, n_chans1=32):
        super().__init__()
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(n_chans1, n_chans1 // 2, kernel_size=3,
                               padding=1)
        self.fc1 = nn.Linear(8 * 8 * n_chans1 // 2, 32)
        self.fc2 = nn.Linear(32, 100)
        
    def forward(self, x):
        out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
        out = F.max_pool2d(torch.tanh(self.conv2(out)), 2)
        out = out.view(-1, 8 * 8 * self.n_chans1 // 2)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out


