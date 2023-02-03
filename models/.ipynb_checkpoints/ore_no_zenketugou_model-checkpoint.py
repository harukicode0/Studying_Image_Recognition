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
        self.fc1 = nn.Linear(32*32*3,4096)
        self.fc2 = nn.Linear(4096,4096)
        self.fc3 = nn.Linear(4096,2048)
        self.fc4 = nn.Linear(2048,100)
    
    def forward(self, x):
        out = x.view(-1,32*32*3)
        out = torch.relu(self.fc1(out))
        out = torch.relu(self.fc2(out))
        out = torch.relu(self.fc3(out))
        out = self.fc4(out)
        return out

# ドロップアウト付き全結合の実装
class DropoutZenketugou(nn.Module):
    def __init__(self, per):
        super().__init__()
        self.per = per
        self.fc1 = nn.Linear(3072,4096)
        self.fc1_dropout = nn.Dropout(p=per)
        self.fc2 = nn.Linear(4096,4096)
        self.fc2_dropout = nn.Dropout(p=per)
        self.fc3 = nn.Linear(4096,2048)
        self.fc3_dropout = nn.Dropout(p=per)
        self.fc4 = nn.Linear(2048,100)
    
    def forward(self, x):
        out = x.view(-1,32*32*3)
        out = torch.relu(self.fc1(out))
        out = self.fc1_dropout(out)
        out = torch.relu(self.fc2(out))
        out = self.fc2_dropout(out)
        out = torch.relu(self.fc3(out))
        out = self.fc3_dropout(out)
        out = self.fc4(out)
        return out

# CNN
class CNN(nn.Module):
    def __init__(self, n_chans1):
        super().__init__()
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(n_chans1, n_chans1 // 2, kernel_size=3,padding=1)
        self.fc1 = nn.Linear(8 * 8 * n_chans1 // 2, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 100)
        
    def forward(self, x):
        out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
        out = F.max_pool2d(torch.tanh(self.conv2(out)), 2)
        out = out.view(-1, 8 * 8 * self.n_chans1 // 2)
        out = torch.tanh(self.fc1(out))
        out = torch.tanh(self.fc2(out))
        out = torch.tanh(self.fc3(out))
        return out

class CNN_ReLU(nn.Module):
    def __init__(self, n_chans1):
        super().__init__()
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(n_chans1, n_chans1 // 2, kernel_size=3,padding=1)
        self.fc1 = nn.Linear(8 * 8 * n_chans1 // 2, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 100)
        
    def forward(self, x):
        out = F.max_pool2d(torch.relu(self.conv1(x)), 2)
        out = F.max_pool2d(torch.relu(self.conv2(out)), 2)
        out = out.view(-1, 8 * 8 * self.n_chans1 // 2)
        out = torch.relu(self.fc1(out))
        out = torch.relu(self.fc2(out))
        out = torch.relu(self.fc3(out))
        return out

# BatchNorm
class CNNBatchNorm(nn.Module):
    def __init__(self, n_chans1):
        super().__init__()
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=3, padding=1)
        self.conv1_batchnorm = nn.BatchNorm2d(num_features=n_chans1)
        self.conv2 = nn.Conv2d(n_chans1, n_chans1 // 2, kernel_size=3,padding=1)
        self.conv2_batchnorm = nn.BatchNorm2d(num_features=n_chans1//2)        
        self.fc1 = nn.Linear(8 * 8 * n_chans1 // 2, 4096)
        self.fc2 = nn.Linear(4096,2048)
        self.fc3 = nn.Linear(2048, 100)
        
    def forward(self, x):
        # バッチノーマリゼーションは活性化関数に入力する前に行う
        out = self.conv1_batchnorm(self.conv1(x))
        out = F.max_pool2d(torch.tanh(out), 2)
        out = self.conv2_batchnorm(self.conv2(out))
        out = F.max_pool2d(torch.tanh(out), 2)
        out = out.view(-1, 8 * 8 * self.n_chans1 // 2)
        out = torch.tanh(self.fc1(out))
        out = torch.tanh(self.fc2(out))
        out = torch.tanh(self.fc3(out))
        return out

class CNNBatchNorm_ReLU(nn.Module):
    def __init__(self, n_chans1):
        super().__init__()
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=3, padding=1)
        self.conv1_batchnorm = nn.BatchNorm2d(num_features=n_chans1)
        self.conv2 = nn.Conv2d(n_chans1, n_chans1 // 2, kernel_size=3,padding=1)
        self.conv2_batchnorm = nn.BatchNorm2d(num_features=n_chans1//2)        
        self.fc1 = nn.Linear(8 * 8 * n_chans1 // 2, 4096)
        # self.fc1_batchnorm = nn.BatchNorm1d(num_features=256)
        self.fc2 = nn.Linear(4096,2048)
        # self.fc2_batchnorm = nn.BatchNorm1d(num_features=128)
        self.fc3 = nn.Linear(2048, 100)
        
    def forward(self, x):
        # バッチノーマリゼーションは活性化関数に入力する前に行う
        out = self.conv1_batchnorm(self.conv1(x))
        out = F.max_pool2d(torch.relu(out), 2)
        out = self.conv2_batchnorm(self.conv2(out))
        out = F.max_pool2d(torch.relu(out), 2)
        out = out.view(-1, 8 * 8 * self.n_chans1 // 2)
        out = torch.relu(self.fc1(out))
        out = torch.relu(self.fc2(out))
        out = torch.relu(self.fc3(out))
        return out
    
class CNNBatchNorm_ReLU_dropout(nn.Module):
    def __init__(self, n_chans1, per):
        super().__init__()
        self.n_chans1 = n_chans1
        self.per = per
        self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=3, padding=1)
        self.conv1_batchnorm = nn.BatchNorm2d(num_features=n_chans1)
        self.conv2 = nn.Conv2d(n_chans1, n_chans1 // 2, kernel_size=3,padding=1)
        self.conv2_batchnorm = nn.BatchNorm2d(num_features=n_chans1//2)        
        self.fc1 = nn.Linear(8 * 8 * n_chans1 // 2, 4096)
        self.fc1_dropout = nn.Dropout(p=per)
        self.fc2 = nn.Linear(4096,2048)
        self.fc2_dropout = nn.Dropout(p=per)
        self.fc3 = nn.Linear(2048, 100)
        
    def forward(self, x):
        # バッチノーマリゼーションは活性化関数に入力する前に行う
        out = self.conv1_batchnorm(self.conv1(x))
        out = F.max_pool2d(torch.relu(out), 2)
        out = self.conv2_batchnorm(self.conv2(out))
        out = F.max_pool2d(torch.relu(out), 2)
        out = out.view(-1, 8 * 8 * self.n_chans1 // 2)
        out = torch.relu(self.fc1(out))
        out = self.fc1_dropout(out)
        out = torch.relu(self.fc2(out))
        out = self.fc2_dropout(out)
        out = torch.relu(self.fc3(out))
        return out
    
class CNNBatchNorm_sigmoid(nn.Module):
    def __init__(self, n_chans1):
        super().__init__()
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=3, padding=1)
        self.conv1_batchnorm = nn.BatchNorm2d(num_features=n_chans1)
        self.conv2 = nn.Conv2d(n_chans1, n_chans1 // 2, kernel_size=3,padding=1)
        self.conv2_batchnorm = nn.BatchNorm2d(num_features=n_chans1//2)        
        self.fc1 = nn.Linear(8 * 8 * n_chans1 // 2, 4096)
        # self.fc1_batchnorm = nn.BatchNorm1d(num_features=256)
        self.fc2 = nn.Linear(4096,2048)
        # self.fc2_batchnorm = nn.BatchNorm1d(num_features=128)
        self.fc3 = nn.Linear(2048, 100)
        
    def forward(self, x):
        # バッチノーマリゼーションは活性化関数に入力する前に行う
        out = self.conv1_batchnorm(self.conv1(x))
        out = F.max_pool2d(torch.sigmoid(out), 2)
        out = self.conv2_batchnorm(self.conv2(out))
        out = F.max_pool2d(torch.sigmoid(out), 2)
        out = out.view(-1, 8 * 8 * self.n_chans1 // 2)
        out = torch.sigmoid(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        out = torch.sigmoid(self.fc3(out))
        return out

# resnetの作成
class ResBlock(nn.Module):
    def __init__(self, n_chans):
        super(ResBlock, self).__init__()
        self.conv = nn.Conv2d(n_chans,n_chans,kernel_size=3,padding=1,bias=False)
        self.batch_norm = nn.BatchNorm2d(num_features=n_chans)
        # torch.nn.init.kaiming_normal_(self.conv.weight,nonlinearity='tanh')
        torch.nn.init.kaiming_normal_(self.conv.weight,nonlinearity='relu')
        torch.nn.init.constant_(self.batch_norm.weight,0.5)
        torch.nn.init.zeros_(self.batch_norm.bias)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.batch_norm(out)
        # out = nn.tanh(out)
        out = torch.relu(out)
        return out + x

class ResNetReLU(nn.Module):
    def __init__(self, n_chans1, n_blocks):
        super().__init__()
        self.n_chans1 = n_chans1
        self.n_blocks = n_blocks
        self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=3, padding=1)
        self.resblocks = nn.Sequential(*(n_blocks*[ResBlock(n_chans=n_chans1)]))
        self.fc1 = nn.Linear(8*8*n_chans1,2048)
        self.fc2 = nn.Linear(2048,2048)
        self.fc3 = nn.Linear(2048,100)
    
    def forward(self, x):
        out = F.max_pool2d(torch.relu(self.conv1(x)), 2)
        out = self.resblocks(out)
        out = F.max_pool2d(out, 2)
        out = out.view(-1,8*8*self.n_chans1)
        out = torch.relu(self.fc1(out))
        out = torch.relu(self.fc2(out))
        out = torch.relu(self.fc3(out))
        return out

class ResNetTanh(nn.Module):
    def __init__(self, n_chans1, n_blocks):
        super().__init__()
        self.n_chans1 = n_chans1
        self.n_blocks = n_blocks
        self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=3, padding=1)
        self.resblocks = nn.Sequential(*(n_blocks*[ResBlock(n_chans=n_chans1)]))
        self.fc1 = nn.Linear(8*8*n_chans1,1024)
        self.fc2 = nn.Linear(1024,1024)
        self.fc3 = nn.Linear(1024,100)
    
    def forward(self, x):
        out = F.max_pool2d(torch.relu(self.conv1(x)), 2)
        out = self.resblocks(out)
        out = F.max_pool2d(out, 2)
        out = out.view(-1,8*8*self.n_chans1)
        out = torch.tanh(self.fc1(out))
        out = torch.tanh(self.fc2(out))
        out = torch.tanh(self.fc3(out))
        return out
