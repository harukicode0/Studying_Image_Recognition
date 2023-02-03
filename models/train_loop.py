import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import datetime
import torch

# 訓練ループ
def train_loop(n_epochs, optimizer, model, loss_fn, train, val):
    loss_train_lists = np.array([])
    loss_val_lists = np.array([])
    train_accuracies = np.array([])
    val_accuracies = np.array([])
    
    device = (torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu'))
    # GPU上で訓練するのでモデルを移動。toを使う
    model.to(device=device)
    for epoch in range(1,n_epochs+1):
        loss_train = 0.0 # エポックごとにtrainとvalの損失関数を計算する
        loss_val = 0.0
        
        model.train()
        for imgs, labels in train:
            imgs = imgs.to(device=device) # 画像とラベルをGPU上へ送る
            labels = labels.to(device=device)
            outputs = model(imgs) # 訓練結果
            loss = loss_fn(outputs,labels) # 損失の計算

            optimizer.zero_grad() # 勾配の値を一旦初期化。勾配は累積される。エポック毎に累積されないようにする。
            loss.backward() # バックプロぱゲーションで勾配を計算
            optimizer.step() # 計算した勾配を基にパラメータを調整

            loss_train += loss.item()

        model.eval()
        for val_imgs, val_labels in val:
            with torch.no_grad():
                val_imgs = val_imgs.to(device=device)
                val_labels = val_labels.to(device=device)
                val_outputs = model(val_imgs)
                val_loss = loss_fn(val_outputs, val_labels)
            loss_val += val_loss.item()

        loss_train_lists = np.append(loss_train_lists, loss_train/len(train))
        loss_val_lists = np.append(loss_val_lists, loss_val/len(val))
        
        # train_accuracy, val_accuracy = validate(model, train_loader, val_loader)
        train_accuracy, val_accuracy = validate(model, train, val)
        train_accuracies = np.append(train_accuracies, train_accuracy)
        val_accuracies = np.append(val_accuracies, val_accuracy)
        

        if epoch % 10 == 0 or epoch in [1,2,3]:
            print(f'{datetime.datetime.now()}Epoch:{epoch},Traing_loss:{loss_train/len(train)},val_loss{loss_val/len(val)}')
        
    return model, loss_train_lists, loss_val_lists, train_accuracies, val_accuracies






# 訓練ループ（l２正則化）
def train_loop_l2(n_epochs, optimizer, model, loss_fn, train, val):
    loss_train_lists = np.array([])
    loss_val_lists = np.array([])
    train_accuracies = np.array([])
    val_accuracies = np.array([])
    
    device = (torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu'))
    # device = 'cpu'
    model.to(device=device) # GPU上で訓練するのでモデルを移動。toを使う
    for epoch in range(1,n_epochs+1):
        loss_train = 0.0 # エポックごとにtrainとvalの損失関数を計算する
        loss_val = 0.0

        model.train()
        for imgs, labels in train:
            imgs = imgs.to(device=device) # 画像とラベルをGPU上へ送る
            labels = labels.to(device=device)
            outputs = model(imgs) # 訓練結果
            loss = loss_fn(outputs,labels) # 損失の計算
            
            # L2_regularization
            l2_lambda = 1e-5
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            loss = loss + l2_lambda * l2_norm
            
            # L1_regularization
            # l1_lambda = 0.001
            # l1_norm = sum(p.abs().sum() for p in model.parameters())
            # loss = loss + l1_lambda * l1_norm

            optimizer.zero_grad() # 勾配の値を一旦初期化。勾配は累積される。エポック毎に累積されないようにする。
            loss.backward() # バックプロぱゲーションで勾配を計算
            optimizer.step() # 計算した勾配を基にパラメータを調整

            loss_train += loss.item()

        model.eval()
        for val_imgs, val_labels in val:
            with torch.no_grad():
                val_imgs = val_imgs.to(device=device)
                val_labels = val_labels.to(device=device)
                val_outputs = model(val_imgs)
                val_loss = loss_fn(val_outputs, val_labels)
            loss_val += val_loss.item()

        loss_train_lists = np.append(loss_train_lists, loss_train/len(train))
        loss_val_lists = np.append(loss_val_lists, loss_val/len(val))
        
        train_accuracy, val_accuracy = validate(model, train, val)
        # train_accuracy, val_accuracy = validate(model, train_loader, val_loader)
        train_accuracies = np.append(train_accuracies, train_accuracy)
        val_accuracies = np.append(val_accuracies, val_accuracy)
        

        if epoch % 10 == 0 or epoch in [1,2,3]:
            print(f'{datetime.datetime.now()}Epoch:{epoch},Traing_loss:{loss_train/len(train)},val_loss{loss_val/len(val)}')
        
    return model, loss_train_lists, loss_val_lists, train_accuracies, val_accuracies








# fine_tuning
def fine_tuning(n_epochs, optimizer, model, loss_fn, train, val):
    loss_train_lists = np.array([])
    loss_val_lists = np.array([])
    train_accuracies = np.array([])
    val_accuracies = np.array([])
    # best_model_wts = copy.deepcopy(model.state_dict())
    
    device = (torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu'))
    # device = 'cpu'
    model.to(device=device) # GPU上で訓練するのでモデルを移動。toを使う
    for epoch in range(1,n_epochs+1):
        loss_train = 0.0 # エポックごとにtrainとvalの損失関数を計算する
        loss_val = 0.0

        model.train()
        for imgs, labels in train:
            imgs = imgs.to(device=device) # 画像とラベルをGPU上へ送る
            labels = labels.to(device=device)
            outputs = model(imgs) # 訓練結果
            loss = loss_fn(outputs,labels) # 損失の計算

            optimizer.zero_grad() # 勾配の値を一旦初期化。勾配は累積される。エポック毎に累積されないようにする。
            loss.backward() # バックプロぱゲーションで勾配を計算
            optimizer.step() # 計算した勾配を基にパラメータを調整

            loss_train += loss.item()

        model.eval()
        for val_imgs, val_labels in val:
            with torch.no_grad():
                val_imgs = val_imgs.to(device=device)
                val_labels = val_labels.to(device=device)
                val_outputs = model(val_imgs)
                val_loss = loss_fn(val_outputs, val_labels)
            loss_val += val_loss.item()

        loss_train_lists = np.append(loss_train_lists, loss_train/len(train))
        loss_val_lists = np.append(loss_val_lists, loss_val/len(val))
        
        train_accuracy, val_accuracy = validate(model, train, val)
        # train_accuracy, val_accuracy = validate(model, train_loader, val_loader)
        train_accuracies = np.append(train_accuracies, train_accuracy)
        val_accuracies = np.append(val_accuracies, val_accuracy)
        

        print(f'{datetime.datetime.now()}Epoch:{epoch},Traing_loss:{loss_train/len(train)},val_loss{loss_val/len(val)}')
        
    return model, loss_train_lists, loss_val_lists, train_accuracies, val_accuracies








# 正解率の確認
def validate(model, train_loader, val_loader):
    use_device = 'mps'  # モデルをGPU上に持ってくる
    for name, loader in [("train", train_loader), ("val", val_loader)]: # 最初にTrain、printで結果を表示後、valを計算して結果を表示する流れ
        correct = 0
        total = 0
        with torch.no_grad(): # 勾配を計算させないのための記述
            for imgs, labels in loader:
                model.to(device=use_device)
                imgs = imgs.to(device=use_device)
                labels = labels.to(device = use_device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.shape[0]
                correct += int((predicted == labels).sum())
                # 最初にtrain、後にvalの正解率を出す。
                if name == 'train':
                    train_accuracy = correct / total
                else:
                    val_accuracy = correct / total
    return train_accuracy, val_accuracy