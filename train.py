from utils import load_data
from MLP import mlp
import torch
import torchvision
from torch.nn import Module
import cifar10 as dataset
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd
from logentropy import LogEntropyLoss



epochs = 128

lr = 0.1
label = "l1loss_4classeswithResnet"



def train(train_loader, model, optimizer, train_criterion, use_cuda, epoch):
    sum_loss = 0.0
    for data in train_loader:
        img, label = data
        if use_cuda:
            img = Variable(img).cuda()
            label = Variable(label).cuda()
        else:
            img = Variable(img)
            label = Variable(label)
        out = model(img)
        loss = train_criterion(out, label)
        print_loss = loss.data.item()
        sum_loss += print_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Epoch', epoch, 'Train Loss:', sum_loss)
    return sum_loss

def watch(watch_loader, model, watch_criterion, use_cuda, epoch):
    sum_loss = 0.0
    with torch.no_grad():
        for data in watch_loader:
            val_inputs, label = data
            if use_cuda:
                val_inputs = Variable(val_inputs).cuda()
                label = Variable(label).cuda()
            val_outputs = model(val_inputs)
            loss = watch_criterion(val_outputs, label)
            print_loss = loss.data.item()
            sum_loss += print_loss
        print('Epoch:', epoch, 'True Loss:', sum_loss)
    return sum_loss

def valid(valid_loader, model, use_cuda, epoch):
    total_correct = 0.0
    total_num = 0.0
    with torch.no_grad():
        for data in valid_loader:
            val_inputs, label = data
            if use_cuda:
                val_inputs = Variable(val_inputs).cuda()
                label = Variable(label).cuda()
            val_outputs = model(val_inputs)
            pred = val_outputs.argmax(dim=1)
            total_correct += torch.eq(pred,label).float().sum().item() #分别为是否相等，scalar tensor转换为float，求和，拿出值
            total_num += label.size(0)
        acc = total_correct/total_num
        print('Epoch:', epoch, 'Val Acc:', acc)
        return acc   

def plot_curve(epochs, train_losses, true_losses, valid_accs, label):
    epoch_num = epochs
    x1 = range(0, epoch_num)
    x2 = range(0, epoch_num)
    x3 = range(0, epoch_num)
    plt.subplot(3, 1, 1)
    plt.plot(x1, valid_accs, 'o-')
    plt.ylabel('Val Acc')
    plt.subplot(3, 1, 2)
    plt.plot(x2, train_losses, '.-')
    plt.ylabel('Train Loss')
    plt.subplot(3, 1, 3)
    plt.plot(x3, true_losses, '.-')
    plt.xlabel('epochs')
    plt.ylabel('True Loss')
    plt.savefig('./logs/'+label +".png")

def log(tag, train_loss, true_loss, val_acc, e_losses, c_losses):
    data = {'train loss':train_loss, "true loss":true_loss, "val acc":val_acc, 'loss term 1':e_losses, 'loss term 2':c_losses}
    df = pd.DataFrame(data)
    df.to_csv('./logs/'+label+tag+'.csv')


def main():
    use_cuda = torch.cuda.is_available()
    print("cuda:",use_cuda)
    #model = mlp()
    model = torchvision.models.resnet18(pretrained=True)
    fc_in = model.fc.in_features  # 获取全连接层的输入特征维度
    model.fc =torch.nn.Linear(fc_in, 10)
    if use_cuda:
        model = model.cuda()
    train_loader, watch_loader, valid_loader = load_data()
    train_criterion = torch.nn.L1Loss()
    #ce_loss = LogEntropyLoss()
    watch_criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80], gamma=0.1)
    train_losses = []
    true_losses = []
    valid_accs = []
    for epoch in range(epochs):
        train_loss = train(train_loader, model, optimizer, train_criterion, use_cuda, epoch)
        train_losses.append(train_loss)
        true_loss = watch(watch_loader, model, watch_criterion, use_cuda, epoch)
        true_losses.append(true_loss)
        val_acc = valid(valid_loader, model, use_cuda, epoch)
        valid_accs.append(val_acc)
        # if epoch % 200 == 1:
        #     log(str(epoch), train_losses, true_losses, valid_accs, entropy_losses, cons_losses)
        scheduler.step()
    #log(str(epoch), train_losses, true_losses, valid_accs, entropy_losses, cons_losses)
    plot_curve(epochs, train_losses, true_losses, valid_accs, label)


if __name__ == '__main__':
    main()
        
        

