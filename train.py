from utils import *
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
from logentropy import LogEntropyLoss, MAELoss



epochs = 500
lr = 0.01
launchTimeStamp = "23_19_06_22"
label = "l1loss-10classes-card2"
checkpoint = True
use_cuda = torch.cuda.is_available()



def train(train_loader, model, optimizer, train_criterion, watch_criterion, use_cuda, epoch):
    model.train()
    sum_loss = 0.0
    true_loss = 0.0
    l = len(train_loader)
    for data in train_loader:
        img, (label, true_label) = data
        if use_cuda:
            img = Variable(img).cuda()
            label = Variable(label).cuda()
            true_label = Variable(true_label).cuda()
        else:
            img = Variable(img)
            label = Variable(label)
            true_label = Variable(true_label)
        out = model(img)
        loss = train_criterion(out, label)
        print_loss = loss.data.item()
        sum_loss += print_loss/l
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            true_label = F.one_hot(true_label, 10)
            loss = watch_criterion.__call__(out, true_label)
            print_loss = loss.data.item()
            true_loss += print_loss/l
    print('Epoch', epoch, 'Train Loss:', sum_loss)
    print('Epoch', epoch, 'True Loss:', true_loss)
    return sum_loss, true_loss

def train_pseodu(train_loader, model, optimizer, train_criterion, use_cuda, epoch):
    sum_loss = 0.0
    l = len(train_loader)
    for data in train_loader:
        model.train()
        img, (_, label) = data
        if use_cuda:
            img = Variable(img).cuda()
            label = Variable(label).cuda()
        else:
            img = Variable(img)
            label = Variable(label)
        out = model(img)
        #pred = out.argmax(dim=1)
        #label_one = label.argmax(dim=1)
        #total_correct += torch.eq(pred,label_one).float().sum().item() #分别为是否相等，scalar tensor转换为float，求和，拿出值
       # total_num += label.size(0)
        #loss = train_criterion(out, one_hot_label)
        label = F.one_hot(label, 10)
        loss = train_criterion(out, label)
        print_loss = loss.data.item()
        sum_loss += print_loss/l
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    #acc = total_correct/total_num
    print('Epoch', epoch, 'Pseudo Train Loss:', sum_loss)
    return sum_loss


def watch(watch_loader, model, watch_criterion, use_cuda, epoch, l=None):
    sum_loss = 0.0
    with torch.no_grad():
        for data in watch_loader:
            val_inputs, label = data
            bs = len(label)
            label = label.unsqueeze(-1)
            one_hot_label = torch.zeros(bs, 10).scatter_(1, label, 1)
            if use_cuda:
                val_inputs = Variable(val_inputs).cuda()
                label = Variable(label).cuda()
                one_hot_label = Variable(one_hot_label).cuda()
            val_outputs = model(val_inputs)
            loss = watch_criterion(val_outputs, one_hot_label)
            print_loss = loss.data.item()
            sum_loss += print_loss
        if l is None:
            print('Epoch:', epoch, 'True Loss:', sum_loss)
        else:
            print('Epoch:', epoch, 'Pseudo True Loss:', sum_loss)
    return sum_loss

def valid(valid_loader, model, criterion, use_cuda, epoch):
    total_correct = 0.0
    total_num = 0.0
    total_loss = 0.0
    with torch.no_grad():
        model.eval()
        l = len(valid_loader)
        for data in valid_loader:
            val_inputs, label = data
            if use_cuda:
                val_inputs = Variable(val_inputs).cuda()
                label = Variable(label).cuda()
            one_hot_label = F.one_hot(label, 10)
            val_outputs = model(val_inputs)
            loss = criterion.__call__(val_outputs, one_hot_label)
            loss = loss.data.item()
            total_loss += loss/l
            pred = val_outputs.argmax(dim=1)
            total_correct += torch.eq(pred,label).float().sum().item() #分别为是否相等，scalar tensor转换为float，求和，拿出值
            total_num += label.size(0)
        acc = total_correct/total_num
        print('Epoch:', epoch, 'Val Acc:', acc)
        print('Epoch:', epoch, 'Val Loss:', total_loss)
    return acc, total_loss

def plot_curve(epochs, train_losses, pseudo_losses, true_losses, pseudo_true_losses, valid_accs, label):
    plt.figure(figsize=(8, 12))
    epoch_num = epochs
    x1 = range(0, epoch_num)
    x2 = range(0, epoch_num)
    x3 = range(0, epoch_num)
    x4 = range(0, epoch_num)
    plt.subplot(5, 1, 1)
    plt.plot(x1, valid_accs, 'o-')
    plt.ylabel('Val Acc')
    plt.subplot(5, 1, 2)
    plt.plot(x2, train_losses, '.-')
    plt.ylabel('Train Loss')
    plt.subplot(5, 1, 3)
    plt.plot(x1, pseudo_losses, '.-')
    plt.ylabel('Pseudo Train Loss')
    plt.subplot(5, 1, 4)
    plt.plot(x1, pseudo_true_losses, '.-')
    plt.ylabel('Pseudo True Loss')
    plt.subplot(5, 1, 5)
    plt.plot(x3, true_losses, '.-')
    plt.xlabel('epochs')
    plt.ylabel('True Loss')
    plt.savefig('./logs/'+label +".png")


def log(tag, train_loss, true_loss, val_acc, e_losses, c_losses):
    data = {'train loss':train_loss, "true loss":true_loss, "val acc":val_acc, 'loss term 1':e_losses, 'loss term 2':c_losses}
    df = pd.DataFrame(data)
    df.to_csv('./logs/'+label+tag+'.csv')


def load_checkpoint(model, checkpoint_PATH, optimizer):
    if checkpoint != None:
        model_CKPT = torch.load(checkpoint_PATH)
        model.load_state_dict(model_CKPT['state_dict'])
        print('loading checkpoint!')
        optimizer.load_state_dict(model_CKPT['optimizer'])
    return model, optimizer

def log_dict(tag, data):
    df = pd.DataFrame(data)
    df.to_csv('./logs/'+tag+'.csv')


def test_true_label(model, true_loader, valid_loader, train_criterion, valid_criterion,  optimizer, scheduler=None):
    train_losses = []
    val_losses = []
    val_accs = []
    for epoch in range(epochs):
        train_loss = train_pseodu(true_loader, model, optimizer, train_criterion, use_cuda, epoch)
        val_acc, val_loss = valid(valid_loader, model, valid_criterion, use_cuda, epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        if scheduler is not None:
            scheduler.step()
    data = {'Train Loss':train_losses, 'Val Loss':val_losses, 'Val Acc':val_accs}
    log_dict('accurateLabell1loss'+launchTimeStamp, data)     


def test_subset_label(model, watch_loader, valid_loader, train_criterion, watch_criterion, valid_criterion, optimizer, scheduler=None):
    train_losses = []
    true_losses = []
    valid_losses = []
    valid_accs = []
    for epoch in range(epochs):
        train_loss, true_loss = train(watch_loader, model, optimizer, train_criterion, watch_criterion, use_cuda, epoch)
        train_losses.append(train_loss)
        true_losses.append(true_loss)
        val_acc, val_loss = valid(valid_loader, model, valid_criterion, use_cuda, epoch)
        valid_accs.append(val_acc)
        valid_losses.append(val_loss)
        if scheduler is not None:
            scheduler.step()
    data = {'Train Loss':train_losses, 'True Loss':true_losses, 'Val Loss':valid_losses, 'Val Acc':valid_accs}
    log_dict('l1loss'+launchTimeStamp, data)     


def main():
    
    print("cuda:",use_cuda)
    model = torchvision.models.resnet18(pretrained=True)
    fc_in = model.fc.in_features  # 获取全连接层的输入特征维度
    model.fc =torch.nn.Linear(fc_in, 10)
    model_psuedo = torchvision.models.resnet18(pretrained=True)
    fc_in = model_psuedo.fc.in_features  # 获取全连接层的输入特征维度
    model_psuedo.fc =torch.nn.Linear(fc_in, 10)
    if use_cuda:
        model = model.cuda()
        model_psuedo = model_psuedo.cuda()
    watch_loader, valid_loader = load_data_custom()
    #train_criterion = torch.nn.CrossEntropyLoss()
    train_criterion = MAELoss()
    valid_criterion = MAELoss()
    #watch_criterion = MAELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    #model, optimizer = load_checkpoint(model, 'newmodels/m-22_15_06_13-123.0882.pth.tar',optimizer)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40], gamma=0.1)
    test_true_label(model, watch_loader, valid_loader, train_criterion, valid_criterion, optimizer, scheduler)
    #test_subset_label(model, watch_loader, valid_loader, train_criterion, watch_criterion, valid_criterion, optimizer, scheduler)



if __name__ == '__main__':

    main()
        
        

