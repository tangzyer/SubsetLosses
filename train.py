from utils import load_data, load_data_from_npz
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
from logentropy import MAELoss



epochs = 100
lr = 0.1
launchTimeStamp = "12_36_06_12"
label = "l1loss-10classes-card2"
checkpoint = True



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

def train_pseodu(train_loader, model, optimizer, train_criterion, use_cuda, epoch):
    sum_loss = 0.0
    for data in train_loader:
        img, label = data
        bs = len(label)
        label = label.unsqueeze(-1)
        one_hot_label = torch.zeros(bs, 10).scatter_(1, label, 1)
        if use_cuda:
            img = Variable(img).cuda()
            label = Variable(label).cuda()
            one_hot_label = Variable(one_hot_label).cuda()
        else:
            img = Variable(img)
            label = Variable(label)
            one_hot_label = Variable(one_hot_label)
        out = model(img)
        loss = train_criterion(out, one_hot_label)
        print_loss = loss.data.item()
        sum_loss += print_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
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

def plot_curve(epochs, train_losses, pseudo_losses, true_losses, pseudo_true_losses, valid_accs, label):
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


def main():
    use_cuda = torch.cuda.is_available()
    print("cuda:",use_cuda)
    #model = mlp()
    #checkpoint = True
    model = torchvision.models.resnet18(pretrained=True)
    fc_in = model.fc.in_features  # 获取全连接层的输入特征维度
    model.fc =torch.nn.Linear(fc_in, 10)
    model_psuedo = torchvision.models.resnet18(pretrained=True)
    fc_in = model_psuedo.fc.in_features  # 获取全连接层的输入特征维度
    model_psuedo.fc =torch.nn.Linear(fc_in, 10)
    if use_cuda:
        model = model.cuda()
        model_psuedo = model_psuedo.cuda()
    train_loader, watch_loader, psuedo_loader, valid_loader = load_data_from_npz()
    train_criterion = MAELoss()
    #ce_loss = LogEntropyLoss()
    watch_criterion = MAELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    optimizer_p = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    #model, optimizer = load_checkpoint(model, 'models/m-10_36_06_09-118.4261.pth.tar',optimizer)
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[256, 1000, 3000], gamma=0.1)
    scheduler_p = torch.optim.lr_scheduler.MultiStepLR(optimizer_p, milestones=[256, 1000, 3000], gamma=0.1)
    train_losses = []
    true_losses = []
    pseudo_losses = []
    pseudo_true_losses = []
    valid_accs = []
    for epoch in range(epochs):
        train_loss = train(train_loader, model, optimizer, train_criterion, use_cuda, epoch)
        train_losses.append(train_loss)
        pseudo_train_loss = train_pseodu(psuedo_loader, model_psuedo, optimizer_p, train_criterion, use_cuda, epoch)
        pseudo_losses.append(pseudo_train_loss)
        true_loss = watch(watch_loader, model, watch_criterion, use_cuda, epoch)
        pseudo_true_loss = watch(watch_loader, model_psuedo, watch_criterion, use_cuda, epoch, l='p')
        pseudo_true_losses.append(pseudo_true_loss)
        true_losses.append(true_loss)
        val_acc = valid(valid_loader, model, use_cuda, epoch)
        valid_accs.append(val_acc)
        # if epoch % 200 == 1:
        #     log(str(epoch), train_losses, true_losses, valid_accs, entropy_losses, cons_losses)
        #scheduler.step()
        #scheduler_p.step()
        if epoch % 1000 == 1 or epoch % 1000 == 999:
            lossMIN = true_loss
            torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 
                            'optimizer': optimizer.state_dict()},
                           './newmodels' + '/m-' + launchTimeStamp + '-' + str("%.4f" % lossMIN) + '.pth.tar')
            lossMIN = pseudo_train_loss
            torch.save({'epoch': epoch + 1, 'state_dict': model_psuedo.state_dict(), 
                            'optimizer': optimizer_p.state_dict()},
                           './newmodels' + '/p-' + launchTimeStamp + '-' + str("%.4f" % lossMIN) + '.pth.tar')
            
    #log(str(epoch), train_losses, true_losses, valid_accs, entropy_losses, cons_losses)
    plot_curve(epochs, train_losses, pseudo_losses, true_losses, pseudo_true_losses, valid_accs, label)

if __name__ == '__main__':

    main()
        
        

