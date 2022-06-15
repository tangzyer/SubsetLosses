from copy import deepcopy
import torch.nn as nn
import torch
import numpy as np
import math
from sklearn.metrics import accuracy_score
import torch.utils.data as Data
import torch.utils.data 
import torchvision
from torchvision import datasets
from zmq import TOS
from subset import subset
import torchvision.transforms as transforms
import cifar10 as dataset
from random import choice
from collections import Counter


batch_size = 100
augment_K = 4

sub = subset(10)

# transform_train = transforms.Compose([
#         dataset.RandomPadandCrop(32),
#         dataset.RandomFlip(),
#         dataset.ToTensor(),
#         transforms.Normalize([0.395, 0.431, 0.516], [0.246, 0.223, 0.234])
#     ])


# transform_val = transforms.Compose([
#     # dataset.RandomPadandCrop(32),
#     # dataset.RandomFlip(),
#     # dataset.Normalize(),
#     dataset.ToTensor(),
#     transforms.Normalize([0.395, 0.431, 0.516], [0.246, 0.223, 0.234])
    
# ])
class ToSubset(object):
    def __init__(self, mode=None):
        self.mode = mode
    

    def __call__(self, y):
        subset_y = sub.index_to_limited_subset(y)[1]
        return torch.from_numpy(np.array(subset_y))

transform_targets = ToSubset()

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


class MyDataset(Data.Dataset):

    def __init__(self, data, targets, transform=None):
        super(MyDataset, self).__init__()
        self.data = data
        self.targets = targets
        self.transform = transform
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)

class MyDatasetWithAugmentation(Data.Dataset):
    def __init__(self, data, targets, transform=None, augment_K=2):
        super(MyDatasetWithAugmentation, self).__init__()
        self.data = data
        self.targets = targets
        self.transform = transform
        self.K = augment_K
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        augments = []
        if self.transform is not None:
            for i in range(self.K):
                augments.append(self.transform(img))
        return augments, target

    def __len__(self):
        return len(self.data)

def load_data():
    

    train_data = datasets.CIFAR10(root='.', train=True, download=True)
    test_data = datasets.CIFAR10(root='.', train=False, download=True)

    x_train = train_data.data
    x_test = test_data.data
    y_train = train_data.targets
    y_test = test_data.targets
    

    X_train_true = np.zeros(shape=(50000, 3, 32, 32), dtype=float)
    validation_X = np.zeros(shape=(10000, 3, 32, 32), dtype=float)

    validation_y = y_test
    # X_test_true = np.zeros(shape=(10000, 3, 32, 32), dtype=float)

    # # data preparation
    for index, sample in enumerate(x_train):
        X_train_true[index] = sample.reshape(3,32,32).astype(float)/255

    for index, sample in enumerate(x_test):
        validation_X[index] = sample.reshape(3,32,32).astype(float)/255

    # for index, sample in enumerate(x_test):
    #     X_test_true[index] = sample.reshape(3,32,32).astype(float) / 255

    # split
    from sklearn.model_selection import train_test_split

    train_X, train_y = X_train_true, y_train

    #train_X, validation_X, train_y, validation_y = train_test_split(X_train_true, y_train, test_size=0.1, random_state=42)

    #train_X, _, train_y, _ = train_test_split(X_train_true, y_train, test_size=0.05, random_state=42)
    #test_X, validation_X, test_y, validation_y = train_test_split(X_test_true, y_test, test_size=0.2, random_state=42)

    #print(len(validation_X))
    #print(Counter(validation_y))
    print(Counter(train_y))

    test_labels = []
    train_labels = []
    #X_validation = []
    #validation_labels = []
    true_labels = []
    psuedo_labels =[]
    #X_train = []
    X_test = []
    monitor = [0,0,0,0]


    for index, y in enumerate(train_y):
        #size_of_subset, subset_y, obfuscated_y = sub.index_to_stack_obfuscated(y)
        sub_label, multi_hot_subset = sub.index_to_limited_subset(y, max_class=10)
        #X_train.append(train_X[index])
        train_labels.append(multi_hot_subset)
        #true_labels.append(y)
        psuedo_labels.append(choice(sub_label))
        # if y == 0:
        #     for i in sub_label:
        #         monitor[i] += 1
    
    true_labels = train_y
    #print(monitor)


    # for index, y in enumerate(test_y):
    #     if y in classes:
    #         X_test.append(test_X[index])
    #         test_labels.append(y)

    # for index,y in enumerate(validation_y):
    #     if y in classes:
    #         X_validation.append(validation_X[index])
    #         validation_labels.append(y)

    print(torch.utils.data.dataloader.__file__)
  
    input = np.array(train_X).astype(np.float32)
    watch_input = deepcopy(input)
    label = np.array(train_labels).astype(np.int64)
    true_labels = np.array(train_y).astype(np.long)
    psuedo_labels = np.array(psuedo_labels).astype(np.long)
    np.savez('./data/train.npz', train_x=input, subset_y=label, true_y=true_labels, pseudo_y=psuedo_labels)
    train_dataset = MyDataset(input, label, transform_train)
    #torch_dataset_withaug = MyDatasetWithAugmentation(input, label, transform_train, augment_K)
    watch_dataset = MyDataset(watch_input, true_labels, transform_train)
    train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #train_loader_augmentation = Data.DataLoader(torch_dataset_withaug, batch_size=batch_size, shuffle=True)
    watch_loader = Data.DataLoader(watch_dataset, batch_size=batch_size, shuffle=True)


    val_input = np.array(validation_X).astype(np.float32)
    val_label = np.array(validation_y).astype(np.long)
    np.savez('./data/valid.npz', valid_x=val_input, valid_y=val_label)
    valid_dataset = MyDataset(val_input, val_label, transform_val)
    valid_loader = Data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, watch_loader, valid_loader

def load_data_custom():
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train, target_transform=transform_targets) #训练数据集
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)   #生成一个个batch进行批训练，组成batch的时候顺序打乱取
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)
    return trainloader, testloader





def load_data_from_npz(mode='subset'):
    data_train = np.load('./data/train.npz')
    input = data_train['train_x']
    label = data_train['subset_y']
    true_labels = data_train['true_y']
    psuedo_labels = data_train['pseudo_y']
    torch_dataset = MyDataset(input, label, transform_train)
    watch_dataset = MyDataset(input, true_labels, transform_val)
    pseudo_dataset = MyDataset(input, psuedo_labels, transform_train)
    train_loader = Data.DataLoader(torch_dataset, batch_size=batch_size, shuffle=True)
    # else:
    #     train_loader = Data.DataLoader(pseudo_dataset, batch_size=batch_size, shuffle=True)
    psuedo_loader = Data.DataLoader(pseudo_dataset, batch_size=batch_size, shuffle=True)
    watch_loader = Data.DataLoader(watch_dataset, batch_size=batch_size, shuffle=False)
    
    data_valid = np.load('./data/valid.npz')
    input = data_valid['valid_x']
    label = data_valid['valid_y']
    torch_dataset = MyDataset(input, label, transform_val)
    valid_loader = Data.DataLoader(torch_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, watch_loader, psuedo_loader, valid_loader
