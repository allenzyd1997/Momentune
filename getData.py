import os

import numpy as np
import torch
import torch.nn as nn
import tarfile

from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor


transforms = transforms.Compose([transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor()])

def splitTrainTestData(dataset):
    train_datasets = datasets.ImageFolder(dataset, transform=transforms)

    train_size = 14000
    test_size = len(train_datasets) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(train_datasets, [train_size,test_size] )

    train_dataset = train_dataset.dataset
    return train_dataset, test_dataset

def getIndoorData(configs):
    data_dir = "/home/zyd/exp1/re_para/data_indoor/Images"
    torch.manual_seed(configs.seed)
    dataset, test_ds = splitTrainTestData(data_dir)

    val_size = (len(dataset) // 10 )* 3
    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    # train_ds = train_ds.dataset
    train_dl = DataLoader(train_ds, configs.batch_size, shuffle=True, num_workers=configs.num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, configs.batch_size * 2, configs.num_workers, pin_memory=True)
    test_dl = DataLoader(test_ds, configs.batch_size * 2, configs.num_workers, pin_memory=True)
    return train_dl, val_dl, test_dl

def getCifar100Data(configs):
    data_dir = "/home/zyd/exp1/re_para/data_cifar/cifar100"
    dataset = ImageFolder(data_dir + '/train', transform=ToTensor())

    torch.manual_seed(configs.seed)

    val_size = len(dataset)//10*2
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_dl = DataLoader(train_ds, configs.batch_size, shuffle=True, num_workers=configs.num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, configs.batch_size * 2, configs.num_workers, pin_memory=True)

    test_ds = ImageFolder(data_dir + '/test', transform=ToTensor())
    test_dl = DataLoader(test_ds, configs.batch_size * 2, configs.num_workers, pin_memory=True)

    return train_dl, val_dl, test_dl

def getCifar10Data(configs):

    # dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
    # dataset_url = "http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar"
    #
    # download_url(dataset_url, '.')
    # with tarfile.open('./cifar100.tgz', 'r:gz') as tar:
    #     tar.extractall(path='./data_cifar')
    data_dir = "/home/zyd/exp1/re_para/data_cifar/cifar10"
    # airplane_files = os.listdir(data_dir + "/train/airplane")
    # print('No. of training examples for airplanes:', len(airplane_files))
    # print(airplane_files[:5])

    dataset = ImageFolder(data_dir + '/train', transform=ToTensor())

    torch.manual_seed(configs.seed)

    val_size = len(dataset)//10*2
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_dl = DataLoader(train_ds, configs.batch_size, shuffle=True, num_workers=configs.num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, configs.batch_size * 2, configs.num_workers, pin_memory=True)

    test_ds = ImageFolder(data_dir + '/test', transform=ToTensor())
    test_dl = DataLoader(test_ds, configs.batch_size * 2, configs.num_workers, pin_memory=True)

    return train_dl, val_dl, test_dl

def getYourData(configs):
    if configs.dataset ==0:
        return getIndoorData(configs)
    elif configs.dataset ==1:
        return getCifar100Data(configs)
    else:
        return getCifar10Data(configs)