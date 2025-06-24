import os, time, random
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
from timm.data import create_transform


def build_train_transforms():
    transform = create_transform(
            input_size=224,
            is_training=True,
            color_jitter=0.4,
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation='bicubic',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
        )
    return transform


def get_dataloaders_imagenet(BATCH_SIZE = 128):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    train_transform = build_train_transforms()
    # print(train_transform) 
    
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])  # here is actually the validation dataset

    st = time.time()
    train_data = datasets.ImageNet(root='/data/imagenet_data', split='train', transform=train_transform)
    print('time:', time.time()-st)
    test_data = datasets.ImageNet(root='/data/imagenet_data', split='val', transform=test_transform)
        
    train_loader = torch.utils.data.DataLoader(
                    train_data,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=True)
    
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=BATCH_SIZE,
                                              shuffle=False,
                                              num_workers=4,
                                              pin_memory=True)
    return train_loader, test_loader


def get_dataloaders_cifar10(BATCH_SIZE = 128):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    # train_transform = build_train_transforms()
    print(train_transform)
    
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])  # here is actually the validation dataset

    st = time.time()
    train_data = datasets.CIFAR10(root='/data/cifar10_data', train=True, transform=train_transform, download=True)
    print('time:', time.time()-st)
    test_data = datasets.CIFAR10(root='/data/cifar10_data', train=False, transform=test_transform)
        
    train_loader = torch.utils.data.DataLoader(
                    train_data,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=True)
    
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=BATCH_SIZE,
                                              shuffle=False,
                                              num_workers=4,
                                              pin_memory=True)
    return train_loader, test_loader