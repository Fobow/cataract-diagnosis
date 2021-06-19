import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.optim as optim
import os
from tqdm import tqdm
import pickle as pkl
import cv2
import random

def read_txt(txt, shuffle=False):
    # 读一个txt，返回list
    fh = open(txt, 'r', encoding='utf-8')  # 按照传入的路径和txt文本参数，打开这个文本，并读取内容
    imgs = []
    for line in fh:  # 迭代该列表#按行循环txt文本中的内
        line = line.strip('\n')
        line = line.rstrip('\n')  # 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
        words = line.split()  # 用split将该行分割成列表  split的默认参数是空格，所以不传递任何参数时分割空格
        # print(words)
        imgs.append((words[0], int(words[1]), int(words[2]),
                     int(words[3]), int(words[4])))
    if shuffle is True:
        random.shuffle(imgs)
    return imgs

def read_txt_2(txt, shuffle=False):
    # 读一个txt，返回list
    fh = open(txt, 'r', encoding='utf-8')  # 按照传入的路径和txt文本参数，打开这个文本，并读取内容
    imgs = []
    for line in fh:  # 迭代该列表#按行循环txt文本中的内
        line = line.strip('\n')
        line = line.rstrip('\n')  # 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
        words = line.split()  # 用split将该行分割成列表  split的默认参数是空格，所以不传递任何参数时分割空格
        # print(words)
        imgs.append((words[0], int(words[1])))
    if shuffle is True:
        random.shuffle(imgs)
    return imgs


def default_loader(path):
    # return Image.open(path)
    return Image.open(path).convert('RGB')


# 首先继承上面的dataset类。然后在__init__()方法中得到图像的路径，然后将图像路径组成一个数组，这样在__getitim__()中就可以直接读取：
class MyDataset(Dataset):  # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, list, transform=None, loader=default_loader):  # 初始化一些需要传入的参数
        super(MyDataset, self).__init__()  # 对继承自父类的属性进行初始化

        self.imgs = list
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        fn, label,_,_,_ = self.imgs[index]  # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        img = self.loader(fn)  # 按照路径读取图片
        if self.transform is not None:
            img = self.transform(img)  # 数据标签转换为Tensor
        return img, label  # return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)

class MyDataset_MTL(Dataset):  # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, list, transform=None,loader=default_loader):  # 初始化一些需要传入的参数
        super(MyDataset_MTL, self).__init__()  # 对继承自父类的属性进行初始化
        self.imgs = list
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        fn, label, shape, thickness, echo = self.imgs[index]  # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        img = self.loader(fn)  # 按照路径读取图片
        if self.transform is not None:
            img = self.transform(img)  # 数据标签转换为Tensor
        return img, (label, shape, thickness, echo)
    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)

class MyDataset_2(Dataset):  # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, list, transform=None,loader=default_loader):  # 初始化一些需要传入的参数
        super(MyDataset_2, self).__init__()  # 对继承自父类的属性进行初始化
        self.imgs = list
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        fn, label = self.imgs[index]  # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        img = self.loader(fn)  # 按照路径读取图片
        if self.transform is not None:
            img = self.transform(img)  # 数据标签转换为Tensor
        return img, label
    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)



transform = transforms.Compose([
    #transforms.CenterCrop(256),
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])  # Magic number - -!

test_transform = transforms.Compose([
    #transforms.CenterCrop(256),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])  # Magic number - -!


if __name__ == '__main__':
    # 数据集加载方式设置
    txt = 'data/1-5组所有目标检测只01.txt'
    file_list = read_txt(txt)
    train_list = file_list[:2800]
    test_list = file_list[2800:]
    train_data = MyDataset(train_list, transform=transform)
    test_data = MyDataset(test_list, transform=test_transform)
    # 然后就是调用DataLoader和刚刚创建的数据集，来创建dataloader，这里提一句，loader的长度是有多少个batch，所以和batch_size有关
    train_loader = DataLoader(dataset=train_data, batch_size=8, shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset=test_data, batch_size=8, shuffle=False, num_workers=4)
    print('num_of_trainData:', len(train_data))
    print('num_of_testData:', len(test_data))
