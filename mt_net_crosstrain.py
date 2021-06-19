import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import cv2
from data_loader import MyDataset,MyDataset_MTL,read_txt
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# 定义读取文件的格式
def default_loader(path):
    # return Image.open(path)
    return Image.open(path).convert('RGB')


# 首先继承上面的dataset类。然后在__init__()方法中得到图像的路径，然后将图像路径组成一个数组，这样在__getitim__()中就可以直接读取：


# transform2 = transforms.Compose([
#     transforms.RandomResizedCrop((224, 224)),
#     transforms.RandomHorizontalFlip(0.5),
#     transforms.ColorJitter(brightness=0.4, contrast=0.4,
#                            saturation=0.4),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
#
# test_transform2 = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.CenterCrop((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

transform = transforms.Compose([
    # transforms.CenterCrop(256),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])  # Magic number - -!

test_transform = transforms.Compose([
    # transforms.CenterCrop(256),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])  # Magic number - -!


# 数据集加载方式设置
txt_path = 'data/1-5组所有目标检测只01.txt'

file_list = read_txt(txt_path)
train_list = file_list[:2822]
test_list = file_list[2822:]
# all_data = MyDataset(txt=txt_path,transform=None)
# print(len(all_data))
train_data = MyDataset_MTL(list=train_list,transform=transform)
test_data = MyDataset_MTL(list=test_list,transform=test_transform)
# train_size = int(0.9 * len(all_data))
# test_size = len(all_data) - train_size
# # 随机划分训练集和验证集
# train_data, test_data = torch.utils.data.random_split(all_data, [train_size, test_size])
# 然后就是调用DataLoader和刚刚创建的数据集，来创建dataloader，这里提一句，loader的长度是有多少个batch，所以和batch_size有关
train_loader = DataLoader(dataset=train_data, batch_size=8, shuffle=True, num_workers=4)
test_loader = DataLoader(dataset=test_data, batch_size=8, shuffle=False, num_workers=4)
print('num_of_trainData:', len(train_data))
print('num_of_trainData:', len(test_data))


class VGGNet(nn.Module):
    def __init__(self, num_classes=2):  # num_classes，此处为 二分类值为2
        super(VGGNet, self).__init__()
        model = models.resnet18(pretrained=False)
        # print(model)
        # exit()
        num_label, num_shape, num_thick, num_echo = 2, 3, 3, 2
        fc_features = model.fc.in_features
        model.fc = nn.Sequential()
        # model.fc = nn.Sequential(    # 定义自己的分类层
        #         nn.Linear(fc_features, 1024),  # 512 * 7 * 7不能改变 ，由VGG16网络决定的，第二个参数为神经元个数可以微调
        #         nn.Linear(1024, 3),
        #         )
        self.fc1 = nn.Sequential(nn.Linear(fc_features, num_label))
        self.fc2 = nn.Sequential(nn.Linear(fc_features, num_shape))
        self.fc3 = nn.Sequential(nn.Linear(fc_features, num_thick))
        self.fc4 = nn.Sequential(nn.Linear(fc_features, num_echo))
        self.softmax = nn.Softmax(dim=1)
        self.features = model
        # fc_features = model.fc.in_features

    def forward(self, x):
        x = self.features(x)
        # print('feature:',x.shape)
        x = x.view(x.size(0), -1)
        # print('view:',x.shape)
        pred_label = self.fc1(x)
        pred_shape = self.fc2(x)
        pred_thickness = self.fc3(x)
        pred_echo = self.fc4(x)
        return pred_label, pred_shape, pred_thickness, pred_echo


if __name__ == '__main__':
    EPOCH = 150
    batch_size = 8
    mode1_vgg = VGGNet()
    mode1_vgg = mode1_vgg.cuda()

    optimizer = optim.Adam(mode1_vgg.parameters(), lr=0.0001, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    # 动态调整学习率
    loss_func = nn.CrossEntropyLoss()
    # 保证每次初始化一样
    # np.random.seed(1)
    # torch.manual_seed(1)
    # torch.cuda.manual_seed_all(1)
    # torch.backends.cudnn.deterministic = True
    best_acc = 0.
    aver_train_acc = []
    aver_test_acc = []
    aver_train_loss = []
    aver_test_loss = []
    for epoch in range(EPOCH):
        print('EPOCH {}'.format(epoch + 1))
        # training-----------------------------
        mode1_vgg.train()
        # print("net have {} paramerters in total".format(sum(x.numel() for x in mode1_vgg.parameters())))
        train_loss = 0.
        train_acc = 0.
        acc2, acc3, acc4 = 0., 0., 0.
        FN, FP, TP, TN = 0, 0, 0, 0
        Precision, Recall = 0, 0
        for image, (lable, shape, thickness, echo) in train_loader:
            image, lable, shape, thickness, echo = \
                image.cuda(), lable.cuda(), shape.cuda(), thickness.cuda(), echo.cuda()
            pred_label, pred_shape, pred_thickness, pred_echo = mode1_vgg(image)
            # out = F.softmax(out, dim=1)
            loss = loss_func(pred_label, lable)
            # print(pred_label.shape,lable.shape)
            # exit()
            loss2 = loss_func(pred_shape, shape)
            loss3 = loss_func(pred_thickness, thickness)
            loss4 = loss_func(pred_echo, echo)
            train_loss = loss + 0.5*loss2 + 0.5*loss3 + 0.5*loss4

            pred = torch.max(pred_label, 1)[1].cuda()
            train_correct = (pred == lable).sum().cuda()

            pred_shape = torch.max(pred_shape, 1)[1]
            pred_thickness = torch.max(pred_thickness, 1)[1]
            pred_echo = torch.max(pred_echo, 1)[1]

            shape_correct = (pred_shape == shape).sum().cuda()
            thickness_correct = (pred_thickness == thickness).sum().cuda()
            echo_correct = (pred_echo == echo).sum().cuda()

            train_acc += train_correct.item()
            acc2 += shape_correct.item()
            acc3 += thickness_correct.item()
            acc4 += echo_correct.item()

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        # 记录数据
        scheduler.step()
        aver_train_acc.append(train_acc / (len(train_data)))
        aver_train_loss.append(train_loss.item() / (len(train_data)))

        print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(
            train_data)), train_acc / (len(train_data))))
        print('Shape_correct: {:.6f},Thickness_correct: {:.6f},Echo_correct: {:.6f} '.
              format(acc2 / (len(train_data)),
                     acc3 / (len(train_data)),
                     acc4 / (len(train_data))))

        # evaluation-------------------------------- #
        mode1_vgg.eval()
        eval_loss = 0.
        eval_acc = 0.
        eval_acc2, eval_acc3, eval_acc4 = 0., 0., 0.
        for image, (lable, shape, thickness, echo) in test_loader:
            image, lable, shape, thickness, echo = \
                image.cuda(), lable.cuda(), shape.cuda(), thickness.cuda(), echo.cuda()
            pred_label, pred_shape, pred_thickness, pred_echo = mode1_vgg(image)
            # out = F.softmax(out,dim=1)
            loss = loss_func(pred_label, lable)
            pred = torch.max(pred_label, 1)[1].cuda()

            eval_loss += loss.item()
            num_correct = (pred == lable).sum().cuda()
            eval_acc += num_correct.item()

            pred_shape = torch.max(pred_shape, 1)[1]
            pred_thickness = torch.max(pred_thickness, 1)[1]
            pred_echo = torch.max(pred_echo, 1)[1]

            e_shape_correct = (pred_shape == shape).sum().cuda()
            e_thickness_correct = (pred_thickness == thickness).sum().cuda()
            e_echo_correct = (pred_echo == echo).sum().cuda()
            eval_acc2 += e_shape_correct.item()
            eval_acc3 += e_thickness_correct.item()
            eval_acc4 += e_echo_correct.item()
            # ---------------------------------------------------------------
            # 计算精准率和召回率
            zes = torch.zeros(len(lable)).type(torch.LongTensor).cuda()  # 全0变量
            ons = torch.ones(len(lable)).type(torch.LongTensor).cuda()  # 全1变量
            train_correct01 = ((pred == zes) & (lable == ons)).sum()  # 原标签为1，预测为 0 的总数
            train_correct10 = ((pred == ons) & (lable == zes)).sum()  # 原标签为0，预测为1  的总数
            train_correct11 = ((pred == ons) & (lable == ons)).sum()
            train_correct00 = ((pred == zes) & (lable == zes)).sum()

            FN += train_correct01.item()
            FP += train_correct10.item()
            TP += train_correct11.item()
            TN += train_correct00.item()
            # ----------------------------------------------------------------------
        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)
        # Specificity = TN / (FP + TN)
        F1_score = 2 * Precision * Recall / (Precision + Recall)

        print('\nPrecision:{:.6f}, Recall:{:.6f}, F1-score:{:.6f}'
              .format(Precision, Recall, F1_score))

        aver_test_acc.append(eval_acc / (len(test_data)))
        aver_test_loss.append(eval_loss / (len(test_data)))

        print('Val Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
            test_data)), eval_acc / (len(test_data))))
        print('Shape_correct: {:.6f},Thickness_correct: {:.6f},Echo_correct: {:.6f} '.
              format(eval_acc2 / (len(test_data)),
                     eval_acc3 / (len(test_data)),
                     eval_acc4 / (len(test_data))) + '\n')

        if eval_acc / (len(test_data)) > best_acc:
            best_acc = eval_acc / (len(test_data))
            print('best_acc:', best_acc)
            torch.save({
                'state_dict': mode1_vgg.state_dict()
            }, 'model_for_Dachuang.pkl', _use_new_zipfile_serialization=False)
            # torch.save(mode1_vgg.state_dict(), './best_acc.pkl')

    # print('\nTrain average acc:', np.mean(aver_train_acc[100:150]), '\n',
    #       'Train average loss:', np.mean(aver_train_loss[100:150]), '\n',
    #       'Test average acc:', np.mean(aver_test_acc[100:150]), '\n',
    #       'Test average loss:', np.mean(aver_test_loss[100:150]), '\n')
    print('Best accuracy is:' + str(best_acc))
