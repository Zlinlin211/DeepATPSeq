# -*- coding: utf-8 -*-
import sys

import torch
from torch import device, nn
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset


class DataHandle(Dataset):

    def __init__(self, path):
        super(DataHandle, self).__init__()
        self.data_info = self.get_protein_info(path)

    def __getitem__(self, index):
        feature = self.data_info[index]
        return feature

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_protein_info(path):
        file = open(path, "r")
        data_info = []
        for line in file:

            line = line.split(',')
            xy = np.loadtxt(line)
            # 一行代表一个滑动窗口，将一行数据读取转化为win_col*win_size的矩阵
            x_data = torch.from_numpy(xy[:]).view(win_col, fea_num)

            # 读取二维矩阵的每一列，把数据变成1维的，方便计算
            fea_info = []
            for i in range(0, fea_num):
                # 读取矩阵的每一列，win_size个win_col*1,通过transpose转化维度，使得数据变成win_size个1*win_col
                fea_in = x_data[:, i:i + 1]
                fea_in = torch.transpose(fea_in, 0, 1)
                fea_info.append(fea_in)
            # fea_info中存放每个残基的特征，就pssm而言，一个残基有20个特征，滑动窗口的大小为17，即20个1*17的数据，通过torch.stack进行维度堆叠，变成20*1*17
            fea_info = torch.stack(fea_info)  # 20*1*17
            fea_info = torch.transpose(fea_info, 0, 2)  # 17*1*20
            fea_info = torch.transpose(fea_info, 0, 1)  # 1*17*20

            data_info.append(fea_info)

        return data_info


class CnnModel(nn.Module):
    def __init__(self):
        super(CnnModel, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(fea_num, 64, kernel_size=3, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=1)
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, stride=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=1)
        )

        self.fc = nn.Sequential(
            nn.Linear(2560, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )
        self.sigmod = nn.Sigmoid()

    def forward(self, out):
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.sigmod(out)

        return out


if __name__ == '__main__':
    dcnn_model = sys.argv[1]
    dcnn_test = sys.argv[2]
    dcnn_prod = sys.argv[3]
    fea_num = 21
    win_col = 17

    model_save_file = dcnn_model
    dcnn_model = CnnModel()
    dcnn_model.load_state_dict(torch.load(model_save_file, map_location=lambda storage, loc: storage))
    dcnn_model.eval()

    test_dataset = DataHandle(path=dcnn_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, drop_last=False)

    f = open(dcnn_prod, 'w')

    for i, data in enumerate(test_loader):
        features = data

        features = features

        # 批训练，batchSize=128，in_channel:20 输入数据是维度1*17*20 ，在维度为dim=1的位置上压缩张量（128*17*20）
        features = Variable(torch.squeeze(features, dim=1).float(), requires_grad=False)

        # 转换张量的维度，128*17*20-->128*20*17,符合1d的卷积要求
        features = torch.transpose(features, 1, 2)

        output = dcnn_model(features)

        # 输出模型预测样本类别的索引值
        _, pre = torch.max(output.data, 1)

        # 将模型预测二分类的概率写入到文件中，保存成libsvm的格式
        if int(output[0][0] > 0.76):  # 表示真实标签是Negative,labels = [1,0]
            f.write('0.000000' + '\t' + str(format(output[0][1].item(), '.6f')) + '\t' + str(
                format(output[0][0].item(), '.6f')) + '\n')
        else:
            f.write('1.000000' + '\t' + str(format(output[0][1].item(), '.6f')) + '\t' + str(
                format(output[0][0].item(), '.6f')) + '\n')
    f.close()
