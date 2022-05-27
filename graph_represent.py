import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import copy
import datetime
import random


class MLP(nn.Module):
    def __init__(self, input_size, common_size):
        super(MLP, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, common_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(common_size, common_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        out = self.linear(x)
        return out


def AvgPooling(patches):
    # gloContext = patches[0]
    gloContext = torch.zeros(patches[0].shape).to(device)
    for i in range(len(patches)):
        gloContext += patches[i]
    gloContext /= len(patches)

    return gloContext


def to_tensor(data):
    # return torch.tensor(data, dtype=torch.float)
    return data.clone().detach()

'''
def stack(features):
    X = copy.deepcopy(features[0])
    for i in range(len(features)):
        if i != 0:
            X = torch.cat((X, features[i]), 0)
    return X
'''

# 计算两个向量的距离，用的是欧几里得距离
def distEclud(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    mean = torch.mean(adj)
    b = adj <= mean
    adj = b.float()
    # print(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.Tensor(np.diag(d_inv_sqrt))
    temp = torch.matmul(d_mat_inv_sqrt, adj)
    result = torch.matmul(temp, d_mat_inv_sqrt)
    return result


def cross_correlation(features):
    n = features.shape[0]
    # n = len(features)
    A = torch.zeros((n, n))
    temp = A.numpy()
    for i in range(n-1):
        for j in range(n-i-1):
            # print(type(distEclud(features[i].numpy(),features[i+j+1].numpy())))
            data = distEclud(features[i].cpu().numpy(),features[i+j+1].cpu().numpy())
            temp[i, i+j+1] = copy.deepcopy(data)
            temp[i+j+1, i] = copy.deepcopy(data)
    adj = normalize_adj(A)

    return adj


def get_kfold_data(k, i, X):
    # 返回第 i+1 折 (i = 0 -> k-1) 交叉验证时所需要的训练和验证数据，X_train为训练集，X_valid为验证集
    # fold_size = X.shape[0] // k  # 每份的个数:数据总条数/折数（组数）
    fold_size = len(X) // k

    val_start = i * fold_size
    if i != k - 1:
        val_end = (i + 1) * fold_size
        X_valid = X[val_start:val_end]
        # X_train = torch.cat((X[0:val_start], X[val_end:]), dim=0)
        X_train = X[0:val_start] + X[val_end:]
    else:  # 若是最后一折交叉验证
        X_valid = X[val_start:]  # 若不能整除，将多的case放在最后一折里
        X_train = X[0:val_start]

    return X_train, X_valid


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 定义设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
    print(torch.__version__)
    torch.cuda.empty_cache()
    seed = 1
    k = 5
    torch.manual_seed(seed)
    random.seed(seed)
    densenet = models.densenet121(pretrained=True).to(device)
    densenet.classifier = MLP(input_size=1024, common_size=512).to(device)
    densenet.eval()

    trans_ops = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])

    f_name = open("./wsiname.txt", "r", encoding='UTF-8')
    bag_path = './Bag/'

    # nameList = f_name.readlines()
    nameList = f_name.read().splitlines()
    random.shuffle(nameList)

    for i in range(k):
        print('getDataof ', i+1, 'st fold')
        train_data = []
        test_data = []
        sigtest_data = []
        train_names, test_names = get_kfold_data(k, i, nameList)
        # print(len(train_names), len(test_names))
        num = 0
        for name in train_names:
            img_path = bag_path + name + '/'
            for folder in os.listdir(img_path):
                num += 1
                start_time = datetime.datetime.now()
                print('The ', str(num), 'st Train data named ', folder, end=' ')
                featureList = []
                imgList = []
                fold_path = img_path + folder + '/'
                # for f in os.listdir(fold_path):
                for j, f in enumerate(os.listdir(fold_path)):
                    fp = open(fold_path + f, 'rb')
                    img = Image.open(fp)
                    # image = trans_ops(img).view(-1, 3, 224, 224).to(device)
                    image = trans_ops(img).view(-1, 3, 224, 224)
                    fp.close()
                    imgList.append(image)
                    if (j+1) % 20 == 0:
                        IN = torch.cat(imgList, 0).to(device)  # [N, 3, 244, 244]
                        # print('IN:', IN.shape)
                        OUT = densenet(IN).detach().cpu()
                        # print('OUT:', OUT.shape)
                        featureList.append(OUT)
                        imgList = []
                    # output = densenet(image).detach().cpu()
                    # featureList.append(output)
                    del img
                    del image

                X = torch.cat(featureList, 0)  # [N, 512]
                # print('FEA', X.shape)
                A = cross_correlation(X)
                # print('A', A.shape)
                # X = torch.cat(featureList, 0)  # [N, 512]
                # Y = torch.zeros([1, 2], dtype=torch.float)  # [1, 2] 无毒，有毒
                # Y[0][int(folder.split('_')[-1])] = 1
                # Y = (int(folder.split('_')[-1]) == 1)
                Y = (int(folder.split('-')[-1]) == 1)
                print(Y, end=' ')
                data = {"flow_x": to_tensor(X), "graph": to_tensor(A), "flow_y": Y}
                train_data.append(data)
                end_time = datetime.datetime.now()
                time = str((end_time - start_time).seconds)
                print('with ' + time + ' seconds, ' + 'finished in ' + str(datetime.datetime.now()).split('.')[0])

        num = 0
        for name in test_names:
            img_path = bag_path + name + '/'
            for folder in os.listdir(img_path):
                num += 1
                start_time = datetime.datetime.now()
                print('The ', str(num), 'st Test data named ', folder, end=' ')
                featureList = []
                imgList = []
                fold_path = img_path + folder + '/'
                # for f in os.listdir(fold_path):
                for j, f in enumerate(os.listdir(fold_path)):
                    fp = open(fold_path + f, 'rb')
                    img = Image.open(fp)
                    # image = trans_ops(img).view(-1, 3, 224, 224).to(device)
                    image = trans_ops(img).view(-1, 3, 224, 224)
                    fp.close()
                    imgList.append(image)
                    if (j + 1) % 20 == 0:
                        IN = torch.cat(imgList, 0).to(device)  # [N, 3, 244, 244]
                        # print('IN:', IN.shape)
                        OUT = densenet(IN).detach().cpu()
                        # print('OUT:', OUT.shape)
                        featureList.append(OUT)
                        imgList = []
                    # output = densenet(image).detach().cpu()
                    # featureList.append(output)
                    del img
                    del image

                X = torch.cat(featureList, 0)  # [N, 512]
                # print('FEA', X.shape)
                A = cross_correlation(X)
                # print('A', A.shape)
                # Y = torch.zeros([1, 2], dtype=torch.float)  # [1, 2] 无毒，有毒
                # Y[0][int(folder.split('_')[-1])] = 1
                # Y = (int(folder.split('_')[-1]) == 1)
                Y = (int(folder.split('-')[-1]) == 1)
                print(Y, end=' ')
                data = {"flow_x": to_tensor(X), "graph": to_tensor(A), "flow_y": Y}
                test_data.append(data)
                if folder.split('-')[0].split('+')[-1] == 'A':
                    print(folder)
                    sigtest_data.append(data)
                end_time = datetime.datetime.now()
                time = str((end_time - start_time).seconds)
                print('with ' + time + ' seconds, ' + 'finished in ' + str(datetime.datetime.now()).split('.')[0])
        data_path = "./Feature/" + str(i + 1) + "-fold/"
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        torch.save(train_data, data_path + "da-traindata-" + str(i + 1) + ".pth",
                   _use_new_zipfile_serialization=False)
        torch.save(test_data, data_path + "da-testdata-" + str(i + 1) + ".pth",
                   _use_new_zipfile_serialization=False)
        torch.save(sigtest_data, data_path + "sig-testdata-" + str(i + 1) + ".pth",
                   _use_new_zipfile_serialization=False)