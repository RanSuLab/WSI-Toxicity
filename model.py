import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLP(nn.Module):
    def __init__(self, input_size, common_size):
        super(MLP, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            # nn.BatchNorm1d(input_size // 2, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.3),
            # nn.Linear(input_size // 2, common_size)
            nn.Linear(input_size // 2, input_size // 4),
            # nn.BatchNorm1d(input_size // 4, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.3),
            nn.Linear(input_size // 4, common_size)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear(x) # [B, 1]
        # out = self.sigmoid(out) # [B, 1]
        log_out = F.log_softmax(out, dim=1)  # [B, 2]
        no_out = F.softmax(out, dim=1)
        # out = F.softmax(out,dim = 1).unsqueeze(1) # [B, 1, 2]
        return log_out, no_out
        # return F.log_softmax(out, dim=1)


class GCNbatchAtten(nn.Module):
    def __init__(self, nfeat, nhid_1, nhid_2, nclass, dropout):  # 底层节点的参数，feature的个数；隐层节点个数；最终的分类数
        super(GCNbatchAtten, self).__init__()  # super()._init_()在利用父类里的对象构造函数

        self.L = 512
        self.D = 128
        self.K = 1
        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(self.D, self.K)

        self.gc1 = GraphConvolution(nfeat, nhid_1)  # gc1输入尺寸nfeat，输出尺寸nhid
        self.gc2 = GraphConvolution(nhid_1, nclass)  # gc2输入尺寸nhid，输出尺寸ncalss
        # self.gc1 = GraphConvolution(nfeat, nhid_1)  # gc1输入尺寸nfeat，输出尺寸nhid_1
        # self.gc2 = GraphConvolution(nhid_1, nhid_2)  # gc2输入尺寸nhid_1，输出尺寸nhid_2
        # self.gc3 = GraphConvolution(nhid_2, nclass)  # gc2输入尺寸nhid_2，输出尺寸ncalss
        # self.dropout = dropout
        self.classifier = MLP(nclass, 2) # MLP层
        # self.bn = nn.BatchNorm1d(300, affine=True, track_running_stats=True)

        self.criterion = nn.NLLLoss()  # softmax


    # 输入分别是特征和邻接矩阵。最后输出为输出层做log_softmax变换的结果
    def forward(self, x, adj):

        A_V = self.attention_V(x)  # BxNxD
        A_U = self.attention_U(x)  # BxNxD
        A = self.attention_weights(A_V * A_U)  # element wise multiplication # BxNxK
        A = F.softmax(A, dim=1)  # softmax over N
        T = torch.transpose(A, 1, 2)  # BxKxN
        M = torch.matmul(T, x)  # BxKxL
        # M = torch.squeeze(M)  # BxL
        # M = M.squeeze(1)
        # print("M:", M.shape)
        x = x + M

        x = F.relu(self.gc1(x, adj))  # adj即公式Z=softmax(A~Relu(A~XW(0))W(1))中的A~
        # x = F.dropout(x, self.dropout)  # x要dropout
        x = self.gc2(x, adj)  # [B, N, 2]
        # x = F.relu(self.gc2(x, adj))  # [B, N, 2]
        # x = F.dropout(x, self.dropout)  # x要dropout
        # x = self.gc3(x, adj)  # [B, N, 2]
        A = torch.sum(x, dim=1) / x.shape[1]  # [B, 2]

        Y_prob, Y_score = self.classifier(A)  # Bx2 prob_log
        # Y_hat = Y_prob.max(1)[1].float()

        return Y_prob, Y_score, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y, adj):
        Y = Y.float()
        Y_prob, Y_score, _ = self.forward(X, adj)
        # Y_score = torch.max(Y_soft, 1)[0]
        # Y_hat = Y_score.max(1)[1].float()
        Y_hat = Y_prob.max(1)[1].float()
        # right = Y_hat.eq(Y).sum().item()
        temp = Y_hat.eq(Y).float()
        right = temp.mean().item()

        return right, Y_score

    def calculate_objective(self, X, Y, adj):
        # Y = Y.float()
        Y = Y.long()
        Y_prob, _, A = self.forward(X, adj)
        loss = self.criterion(Y_prob, Y)

        return loss, A