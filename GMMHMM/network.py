import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

N = 15  # 状态数
M = 18  # 一个序列的样本数
K = 5  # 高斯分量数
D = 36  # 降维之后的特征数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()

        layers = []
        layers += [nn.Linear(540, 512)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.Dropout(p=.3)]
        layers += [nn.Linear(512, 256)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.Dropout(p=.3)]
        layers += [nn.Linear(256, 128)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.Dropout(p=.3)]
        layers += [nn.Linear(128, 64)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.Dropout(p=.3)]
        layers += [nn.Linear(64, D - 2)]

        self.encoder = nn.Sequential(*layers).to(device)

        layers = []
        layers += [nn.Linear(D - 2, 64)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.Dropout(p=.3)]
        layers += [nn.Linear(64, 128)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.Dropout(p=.3)]
        layers += [nn.Linear(128, 256)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.Dropout(p=.3)]
        layers += [nn.Linear(256, 512)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.Dropout(p=.3)]
        layers += [nn.Linear(512, 540)]


        self.decoder = nn.Sequential(*layers).to(device)

        layers = []
        layers += [nn.Conv1d(D, N*K, 3, 1, 1, bias=False)]
        # layers += [nn.Linear(64, N * K)]
        layers += [nn.Softmax(dim=1)]
        self.estimate = nn.Sequential(*layers).to(device)

    def forward(self, x):
        encoded = self.encoder(x).to(device)
        decoded = self.decoder(encoded).to(device)
        loss = self.loss_function(x, decoded)
        rec_cosine = F.cosine_similarity(x, decoded, dim=2).unsqueeze(-1)  # batchsize*30*1
        rec_euclidean = self.relative_euclidean_distance(x, decoded).unsqueeze(-1)  # batchsize*30*1
        z = torch.cat([encoded, rec_euclidean, rec_cosine], dim=2)  # batchsize*M*D
        z = z.permute(0, 2, 1)  # batchsize*D*M
        mat = self.estimate(z)
        # 如果mat中有nan，则将其替换为0.0001
        mat[torch.isnan(mat)] = 0.0001
        return loss, z, mat

    def relative_euclidean_distance(self, a, b):
        return (a - b).norm(2, dim=2) / a.norm(2, dim=2)

    def compute_trans_matrix(self, sta):
        # sta=batchsize*M*N
        # 计算初始概率分布向量，以及状态转移矩阵
        sta = sta.to(device)
        trans_matrixs = []
        pis = []
        for i in range(sta.size(0)):
            pi = sta[i][0]
            trans_matrix = torch.zeros((N, N)).to(device)
            for j in range(len(sta[i]) - 1):
                cur_probs = sta[i][j]
                next_probs = sta[i][j + 1]
                for k in range(len(cur_probs)):
                    for l in range(len(next_probs)):
                        trans_matrix[k][l] += cur_probs[k] * next_probs[l]
            # 归一化
            pi = pi / pi.sum()
            trans_matrix /= trans_matrix.sum(dim=1, keepdim=True)

            pis.append(pi)  # batchsize*N
            trans_matrixs.append(trans_matrix)

        p = torch.stack(pis, dim=0)
        t = torch.stack(trans_matrixs, dim=0)
        pai = p.mean(dim=0)
        trans = t.mean(dim=0)
        trans = torch.where(torch.isnan(trans), 0.0001, trans)
        # 保证pai和trans矩阵行和为1
        pai = pai / pai.sum()
        trans /= trans.sum(dim=1, keepdim=True)

        return pai, trans

    def compute_gmm_params(self, gamma, z):
        # gamma维度为batchsize*M*N*K
        # z的维度为batchsize*M*D
        # 对第一、二个维度求和
        weight = gamma.mean(dim=(0, 1))  # N*K
        weight /= weight.sum(dim=1, keepdim=True)

        mu = torch.mean(gamma.unsqueeze(-1) * z.unsqueeze(2).unsqueeze(3), dim=(0, 1))
        # N*K*D

        # z_mu =  b x M x N x K x D
        z_mu = z.unsqueeze(-2).unsqueeze(-2) - mu.unsqueeze(0).unsqueeze(0)
        # z_mu_outer = b x M x N x K x D x D
        z_mu_outer = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)

        cov = torch.mean(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_outer, dim=(0, 1))
        diagonal = cov.diagonal(dim1=-2, dim2=-1)

        return weight, mu, diagonal

    def loss_function(self, x, x_hat):
        return torch.mean((x - x_hat) ** 2)
