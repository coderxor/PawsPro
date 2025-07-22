import numpy as np
from tqdm import tqdm
from hmmlearn import hmm
import network
import torch
import os

N_STATES = 15
TerminalFlag = "terminal"

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
N = 15  # 状态数
M = 18  # 一个序列的样本数
K = 5  # 高斯分量数
D = 36  # 特征维度

def train():
    print("gpu", torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 读取npy文件
    directory = '/home/lijw/2019/'
    fileName = 'ST12000NM0007'
    train = 'train'
    npydata = directory + fileName + train + 'data.npy'

    X = np.load(npydata)
    X = X.reshape(787, 18, 540)
    print(X.shape)

    data = torch.from_numpy(X).float().to(device)

    L = data.shape[0]

    model = network.MyNetwork().to(device)


    epochs = 1000
    batchSize = 32  # 每次随机抽取batchsize序列数据进行训练
    losses = []
    for epoch in tqdm(range(epochs)):
        indices = np.random.choice(L, size=batchSize, replace=False)
        for i in range(50):
            sub_data = data[indices, :, :]
            loss, z, mat = model.forward(sub_data)
            mat = mat.permute(0, 2, 1)  # batchsize*D*M
            mat = mat.reshape(batchSize, M, N, K).to(device)  # batchSize*M*N*K
            z = z.permute(0, 2, 1).to(device)  # batchsize*M*D
            state_pro = mat.sum(dim=3).to(device)  # batchSize*M*N
            pi, trans = model.compute_trans_matrix(state_pro)
            weight, mu, cov = model.compute_gmm_params(mat, z)
            # 修正，保证协方差矩阵正定
            # cov = model.cholesky_correction(cov)
            hmmmodel = hmm.GMMHMM(
                n_components=N,
                n_mix=K,
                n_iter=100,
                covariance_type='diag',
                verbose=False,
                init_params="",
                implementation='log'
            )

            hmmmodel.n_features = D
            hmmmodel.startprob_ = pi.detach().cpu().numpy()
            hmmmodel.transmat_ = trans.detach().cpu().numpy()
            hmmmodel.weights_ = weight.detach().cpu().numpy()
            hmmmodel.means_ = mu.detach().cpu().numpy()
            hmmmodel.covars_ = cov.detach().cpu().numpy()


            z = z.reshape(batchSize * M, D)
            lengths = np.full(batchSize, M)
            obs_pro = -1 * hmmmodel.score(z.detach().cpu().numpy(), lengths)
            # 将模型对数据的拟合程度作为loss
            loss += obs_pro / batchSize  # 每个序列的平均负对数似然
            losses.append(loss)

            optimizer = torch.optim.Adam(model.parameters(), lr=0.3)
            model.zero_grad()
            loss.backward()
            optimizer.step()

        pth = 'saveModel/hmmmodel.pth'
        torch.save(hmmmodel, pth)


if __name__ == '__main__':
    train()
