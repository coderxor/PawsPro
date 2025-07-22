import numpy as np
import torch


def loadData(directory, fileName):
    npydata = directory + fileName + 'train' + 'data.npy'
    npylabel = directory + fileName + 'train' + 'label.npy'

    X = np.load(npydata)
    print(X.shape)
    print(X[0])

    Y = np.load(npylabel)
    print(Y.shape)
    print(Y[0])

if __name__ == "__main__":
    print("gpu", torch.cuda.is_available())

    # 读取npy文件
    directory = '/root/project/2019/'
    fileName = 'ST12000NM0007'
    train = 'train'
    npydata = directory + fileName + train + 'data.npy'
    npylabel = directory + fileName + train + 'label.npy'

    X = np.load(npydata)
    print(X.shape)
    print(X[0])

    Y = np.load(npylabel)
    print(Y.shape)
    print(Y[0])