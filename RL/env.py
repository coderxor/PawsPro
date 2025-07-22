import math
import numpy as np
from gym import spaces

class Env():
    def __init__(self):
        super(Env, self).__init__()
        self.action_space = ['k', 'r']  # 动作空间
        self.observation_space = spaces.Box(low=0, high=1, shape=(18, 30*18), dtype=np.float32)  # 状态空间

        self.xData = None
        self.yData = None
        self.currXData = None
        self.currYData = None

        self.index = 0
        self.rand = 0


    def load_data(self):
        directory = '/root/project/2019/'
        fileName = 'ST12000NM0007'
        npydata = directory + fileName + 'train' + 'data.npy'
        npylabel = directory + fileName + 'train' + 'label.npy'
        X = np.load(npydata)
        self.xData = X.reshape(14166, 30*18).reshape(787, 18, 30*18)
        Y = np.load(npylabel)
        self.yData = Y.reshape(787, 18)

    def reset(self):
        length = len(self.xData)
        # 随机选择一个数据
        rand = np.random.randint(0, length) # 0 <= rand < length
        self.rand = rand
        self.index = 0
        self.currXData = self.xData[self.rand][self.index]
        return self.currXData

    def step(self, action):
        done = False
        if action == 0:
            reward = - 0.375 * (self.index + 1) ** 2 + 0.75 * (self.index + 1) + 35
            self.index += 1
            if self.index < 18:
                self.currXData = self.xData[self.rand][self.index]
            else:
                done = True
        else:
            reward = 90 / (20 - (self.index + 1)) - 70
            done = True
        return self.currXData, reward, done, {}