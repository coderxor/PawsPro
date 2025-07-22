import torch
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

if __name__ == "__main__":

    print("gpu", torch.cuda.is_available())

    dir = 'saveModel/hmm.pth'

    # 加载模型
    model = torch.load(dir)
    data = model.transmat_

    # # 对data数据进行处理，保证行和为1
    # data = np.array(data)
    # data = data / np.sum(data, axis=1)[:, np.newaxis]

    # 绘制热力图，设置颜色映射为反转的灰度映射
    plt.imshow(data, cmap='gray_r', interpolation='nearest', norm=Normalize(vmin=0, vmax=1))

    # 添加颜色条
    plt.colorbar()
    # 设置标题
    plt.title('states=15')
    # 将x轴刻度标签移动到顶部
    plt.gca().xaxis.tick_top()
    # 将x轴刻度标签置于顶部刻度线下方
    plt.gca().xaxis.set_label_position('top')
    # 显示图形
    plt.show()



