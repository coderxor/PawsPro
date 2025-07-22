import numpy as np
import torch
from hmmlearn import hmm
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterGrid
import joblib
from matplotlib.colors import to_rgba

if __name__ == "__main__":
    print("gpu", torch.cuda.is_available())

    # 读取npy文件
    npydata = '../data.npy'

    X = np.load(npydata)
    data_std = X.reshape(14166, 540)
    labels = np.load(npydata)

    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_std)

    # 使用tab20颜色模板创建基础颜色
    base_colors = plt.cm.get_cmap('tab20', 2)(np.linspace(0, 1, 2))

    # 多种墨绿色选项（选择其中一种）
    dark_green = '#32CD32'  # 深墨绿 (RGB: 26,79,39)

    base_colors[0] = to_rgba(dark_green)  # 索引0对应label=1

    red_color = np.array([[1, 0, 0, 1]])
    base_colors[1] = red_color
    custom_cmap = ListedColormap(base_colors)


    # 绘制结果并添加图例
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(data_pca[:, 0], data_pca[:, 1], s=30, c=labels, cmap=custom_cmap, alpha=0.7, edgecolor='k')
    plt.xlabel('PCA-1')
    plt.ylabel('PCA-2')

    # --- 手动创建完整图例 ---
    all_labels = np.arange(2)
    label_names = [f"{i}" for i in all_labels]
    # 创建颜色块句柄（确保颜色顺序与标签对应）
    handles = [
        plt.Line2D([], [],
                   marker='o',
                   markersize=5,
                   linestyle='',
                   color=custom_cmap(i),
                   label=label_names[i])
        for i in all_labels
    ]

    # 添加图例，并调整位置到图像右侧外部
    legend = plt.legend(handles=handles,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),  # 向右偏移 2%
        borderaxespad=0.1,
        ncol=1)  # 分两列显示)  # 将图例移到图像右侧外部 (x, y)

    # 调整图像右侧边距，给图例腾出空间
    # plt.subplots_adjust(right=0.8)  # 数值根据图例宽度调整

    plt.show()