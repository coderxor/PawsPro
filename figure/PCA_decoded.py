import numpy as np
import torch
from hmmlearn import hmm
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D  # 添加3D绘图支持
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterGrid
import joblib
from sklearn.cluster import KMeans
from matplotlib.colors import to_rgba
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    print("gpu", torch.cuda.is_available())

    npydata = '../data.npy'
    X = np.load(npydata)
    data_std = X.reshape(14166, 540)

    pca = PCA(n_components=2)  # 将n_components设置为2
    data_pca = pca.fit_transform(data_std)
    print(data_pca.shape)

    npylabel = 'saveModel/decoded_label.npy'
    labels = np.load(npylabel)
    print(labels.shape)

    n_classes = 15  # 类别数
    # 颜色配置
    base_colors = plt.cm.get_cmap('tab20', n_classes)(np.linspace(0, 1, n_classes))
    dark_green = '#1A4F27'  # 深墨绿 (RGB: 26,79,39)
    base_colors[0] = to_rgba(dark_green)
    forest_green = '#228B22'  # 森林绿 (RGB: 34,139,34)
    base_colors[1] = to_rgba(forest_green)
    olive_green = '#32CD32'  # 橄榄绿 (RGB: 85,107,47)
    base_colors[2] = to_rgba(olive_green)
    orange = '#FFA500'  # 橙色 (RGB: 255,165,0)
    base_colors[3] = to_rgba(orange)
    orange1 = '#CC8B3F'
    base_colors[4] = to_rgba(orange1)
    orange2 = '#B36B00'
    base_colors[5] = to_rgba(orange2)
    orange_red = '#FF9D6E'
    base_colors[7] = to_rgba(orange_red)
    red0 = '#FF0000'
    base_colors[8] = to_rgba(red0)
    red = '#FFB7C5'
    base_colors[6] = to_rgba(red)
    red1 = '#CD071E'
    base_colors[9] = to_rgba(red1)
    color13 = '#B28DB2'
    base_colors[13] = to_rgba(color13)
    color14 = '#800080'
    base_colors[14] = to_rgba(color14)
    custom_cmap = ListedColormap(base_colors)

    # 绘制结果
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(data_pca[:, 0], data_pca[:, 1], s=30, c=labels, cmap=custom_cmap, alpha=0.7, edgecolor='k')
    plt.xlabel('PCA-1')
    plt.ylabel('PCA-2')

    # --- 手动创建完整图例 ---
    all_labels = np.arange(n_classes)
    label_names = [f"{i}" for i in all_labels]  # 可自定义标签名称
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

    # 将图例定位到右侧外部
    legend = plt.legend(
        handles=handles,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),  # 向右偏移 2%
        borderaxespad=0.1,
        ncol=1  # 分两列显示
    )

    # 调整布局防止裁剪
    # plt.subplots_adjust(right=0.8)  # 右侧留出 20% 空白
    plt.show()