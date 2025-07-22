import numpy as np
import torch
import matplotlib.pyplot as plt

# 定义保留曲线函数
# 定义第一个函数
def func1(x):
    return 0.1 * (-x ** 2 + 2 * x + 255)


def func3(x):
    return -40 * (x - 17)
# 定义第二个函数
def func2(x):
    return  (np.exp(x)  - np.exp(17)) / 1000000


def func4(x):
    return 60 * (x - 17)

if __name__ == "__main__":
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Zen Hei']
    plt.rcParams['axes.unicode_minus'] = False  # 修复负号

    print("gpu", torch.cuda.is_available())

    # 生成x轴数据
    x = np.linspace(1, 17, 100)

    # 计算对应的y轴数据
    y1 = func1(x)
    y2 = func2(x)

    x3 = np.linspace(17, 18, 100)
    y3 = func3(x3)
    y4 = func4(x3)

    # 找到交叉点的x坐标
    # diff = np.abs(y1 - y2)
    # min_index = np.argmin(diff)
    cross_x = 17

    plt.figure(figsize=(10, 6))

    # 绘制图形
    plt.plot(x, y1, label='保持', color='b')
    plt.plot(x, y2, label='更换', color='r')
    plt.plot(x3, y3, color='b')
    plt.plot(x3, y4, color='r')
    plt.axvline(x=cross_x, color='y', linestyle='--')

    # 添加标题和坐标轴标签
    plt.xlabel('对样本进行操作的时间步', fontsize=14)
    plt.ylabel('反馈值', fontsize=14)

    custom_ticks = [1, 2, 3, 17, 18]
    custom_labels = ['1', '2', '3', 'T-1', 'T']

    plt.xticks(ticks=custom_ticks, labels=custom_labels, fontsize=14)

    # 设置x轴刻度为整数
    # plt.xticks(np.arange(1, 19, 1))
    plt.tick_params(axis='both', labelsize=14)

    # 添加省略号标注
    plt.annotate('......',
                 xy=(9, -0.01),  # 文本位置坐标
                 xycoords=('data', 'axes fraction'),  # 坐标系类型
                 ha='center', va='top',  # 对齐方式
                 fontsize=16,
                 color='black')

    # 显示图例
    plt.legend(fontsize=14)

    # 显示图形
    plt.show()