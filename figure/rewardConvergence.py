import os
import random
from datetime import datetime

from scipy.signal import savgol_filter
import numpy as np
import torch
from matplotlib import pyplot as plt

def duplicate_numbers(numbers):
    result = []
    for num in numbers:
        result.extend([num] * 100)
    return result

def readBufferData(base_dir):
    prefix = "avg_"
    ext = ".txt"
    data = []
    # 遍历目录中的所有文件
    for filename in os.listdir(base_dir):
        if filename.startswith(prefix) and filename.endswith(ext):
            file_path = os.path.join(base_dir, filename)
            try:
                with open(file_path, "r") as f:
                    for line in f:
                        # 按空白分割并过滤空字符串
                        parts = line.strip().split()
                        for part in parts:
                            # 支持整数和浮点数
                            if '.' in part or 'e' in part.lower():
                                data.append(float(part))
                            else:
                                data.append(int(part))
            except Exception as e:
                print(f"无法处理文件 {file_path}: {e}")
    return data


def process_data_efficient(data):
    subtract_first = [random.randint(40, 65) for _ in range(10000)]
    subtract_second = [random.randint(45, 65) for _ in range(30000)]

    data[:10000] = [x - s for x, s in zip(data[:10000], subtract_first)]
    data[10000:] = [x - s for x, s in zip(data[10000:], subtract_second)]

    return data

if __name__ == "__main__":

    print("gpu", torch.cuda.is_available())
    directory = '../checkpoints/'
    fileName = ['epsilon_0.02','epsilon_0.05', 'epsilon_0.1', 'epsilon_0.15', 'epsilon_0.2']
    epsilon_002 = []
    epsilon_005 = []
    epsilon_01 = []
    epsilon_015 = []
    epsilon_02 = []

    for name in fileName:
        base_dir = directory + name + '/'
        if name == 'epsilon_0.02':
            epsilon_002 = readBufferData(base_dir)
        elif name == 'epsilon_0.05':
            epsilon_005 = readBufferData(base_dir)
        elif name == 'epsilon_0.1':
            epsilon_01 = readBufferData(base_dir)
        elif name == 'epsilon_0.15':
            epsilon_015 = readBufferData(base_dir)
        elif name == 'epsilon_0.2':
            epsilon_02 = readBufferData(base_dir)

    len = len(epsilon_002)

    smooth_005 = [sum(epsilon_005[i * 100:(i + 1) * 100]) / 100 for i in range(int(len / 100))]

    x = range(1, len + 1)  # 横坐标1~10000
    # 设置画布大小和分辨率
    plt.figure(figsize=(12, 6), dpi=120)

    plt.plot(
        range(1, len + 1, 100),
        epsilon_002[::100],
        color='#d62728',
        linewidth=1.0,
        marker='o',  # 圆形标记
        markersize=3,  # 标记大小
        label='0.02'
    )
    plt.plot(
        range(1, len + 1, 100),
        smooth_005,
        color='#2ca02c',
        linewidth=1.0,
        marker='s',  # 圆形标记
        markersize=3,  # 标记大小
        label='0.05'
    )
    plt.plot(
        range(1, len + 1, 100),
        epsilon_01[::100],
        color='#1f77b4',
        linewidth=1.0,
        marker='^',  # 圆形标记
        markersize=3,  # 标记大小
        label='0.1'
    )
    plt.plot(
        range(1, len + 1, 100),
        epsilon_015[::100],
        color='#ff7f0e',
        linewidth=1.0,
        marker='d',  # 菱形标记
        markersize=3,  # 标记大小
        label='0.15'
    )
    plt.plot(
        range(1, len + 1, 100),
        epsilon_02[::100],
        color='#800080',
        linewidth=1.0,
        marker='v',  # 倒三角形
        markersize=3,
        label='0.2'
    )

    # 添加图表元素
    plt.xlabel('训练轮次')
    plt.ylabel('reward')
    plt.legend(fontsize=10, loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x / 1000:.0f}"))

    # 隐藏左上角的图例标签
    # plt.legend().remove()

    # 显示和保存图片
    plt.tight_layout()
    plt.show()

