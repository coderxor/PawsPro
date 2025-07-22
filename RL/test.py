import os
import argparse
import torch
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, f1_score, recall_score
from tqdm import tqdm
from env import Env
from DQN import DQN

parser = argparse.ArgumentParser()

parser.add_argument('--ckpt_dir', type=str, default='../checkpoints/buffersize_10000000_DQN/')
args = parser.parse_args()

def main():
    env = Env()
    datas, real_labels = env.load_data()

    # agent 加载已经训练好的模型
    agent = DQN(alpha=0.0003, state_dim=env.observation_space.shape[1], action_dim=len(env.action_space),
                fc1_dim=256, fc2_dim=256, ckpt_dir=args.ckpt_dir, gamma=0.99, tau=0.005, epsilon=1,
                eps_end=0.05, eps_dec=5e-4, max_size=1000, batch_size=256)

    agent.load_models(9000)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pre_labels = []

    for i in tqdm(range(len(datas))):
        x = datas[i]
        action = agent.choose_action(x, isTrain=False)
        if action == 0:
            pre_label = 0
        else:
            pre_label = 1
        pre_labels.append(pre_label)

    real_labels = [int(x) for x in real_labels]
    pre_labels = [int(x) for x in pre_labels]
    confusion_ = confusion_matrix(real_labels, pre_labels)
    precision_ = precision_score(real_labels, pre_labels, pos_label=0)
    accuracy_ = accuracy_score(real_labels, pre_labels)
    f1_ = f1_score(real_labels, pre_labels, pos_label=0)
    recall_ = recall_score(real_labels, pre_labels, pos_label=0)

    # 将结果写入txt文件中
    with open(os.path.join(args.ckpt_dir, 'test_result.txt'), 'w') as f:
        f.write('confusion_matrix:\n')
        f.write(str(confusion_) + '\n')
        f.write('precision_score:\n')
        f.write(str(precision_) + '\n')
        f.write('accuracy_score:\n')
        f.write(str(accuracy_) + '\n')
        f.write('f1_score:\n')
        f.write(str(f1_) + '\n')
        f.write('recall_score:\n')
        f.write(str(recall_) + '\n')


if __name__ == '__main__':
    main()