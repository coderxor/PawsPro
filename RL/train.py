import os
import gym
import numpy as np
import argparse
from env import Env
from DQN import DQN
from utils import plot_learning_curve, create_directory

parser = argparse.ArgumentParser()
parser.add_argument('--max_episodes', type=int, default=10000)
parser.add_argument('--ckpt_dir', type=str, default='/root/project/2019/ST12000NM0007/checkpoints/buffersize_1000_DQN/')
parser.add_argument('--reward_path', type=str, default='/root/project/2019/ST12000NM0007/output_images/avg_reward.png')
parser.add_argument('--epsilon_path', type=str, default='/root/project/2019/ST12000NM0007/output_images/epsilon.png')

args = parser.parse_args()

def main():
    env = Env()
    env.load_data()
    agent = DQN(alpha=0.0003, state_dim=env.observation_space.shape[1], action_dim=len(env.action_space),
                fc1_dim=256, fc2_dim=256, ckpt_dir=args.ckpt_dir, gamma=0.99, tau=0.005, epsilon=1,
                eps_end=0.05, eps_dec=5e-4, max_size=1000, batch_size=256)
    create_directory(args.ckpt_dir, sub_dirs=['Q_eval', 'Q_target'])
    total_rewards, avg_rewards, eps_history = [], [], []

    for episode in range(args.max_episodes):
        total_reward = 0
        done = False
        observation = env.reset()
        step = 0
        while not done:
            step += 1
            action = agent.choose_action(observation, isTrain=True)
            observation_, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            total_reward += reward
            observation = observation_

        total_rewards.append(total_reward)
        avg_reward = np.mean(total_rewards[-100:])
        avg_rewards.append(avg_reward)
        eps_history.append(agent.epsilon)
        print('EP:{} reward:{} avg_reward:{} epsilon:{}'.
              format(episode + 1, total_reward, avg_reward, agent.epsilon))

        if (episode + 1) % 50 == 0:
            agent.save_models(episode + 1)

    episodes = [i for i in range(args.max_episodes)]
    # 保存episodes, avg_rewards到txt文件
    np.savetxt(os.path.join(args.ckpt_dir, 'buffersize_1000_avg_rewards.txt'), avg_rewards, fmt='%d')

    plot_learning_curve(episodes, avg_rewards, 'Reward', 'reward', args.reward_path)
    plot_learning_curve(episodes, eps_history, 'Epsilon', 'epsilon', args.epsilon_path)


if __name__ == '__main__':
    main()