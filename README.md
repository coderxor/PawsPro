## Overview of the experiment
Paws Proactive (PawsPro)a two-stage framework that combines time-series modeling with reinforcement learning 
 jointly enhance failure prediction and maintenance decision-making. It centers on the idea that.
1. Hard disk state migration: Based on the hard disk sequence data, the state change of the hard disk is studied.
2. Sequence Decision Making: Using reinforcement learning to build decision models to take different measures 
in different states of the hard disk.

The dataset used in this study is publicly available from Backblaze: https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data


## Project structure 
dataset/  # Experimental data set

├── data1_train_data.npy
├── data1_train_label.npy
├── data1_test_data.npy
├── data1_test_label.npy
├── data2_train_data.npy
├── data2_train_label.npy
├── data2_test_data.npy
└── data2_test_label.npy

GMMHMM/ # GMMHMM model implementation
├── trainHMM.py # Training the GMMHMM model
└── network.py # Network model

RL/ # DQN approach implementation
├── loadData.py # Load data
├── utils.py # Auxiliary functions, including creating catalogs and plotting convergence curves
├── env.py # Reinforcement Learning Environment
├── buffer.py # Experience replay pool
├── DQN.py # deep Q network implement
├── train.py # Train agents
└── test.py # Test

figure/ # Plotting Graphics
├── PCA_raw.py # Principal component analysis of raw labels
├── PCA_decoded.py # Principal component analysis after HMM classification
├── trans.py # Plotting the state transfer matrix
├── reward_function.py # Plotting the reward function
└── rewardConvergence.py # Plotting convergence curves for reward values
