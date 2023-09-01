import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from itertools import product


class Polynomial:
    def __init__(self, num_features, num_actions, alpha):
        self.num_features = num_features    # 다항식 차수
        self.num_actions = num_actions      # 행동 개수
        self.alpha = alpha                  # 학습률
        self.w = np.zeros([self.num_features + 1, self.num_actions])    # 가중치 벡터
        
    def get_features(self, state, action):
        # 다항식 Feature Vector 생성 (n차 다항식)
        features = [1]
        for _ in range(self.num_features):
            for s in state:
                features.append(s ** (_ + 1))
            features.append(action ** (_ + 1))
        return np.array(features)
    
    def predict_values(self, state):
        values = np.zeros(self.num_actions)
        for action in range(self.num_actions):
            features = self.get_features(state, action)
            values[action] = np.dot(self.w[:, action], features)
        return values
    
    def update_weights(self, state, action, target):
        q_values = self.predict_values(state)
        current_q = q_values[action]
        delta = target - current_q
        features = self.get_features(state, action)
        self.w[:, action] += self.alpha * delta * features


class FourierBasis:
    def __init__(self, env, fourier_order, alpha):
        self.n = fourier_order                      # Fourier order
        self.k = 3                                  # State Dimension (x, y)
        self.n_action = 6                           # number of actions
        self.alpha = alpha                          # base learning rate
        self.w = np.zeros([pow(self.n+1, self.k), self.n_action])
        self.env = env
        self.alpha_vec = self.setting_alphas()
        
    def setting_alphas(self):
        '''특징 벡터의 각 차원에 대해 상태 공간의 크기에 따라 다른 학습률을 설정'''
        '''학습률을 상태 공간의 크기로 나누는 방식으로 각 차원에 대한 학습률 조정'''
        orders = list(range(self.n + 1))            # orders {0, 1, ..., n}
        c = list(product(orders, repeat = self.k))  # (n+1)^k Cartesian Products
        norm_c = np.linalg.norm(c, axis = 1)        # (n+1)^k norms
        norm_c[0] = 1.
        return self.alpha / norm_c                  # shape ((n+1)^k)
    
    def normalize(self, state):
        '''상태를 정규화, 주어진 상태를 [0, 1] 범위로 스케일링'''
        norm_state = np.empty(np.shape(state))
        min_state = np.array([0, 0, 0])                         # Minimun Start Zone (0,0)
        max_state = np.array([self.env.size[0], self.env.size[1], self.env.size[2]]) # Maximum Size
        
        state_range = max_state - min_state
        norm_state = (state - min_state) / state_range
        return norm_state
    
    def get_features(self, state):
        '''입력받은 State를 Cosine 함수를 이용하여 특징 벡터로 변환'''
        norm_state = self.normalize(state)
        orders = list(range(self.n + 1))             # orders {0, 1, ..., n} 
        c = list(product(orders, repeat=self.k))     # (n+1)^k Cartesian products
        return np.cos(np.pi * np.dot(c, norm_state)) # (n+1)^k features
    
    def predict_values(self, state):
        '''Q-Value 예측 / θ^T * φ(s, a) 계산'''
        return np.dot(self.w.T, self.get_features(state))
    
    def update_weights(self, state, action, target):
        '''TD 오차를 기반으로 가중치 업데이트'''
        q_values = self.predict_values(state)
        current_q = q_values[action]
        delta = target - current_q
        features = self.get_features(state)
        self.w[:, action] += delta * self.alpha_vec * features


class TileCoding:
    def __init__(self, num_tilings, num_tiles, num_actions, alpha):
        self.num_tilings = num_tilings  # 타일링의 개수
        self.num_tiles = num_tiles      # 한 타일링의 타일 개수
        self.num_actions = num_actions  # 행동 개수
        self.alpha = alpha              # 학습률

        # 각 타일의 영역을 나타내는 offsets과 위치 조정을 위한 배율 scale 설정
        self.offsets = np.random.rand(self.num_tilings, 2) * np.array([1, 1])
        self.scale = np.array([1, 1]) / np.array([self.num_tiles, self.num_tiles])

        # 각 타일에 대한 가중치 벡터 초기화
        self.w = np.zeros([self.num_tilings, self.num_tiles, self.num_tiles, self.num_actions])

    def get_features(self, state, action):
        features = np.zeros([self.num_tilings, self.num_tiles, self.num_tiles, self.num_actions])
        for tiling_idx in range(self.num_tilings):
            # 상태와 offsets, scale을 이용하여 타일의 인덱스를 계산
            tile_idx = tuple(np.clip(np.floor((state + self.offsets[tiling_idx]) * self.scale), 0, self.num_tiles - 1).astype(int))
            features[tiling_idx, tile_idx[0], tile_idx[1], action] = 1
        return features.flatten()

    def predict_values(self, state):
        values = np.zeros(self.num_actions)
        for action in range(self.num_actions):
            features = self.get_features(state, action)
            values[action] = np.sum(self.w[:, :, :, action] * features)
        return values

    def update_weights(self, state, action, target):
        q_values = self.predict_values(state)
        current_q = q_values[action]
        delta = target - current_q
        features = self.get_features(state, action)
        self.w[:, :, :, action] += self.alpha * delta * features.reshape(self.num_tilings, self.num_tiles, self.num_tiles)


class NeuralNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)+