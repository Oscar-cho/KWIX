import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import namedtuple, deque
from ENV import *

class DQN(nn.Module):
    def __init__(self, n_state, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_state, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, 64)
        self.layer5 = nn.Linear(64, n_actions)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        return F.relu(self.layer5(x))

class DQNAgent:
    def __init__(self, env, alpha, gamma, batch_size, replay_buffer_size, target_update_interval, num_episodes):
        self.env = env
        self.alpha = alpha  # learning rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_buffer_size = replay_buffer_size
        self.target_update_interval = target_update_interval
        self.num_episodes = num_episodes
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_state = self.env.size[0] + self.env.size[1] + self.env.size[2]
        self.n_actions = 6
        
        self.policy_model = DQN(self.n_state, self.n_actions).to(self.device)
        self.target_model = DQN(self.n_state, self.n_actions).to(self.device)
        self.target_model.load_state_dict(self.policy_model.state_dict())
        self.target_model.eval()
        
        self.Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'stop'))
        self.replay_buffer = deque(maxlen=self.replay_buffer_size)
        
        self.optimizer = optim.Adam(self.policy_model.parameters(), lr=self.alpha)
        self.loss_fn = nn.MSELoss()
        
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995
        self.epsilon = 0.1
        
        print('Your Device Type is', self.device)
        
    def encoding_state(self, state):
        x, y, z = state
        x_onehot, y_onehot, z_onehot = np.zeros(self.env.size[0]), np.zeros(self.env.size[1]), np.zeros(self.env.size[2])
        
        x_onehot[x] = 1
        y_onehot[y] = 1
        z_onehot[z] = 1
        
        return np.concatenate((x_onehot, y_onehot, z_onehot))
    
    def decoding_state(self, encoded_state):
        x_size, y_size, z_size = self.env.size
        
        x_encoded = encoded_state[:x_size]
        y_encoded = encoded_state[x_size : x_size + y_size]
        z_encoded = encoded_state[x_size + y_size : x_size + y_size + z_size]
        
        x = np.argmax(x_encoded)
        y = np.argmax(y_encoded)
        z = np.argmax(z_encoded)
        
        return (x, y, z)
    
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            with torch.no_grad():
                q_values = self.policy_model(torch.tensor(state, dtype=torch.float32).to(self.device))
                return torch.argmax(q_values).item()
    
    def optimize_model(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        batch = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in batch]
        batch_state, batch_action, batch_next_state, batch_reward, batch_stop = zip(*batch)
        
        batch_state = torch.tensor(np.array(batch_state), dtype=torch.float32).to(self.device)
        batch_action = torch.tensor(np.array(batch_action), dtype=torch.long).unsqueeze(1).to(self.device)
        batch_next_state = torch.tensor(np.array(batch_next_state), dtype=torch.float32).to(self.device)
        batch_reward = torch.tensor(np.array(batch_reward), dtype=torch.float32).unsqueeze(1).to(self.device)
        batch_stop = torch.tensor(np.array(batch_stop), dtype=torch.float32).unsqueeze(1).to(self.device)
        
        current_q_values = self.policy_model(batch_state).gather(1, batch_action)
        with torch.no_grad():
            next_q_values = self.target_model(batch_next_state).max(1)[0].unsqueeze(1)
        target_q_values = batch_reward + (1 - batch_stop) * self.gamma * next_q_values
        
        loss = self.loss_fn(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def learning(self):
        self.step_count = [0] * self.num_episodes
        self.path = [0] * self.num_episodes
        self.total_reward = [0] * self.num_episodes
        print('Learning Start ... !')
        
        for episode in range(self.num_episodes):
            state = self.env.reset()
            
            # one-hot encoding
            state = self.encoding_state(state)
            decoded_state = self.decoding_state(state)
            stop = False
            
            episodic_step = 0
            episodic_path = [(decoded_state, False)]
            episodic_reward = 0
            
            while not stop:
                action = self.get_action(state)
                next_state, reward, stop = self.env.step(action)
                
                next_state = self.encoding_state(next_state)
                decoded_next_state = self.decoding_state(next_state)
                
                episodic_step += 1
                episodic_path.append((decoded_next_state, reward == self.env.terrain_conflict_reward))
                episodic_reward += reward
                
                transition = self.Transition(state, action, next_state, reward, stop)
                self.replay_buffer.append(transition)
                
                state = next_state
                
                self.optimize_model()
                
                if stop:
                    break                    
                
            print(f'Episode {episode + 1} / {self.num_episodes}, 목표위치 : {self.env.reference_path_loss}, Agent 마지막 위치 : {episodic_path[-1], self.env.loss_map[decoded_next_state[0], decoded_next_state[1], decoded_next_state[2]]}, reward : {episodic_reward}, step 횟수 : {episodic_step}')
            self.step_count[episode] = episodic_step
            self.path[episode] = episodic_path
            self.total_reward[episode] = episodic_reward
            
            if episode % self.target_update_interval == 0:
                self.target_model.load_state_dict(self.policy_model.state_dict())
            
            # Epsilon 값을 변경해서 학습을 계속 explore 하게 만들 것.
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
        print('Done!')
        return self.step_count, self.path, self.total_reward
    
env = FANET_ENV((100, 100, 100), 90)
num_episodes = int(input('총 학습 횟수 입력 : '))
#gamma = [0.9, 0.99, 0.999]
#learning_rate = [0.001, 0.0001, 0.00001]
gamma = [0.999]
learning_rate = [0.0001]

for i in gamma:
    for j in learning_rate:
        agent = DQNAgent(env=env, alpha=j, gamma=i, batch_size=64, replay_buffer_size=1000000, target_update_interval=100, num_episodes=num_episodes)
        step_count, path, total_reward = agent.learning()
        data_1 = {
            'episode' : list(range(1, num_episodes + 1)),
            'step_count' : step_count,
            'total_reward' : total_reward
        }
        
        data_2 = {
            'episode' : list(range(1, num_episodes + 1)),
            'path' : path
        }
        
        df = pd.DataFrame(data_1)
        df.to_csv(f'DQN_{i}_learningrate_{j}.csv', index=False)
        
        file_name = f'DQN_{i}_learningrate_{j}.pkl'
        with open(file_name, 'wb') as f:
            pickle.dump(data_2, f)