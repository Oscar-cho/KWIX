import numpy as np
import matplotlib.pyplot as plt
import noise
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm
import pickle
import pandas as pd

from ENV import *

def t(x): return torch.from_numpy(x).float()

class Actor(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, n_actions),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, X):
        return self.model(X)
    

class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, X):
        return self.model(X)
    
class Memory():
    def __init__(self):
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        
    def add(self, log_prob, value, reward, stop):
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(stop)
        
    def clear(self):
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear
        
    def _zip(self):
        return zip(self.log_probs, self.values, self.rewards, self.dones)
    
    def __iter__(self):
        for data in self._zip():
            yield data

    def reverse(self):
        for data in reversed(list(self._zip())):
            yield data
            
    def __len__(self):
        return len(self.rewards)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('your device type is ', device)
max_steps = 10000

env = FANET_ENV((100, 100, 30), 75, 20, max_steps=max_steps)    # 처음은 env size, 두번째는 target db, 3번째는 target level, 마지막은 max_step
state_dim = 3
n_actions = 6
memory = Memory()
lr = 1e-5
gamma = 0.999
actor = Actor(state_dim, n_actions)
critic = Critic(state_dim)
adam_actor = torch.optim.Adam(actor.parameters(), lr)
adam_critic = torch.optim.Adam(critic.parameters(), lr)
    
def train(memory, q_val):
    values = torch.stack(memory.values)
    q_vals = np.zeros((len(memory), 1))
    
    for i,  (_, _, reward, stop) in enumerate(memory.reverse()):
        q_val = reward + gamma * q_val * (1.0 - stop)
        q_vals[len(memory)-1 - i] = q_val
        
    advantage = torch.Tensor(q_vals) - values
    
    critic_loss = advantage.pow(2).mean()
    adam_critic.zero_grad()
    critic_loss.backward()
    adam_critic.step()
    
    adam_loss = (-torch.stack(memory.log_probs) * advantage.detach()).mean()
    adam_actor.zero_grad()
    adam_loss.backward()
    adam_actor.step()
        
num_episodes = 50000

step_count = [0] * num_episodes
path = [0] * num_episodes
conflict = [0] * num_episodes
total_reward = [0] * num_episodes

for episode in tqdm(range(num_episodes)):
    state = env.reset()
    
    episodic_step = 0
    episodic_path = [state]
    episodic_conflict = [False]
    episodic_reward = 0
    stop = False
    
    while not stop:
        probs = actor(t(np.array(state))) 
        dist = torch.distributions.Categorical(probs = probs)
        action = dist.sample()
        
        next_state, reward, stop = env.step(action.detach().data.numpy())
        
        episodic_step += 1
        episodic_reward += reward
        episodic_path.append(next_state)
        episodic_conflict.append(reward == -500)
        
        memory.add(dist.log_prob(action), critic(t(np.array(state))), reward, stop)
        state = next_state
        
        
        if stop or (episode+1 % max_steps == 0):
            last_q_val = critic(t(np.array(next_state))).detach().data.numpy()
            train(memory, last_q_val)
            memory.clear()
    
    print(f'Episode {episode + 1} / {num_episodes}, present_state : {reward == 1000, (state[0], state[1], state[2])}, n_collision : {episodic_conflict.count(True)}, Total Reward : {episodic_reward}, Step Counts : {episodic_step}')
    step_count[episode] = episodic_step
    path[episode] = episodic_path
    conflict[episode] = episodic_conflict
    total_reward[episode] = episodic_reward
    
learning_data = {
    'Step Counts' : step_count,
    'Total Reward' : total_reward
}
path_data = {
    'Path' : path,
    'Conflict' : conflict
}

df = pd.DataFrame(learning_data)
df.to_csv('./KKH/A2C/A2C_data_finish.csv', index=True)

with open('./KKH/A2C/A2C_data_finish.pkl', 'wb') as f:
    pickle.dump(path_data, f)

