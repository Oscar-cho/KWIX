import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import matplotlib
import matplotlib.pyplot as plt
from random import random,randint
from itertools import product
from tqdm import tqdm
from torch.autograd import Variable
from collections import deque
from collections import namedtuple
from itertools import count
from mpl_toolkits.mplot3d import axes3d


class FANET:
    def __init__(self): # size = tuple(int, int, int)
        self.size = (100, 100, 100)
        self.start_point = (10, 10, 1)
        self.terminal_zone = (self.size[0] - 10, self.size[1] - 10, self.size[2] - 10)
        self.reset()
        self.hazard()
        
    def hazard(self):
        self.hazard_map = np.zeros(self.size, dtype=bool)
        hazard_zones = {
            'a': {'start': (40, 40, 0), 'end': (50, 50, 20)},
            'b': {'start': (60, 60, 30), 'end': (60, 60, 60)},
            'c': {'start': (80, 80, 80), 'end': (85, 85, 90)}
        }

        for zone in hazard_zones.values():
            start = zone['start']
            end = zone['end']
            self.hazard_map[start[0]:end[0]+1, start[1]:end[1]+1, start[2]:end[2]+1] = True

    def check_hazard_zone(self):
        check = False
        if self.hazard_map[self.state[0], self.state[1], self.state[2]] == True:
            check = True

        return check

    def reset(self):
        self.state = self.start_point
    
    def terminal_state(self):
        return (self.state[0] >= self.terminal_zone[0] and
                self.state[1] >= self.terminal_zone[1] and
                self.state[2] >= self.terminal_zone[2])

    def reward(self):
        reward = -1
        
        if self.terminal_state():
            reward = 100
        
        if self.check_hazard_zone(): # if True
            reward = -10

        return reward

    def step(self, action):
        x, y, z = self.state
    
        if action == 0 and x < self.size[0] - 1:
            x += 1
        elif action == 1 and x > 0:
            x -= 1
        elif action == 2 and y < self.size[1] - 1:
            y += 1
        elif action == 3 and y > 0:
            y -= 1
        elif action == 4 and z < self.size[2] - 1:
            z += 1
        elif action == 5 and z > 0:
            z -= 1

        self.state = (x, y, z)
        reward = self.reward()
        
        return self.state, reward, self.terminal_state()



'''
matplotlib 설정
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
    
plt.ion()
'''

#cuda 쓸지 아니면 그냥 cpu 쓸지 설정해야 함  

''''
GPU 사용시
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
PATH = 'model.'
torch.save()
'''

env = FANET()
Transition = namedtuple('Transition', ('state', 'action', 'next_state','reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        
    def push(self, *args):
        '''transition을 저장'''
        self.memory.append(Transition(*args))
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    ''' optimization 주에 다른 행동을 결정하기 위해 하나의 요소 or batch를 이용해 호출'''
    
    def forward(self,state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# batch_size -> replay에서 sampling된 transition 수
# eps start -> epsilon 초기값
# eps_end -> epsilon 최종값
# tau -> network update rate
# LR -> AdamW 의 learning rate

BATCH_SIZE = 120
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

n_actions = 6
state, info = env.reset()        
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.parameters(), lr = LR, amsgrad = True)

optimizer = optim.AdamW(policy_net.parameters(), lr = LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) *\
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:          # 기댓값이 더 큰 action을 선택
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device = device, dtype=torch.long)
    
''' plot은 3d로 나중에 할 것

def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
'''

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    
    # batch array의 transition을 transition의 batch array로 전환
    batch = Transition(*zip(*transitions))
    
    non_final_mask = torch.tensor(tuple(map(lamda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    
    # Q(s_t, a) 
