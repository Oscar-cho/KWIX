# Hazard 있는 곳 지나치는 자율주행 차(2차원)

import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import datetime

# 신경망 class

class Network(nn.Module):                      # nn.Module 상속 (nn은 Neural Network의 모든 걸 포함하는 신경망 모델의 base class)
    
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()        # init method 내부에서 상속 시키기 위해 super() 함수 사용
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30)      # input layer와 hidden layer 사이 첫번째 완전 연결 계층인 self.fc1 생성 / nn.Linear class의 객체로 생성
        self.fc2 = nn.Linear(30, nb_action)       # hidden layer(30개의 은닉된 뉴런으로 구성된) 와 output layer 사이 두번째 완전 연결 계층 self.fc2 생성 / linear class의 객체로 생성됨. ouput 계층은 2개의 인수를 받음
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values

# memory 객체를 갖는 다른 class(경험 재현 구현)

class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)

# DQN 구현

class Dqn(object):
    
    def __init__(self, input_size, nb_action, gamma):   # input_size: 입력 상태 vector의 입력 개수(4), nb_action:할 수 있는 행동의 갯수
        self.gamma = gamma
        self.model = Network(input_size, nb_action)     # network class의 객체 (신경망) 이용
        self.memory = ReplayMemory(capacity = 100000)   # ReplayMemory class 객체 이용.(경험 재현 메모리). 100000-> 크기 10만인 메모리 생성하며 최신 전이만 기억하고 나머지는 AI가 기억하게 함
        self.optimizer = optim.Adam(params = self.model.parameters()) # Adam 이용. mini-batch GD를 통해 가중치를 update하는 최적화 알고리즘/ nn.Module class의 self.module.parameter를 사용해 접근 가능
        self.last_state = torch.Tensor(input_size).unsqueeze(0)  # 각 전이의 최신 상태.torch library의 tensor class의 객체로 초기화 됨(input_size 인수 입력을 통해 변경 가능) / unsqueeze(0) -> batch에 해당되는 추가적인 차원을 생성시 사용
        self.last_action = 0                                     # 
        self.last_reward = 0
    
    def select_action(self, state):                                     # softmax를 활용해 반복할때마다 수행할 행동 선택
        probs = F.softmax(self.model(Variable(state))*100)
        action = probs.multinomial(len(probs))
        return action.data[0,0]
    
    def learn(self, batch_states, batch_actions, batch_rewards, batch_next_states):      # 시간차를 계산하고 손실에 따라 최적화 알고리즘으로 손실을 줄이기 위해 가중치 update
        batch_outputs = self.model(batch_states).gather(1, batch_actions.unsqueeze(1)).squeeze(1)  # Q(StB +1, a) 수집 -> batch에서 입력 상태와 수행할 행동에 대해 예측한 Q값.
        batch_next_outputs = self.model(batch_next_states).detach().max(1)[0]                      # target의 argmaxa(Q(StB+1, a)) 계산
        batch_targets = batch_rewards + self.gamma * batch_next_outputs                            # target batch
        td_loss = F.smooth_l1_loss(batch_outputs, batch_targets)                                   # loss 계산.시간차 손실이라는 뜻으로 temporal difference loss.(손실은 batch에서 target과 출력 사이의 차를 제곱한 값의 합:smooth_l1_loss)
        self.optimizer.zero_grad()                                                                 # zero_grad는 Adam class의 method.이걸 이용해 경사 초기화 -> 모든 가중치 경사 = 0
        td_loss.backward()                                                                         # td_loss를 신경망으로 backpropagation
        self.optimizer.step()                                                                      # Adam의 step을 이용하여 가중치를 update
    # batch_states:입력 batch 상태 / batch_actions: 수행할 행동 batch / batch_reward:받게 될 보상 batch / batch_next_states: 도달할 다음 상태의 batch
    
    
    def update(self, new_state, new_reward):                # 새로 도달한 state와 행동을 수행한 직후 받은 새로운 보상을 입력으로.새로운 상태는 map.py 파일의 129 행의 state 변수. 새 reward는 128행부터 145행까지의 reward 변수
        new_state = torch.Tensor(new_state).float().unsqueeze(0)            # 새로운 상태를 torchtensor로 변환하고 압축을 풀어 batch에 해당하는 추가 차원을 생성.연산을 용이하게 하기위해 새로운 상태의 방향과 세 개의 signal에 float를 취함
        self.memory.push((self.last_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward]), new_state)) # memory의 push를 이용해 새로운 전이를 memory에 추가
        new_action = self.select_action(new_state) # DQN class의 select_action() method를 이용해 방금 도달한 새로운 상태의 새로운 행동을 수행
        if len(self.memory.memory) > 100:          # memory 크기가 100보다 큰지 확인. 첫번째 memory는 ReplayMemory에서 생성한 객체 두번째 memory는 33행에서 만든 variable
            batch_states, batch_actions, batch_rewards, batch_next_states = self.memory.sample(100) # self.memory의 sample() method를 활용해 memory에서 100개의 전이를 sampling.크기 100인 4개 batch 반환(batch_states,actions,rewards,next_state_batch)
            self.learn(batch_states, batch_actions, batch_rewards, batch_next_states)       # 동일한 Dqn class의 learn method를 이용해 dkvdml 4개의 batch를 이용해 가중치 update
        self.last_state = new_state     # 도달한 마지막 상태 self.last_state를 new_state로 update
        self.last_action = new_action   # 마지막으로 수행한 행동 self.last_action을 new action으로 update
        self.last_reward = new_reward   # 마지막으로 받은 보상을 new_reward로 update
        return new_action               # 수행할 새로운 행동 반환
    
    def save(self):   # AI의 가중치를 update하고 그 가중치를 저장하는 method.(지도 호출 시 save 클릭하면 바로 호출됨)
        basename = 'last_brain'
        #suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        #filename = "_".join([basename, suffix])
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                    }, basename + '.pth')    # 가중치 저장되고 last_brain.pth라는 파일에 저장되어 파이썬 파일이 포함된 폴더에 자동으로 저장되는 용도.사전 훈련된 AI를 갖기 위함
        
    
    def load(self):  # last_brain.pth 파일에 저장된 가중치를 load하는 method.
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")         # 지도를 실행할 때 load 버튼 누르면 호출됨.자율 주행 자동차 훈련시키지 않아도 사전 훈련된 자율주행 자동차로 시작하기 위함
