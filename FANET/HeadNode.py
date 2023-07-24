'''
class Learning():   # Head Node : A를 위한 Class
    # Q-learning, DQN
    # A 위치 이동
    def __init__(self):
        # 환경을 연결해줘야함. env = FANET_env
        # 학습 파라미터, gamma, alpha 결정
    
    def Q-learning or DQN(self):
        # input -> ENV. State 정보를 받아오면 됨.
        # output -> B,C,D 위치를 결정하는 위치값
        for episode in range(10000):
            # Algorithms
        
    def DQN -> ReplayMemory # Head Node 만을 위한 함수
        # Replay Memory 관련 mini_batch 결정
'''



import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class HeadNode_2D():
    def __init__(self, env, head_pos):
        self.env = env              # FANET 환경 종속변수로 사용
        self.head_pos = head_pos    # A의 위치 정보 // ENV의 State변수가 존재한다면, A의 Position을 받아오면 됨
        #self.head_pos = env.state[0] #예제
        self.gamma = 0.99           # Discount Factor
        self.alpha = 0.0001         # Learning rate
        
    def get_action(self):   # B, C, D의 위치정보를 결정해줘야 함. Env와 연결
        # B, C, D 노드의 목적 위치를 Set()으로 초기화
        B_pos = ()
        C_pos = ()
        D_pos = ()
        
        '''
        class AnotherNode로 부터 현재의 위치를 받아올 것.
        class NeuralNet으로 부터 그다음 Action 값을 결정할 것.
        class FANET_Env를 통해 Action 값을 보내주고 그에 따른 Output을 받아올 것.
        '''
        
    def Q_learning(self):
        pass
    
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


class DQN(object):
    
    def __init__(self, input_size, action, gamma):
        self.gamma = gamma
        self.model = Network(input_size, action) 
        self.memory = ReplayMemory(capacity=1000000)
        self.optimizer = optim.Adam(params = self.model.parameter())
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
        
    def select_action(self, state): #softmax를 이용하여 반복할 때마다 수행할 동작 선택
        probs = F.softmax(self.model(Variable(state))*100)
        action = probs.multinomial(len(probs))  # 다항분포
        return action.data[0,0]
    
    def learn(self, batch_states, batch_actions, batch_rewards, batch_next_states):      # 시간차를 계산하고 손실에 따라 최적화 알고리즘으로 손실을 줄이기 위해 가중치 update
        batch_outputs = self.model(batch_states).gather(1, batch_actions.unsqueeze(1)).squeeze(1)  # Q(StB +1, a) 수집 -> batch에서 입력 상태와 수행할 행동에 대해 예측한 Q값.
        batch_next_outputs = self.model(batch_next_states).detach().max(1)[0]                      # target의 argmaxa(Q(StB+1, a)) 계산
        batch_targets = batch_rewards + self.gamma * batch_next_outputs                            # target batch
        td_loss = F.smooth_l1_loss(batch_outputs, batch_targets)                                   # loss 계산.시간차 손실이라는 뜻으로 temporal difference loss.(손실은 batch에서 target과 출력 사이의 차를 제곱한 값의 합:smooth_l1_loss)
        self.optimizer.zero_grad()                                                                
        td_loss.backward()                                                                         # td_loss를 신경망으로 backpropagation
        self.optimizer.step()                                                                      
''' batch_state = 입력 batch / batch_action '''
    

class ReplayMemory():
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