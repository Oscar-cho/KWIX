import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

class FANET:
    def __init__(self):
        self.size = (500, 500, 100) # (x, y, z)
        self.start_zone = (0, 10, 1)
        self.terminal_zone = (90, 90, 90)
        self.hazard_zone_1 = (50, 50, 50) # (40, 40, 0) ~ (50, 50, 50)
        self.hazard_zone_2 = (150, 150, 30) # (100, 130, 20) ~ (150, 150, 30)
        self.hazard_zone_3 = (450, 450, 70) # (430, 430, 0) ~ (450, 450, 70)
        self.reset()

    def reset(self):
        self.state = self.start_zone
        self.steps = 0

    def step(self, action):
        if action == '+x' and self.state[0] < self.size[0] - 5:
            self.state = (self.state[0] + 5, self.state[1], self.state[2])
        elif action == '-x' and self.state[0] > 5:
            self.state = (self.state[0] - 5, self.state[1], self.state[2])
        elif action == '+y' and self.state[1] < self.size[1] - 5:
            self.state = (self.state[0], self.state[1] + 5, self.state[2])
        elif action == '-y' and self.state[1] > 5:
            self.state = (self.state[0], self.state[1] - 5, self.state[2])
        elif action == '+z' and self.state[2] < self.size[2] - 5:
            self.state = (self.state[0], self.state[1], self.state[2] + 5)
        elif action == '-z' and self.state[2] >= 5:
            self.state = (self.state[0], self.state[1], self.state[2] - 5)
        
        reward = self.reward()
        
        return self.state, reward, self.terminal_state()

    def reward(self):
        reward = -1
        
        if self.terminal_state():
            reward = 100
            
        if (self.hazard_zone_1[0] - 0 <= self.state[0] <= self.hazard_zone_1[0] and
              self.hazard_zone_1[1] - 0 <= self.state[1] <= self.hazard_zone_1[1] and
              self.hazard_zone_1[2] - 50 <= self.state[2] <= self.hazard_zone_1[2]):
            reward = -100
            
        if (self.hazard_zone_2[0] - 50 <= self.state[0] <= self.hazard_zone_2[0] and
              self.hazard_zone_2[1] - 20 <= self.state[1] <= self.hazard_zone_2[1] and
              self.hazard_zone_2[2] - 10 <= self.state[2] <= self.hazard_zone_2[2]):
            reward = -100
            
        if (self.hazard_zone_3[0] - 20 <= self.state[0] <= self.hazard_zone_3[0] and
              self.hazard_zone_3[1] - 20 <= self.state[1] <= self.hazard_zone_3[1] and
              self.hazard_zone_3[2] - 70 <= self.state[2] <= self.hazard_zone_3[2]):
            reward = -100
        
        return reward
    
    def terminal_state(self):
        return (self.state[0] >= self.terminal_zone[0] and
                self.state[1] >= self.terminal_zone[1] and
                self.state[2] >= self.terminal_zone[2])

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class QAgent():
    def __init__(self, env, gamma, epsilon, num_episodes, NN):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        self.optimizer = optim.SGD(NN.parameters(), lr=0.0001)
        self.NN = NN

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(6)
        else:
            state_tensor = torch.FloatTensor(state)
            action_values = self.NN(state_tensor)
            action = torch.argmax(action_values).item()
        return action
    
    def semi_gradient_qlearning(self):
        step_count = []
        total_reward = []

        for episode in tqdm(range(self.num_episodes)):
            self.env.reset()
            state = self.env.state
            stop = False
            episode_step_count = 0
            episode_total_reward = 0

            #for _ in range(100):
            while not stop:
                action = self.get_action(state)
                next_state, reward, stop = self.env.step(['+x', '-x', '+y', '-y', '+z', '-z'][action])
                episode_total_reward += reward

                if stop:
                    target = reward
                else:
                    next_state_tensor = torch.FloatTensor(next_state)
                    next_action_values = self.NN(next_state_tensor)
                    target = reward + self.gamma * torch.max(next_action_values).item()

                state_tensor = torch.FloatTensor(state)
                action_values = self.NN(state_tensor)
                action_value = action_values[action]
                loss = nn.MSELoss()(action_value, torch.tensor(target, dtype=torch.float))
                self.NN.zero_grad()
                loss.backward()
                self.optimizer.step()

                state = next_state
                episode_step_count += 1
                
                if stop:
                    break
            
            step_count.append(episode_step_count)
            total_reward.append(episode_total_reward)

        return step_count, total_reward

    def train(self):
        step_count, total_reward = self.semi_gradient_qlearning()
        self.plot_results(step_count, total_reward)

    def plot_results(self, step_count, total_reward):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(total_reward)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')

        plt.subplot(1, 2, 2)
        plt.plot(step_count)
        plt.title('Episode Step Count')
        plt.xlabel('Episode')
        plt.ylabel('Step Count')

        plt.tight_layout()
        plt.show()
        
    def save_model(self, path):
        torch.save(self.NN.state_dict(), path)
        print("Model saved Successful")
        
    def plot_optimal_path(self):
        pass
        
if __name__ == '__main__':
    env = FANET()
    input_size = 3
    output_size = 6
    NN = NeuralNetwork(input_size, output_size)
    agent = QAgent(env, gamma=0.99, epsilon=0.3, num_episodes=5000, NN=NN)
    agent.train()
    
    model_path = 'trained_model.pt'
    agent.save_model(model_path)
