# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
#https://www.youtube.com/watch?v=ewRw996uevM&list=PLZbbT5o_s2xoWNVdDudn51XM8lOuZ_Njv&index=18
#https://www.youtube.com/watch?v=0bt0SjbS3xc&list=PLZbbT5o_s2xoWNVdDudn51XM8lOuZ_Njv&index=13


# %%
import numpy as np
from cartpole1 import QLearnCartPoleSolver
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import torch.nn.functional as F
import gym

import torch
import torch.nn as nn
import random
import torchvision

def get_device():
    if torch.cuda.is_available(): 
        dev = "cuda:0" 
    else: 
        dev = "cpu" 
    return torch.device(dev)

device = get_device()


# %%
class DQN(nn.Module):
    def __init__(self, height, width):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=height*width*3, out_features=24), 
            nn.Flatten(start_dim=1),
            nn.ReLU(),
            nn.Linear(in_features=24, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=2))


# %%


class DQNSolver(QLearnCartPoleSolver):

    def __init__(self, env,  episodes, device, epsilon_decay_rate=0.995):
        super().__init__(env, episodes=episodes, min_epsilon=0.001)
        self.memory = deque(maxlen=100000)
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon = 1
        self.batch_size = 64
        self.device = device
        self.lr = 0.01
        self.model = DQN(12, 12).to(self.device)
        self.optimizer = torch.optim.RMSprop(self.model.parameters())

    def action(self, state, epsilon):
        if np.random.random() <= epsilon: 
            return torch.tensor(self.env.action_space.sample()).to(self.device)
        else:
            with torch.no_grad(): 
                self.model.predict(state).argmax(dim =1).to(self.device)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def updated_q_value(self, state, action, reward, next_state):
        return (reward + self.discount * self.model.predict(next_state).argmax(dim =1).to(self.device))


    def create_batch_tensor(self,batch):
        states, actions, rewards, next_states, done = batch
        return (torch.tensor(states),
            torch.tensor(actions), 
            torch.tensor(rewards), 
            torch.tensor(next_states),  
            torch.tensor(done))

    def replay(self):
        if self.batch_size >= len(self.memory):
            return
        current = []
        target = []
        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            current_q = self.model(state).gather(dim=1, index=action.unsqueeze(-1))
            target_q = self.updated_q_value(state, action, reward, next_state)
            current.append(current_q)
            target.append(target_q)
        loss = F.mse_loss(current - target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        for episode in range(self.episodes):
            done = False
            reward_current_ep = 0
            step = 1
            state = self.env.reset()
            while not done:
                # self.env.render()
                action = self.action(state, self.get_epsilon(episode))
                next_state, reward, done, _ = self.env.step(action) 
                self.remember(state, action, reward, next_state, done)
                state = next_state
                reward_current_ep += reward
                step +=1
            self.replay()
        # self.env.close()

env = gym.make('CartPole-v0')


model = DQNSolver(env=env, device = device, episodes=100)
model.train()


