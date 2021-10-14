from cartpole1 import QLearnCartPoleSolver
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import gym
import random

class DQNQLearnCartPoleSolver(QLearnCartPoleSolver):

    def __init__(self, env,  episodes):
        super().__init__(env, episodes=episodes)
        self.memory = deque(maxlen=2000)
        self.batch_size = 32
        self.model = Sequential()
        self.model.add(Dense(12, activation='relu', input_dim=4))
        self.model.add(Dense(12, activation='relu'))
        self.model.add(Dense(2))
        self.model.compile(Adam(learning_rate=0.001), 'mse')
    

    def action(self, state):
        return self.env.action_space.sample() if np.random.random() <= self.epsilon else np.argmax(self.model.predict(state)[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def updated_q_value(self, state, action, reward, new_state):
        return (self.learning_rate * (reward + self.discount * np.max(self.model.predict(new_state)[0])))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0

        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = self.updated_q_value(state, action, reward, next_state)
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

    def train(self):
        scores = []
        for episode in range(self.episodes):
            self.learning_rate = self.get_learning_rate(episode)
            self.epsilon = self.get_epsilon(episode)
            state = self.env.reset().reshape(1, self.env.observation_space.shape[0])
            done = False
            reward_current_ep = 0
            step = 1
            while not done:
                # self.env.render()
                action = self.action(state)
                next_state, reward, done, _ = self.env.step(action) 
                next_state = next_state.reshape(1, self.env.observation_space.shape[0])
                self.remember(state, action, reward, next_state, done)
                state = next_state
                reward_current_ep += reward
                # print(f"Trainingsession {episode+1}:", step, "steps")
                step +=1
            scores.append(reward_current_ep)
            print(f"{scores[episode]}  score for ep {episode+1}")
            self.replay()
        print('Finished training!')
        #self.env.close()

    def run(self):
        done = False
        state = self.env.reset().reshape(1, self.env.observation_space.shape[0])
        score = 0
        step = 0
        while not done:
            action = self.action(state)
            next_state, reward, done, _ = self.env.step(action) 
            next_state = next_state.reshape(1, self.env.observation_space.shape[0])
            self.remember(state, action, reward, next_state, done)
            state = next_state
            score += reward
            step +=1
        print(f"score {score}")
        self.env.close()


env = gym.make('CartPole-v0')


if __name__ == '__main__':
    model = DQNQLearnCartPoleSolver(env, episodes=100)
    model.train()
    model.run()
    