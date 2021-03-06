#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-09 20:25:52
@LastEditor: John
LastEditTime: 2021-09-16 00:55:30
@Discription:
@Environment: python 3.7.7
'''
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from gym import spaces

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity  # 经验回放的容量
        self.buffer = []  # 缓冲区
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        ''' 缓冲区是一个队列，容量超出时去掉开始存入的转移(transition)
        '''
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)  # 随机采出小批量转移
        state, action, reward, next_state, done = zip(*batch)  # 解压成状态，动作等
        return state, action, reward, next_state, done

    def __len__(self):
        ''' 返回当前存储的量
        '''
        return len(self.buffer)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, init_w=3e-3):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, action_dim)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, init_w=3e-3):
        super(Critic, self).__init__()

        self.linear1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        # 随机初始化为较小的值
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        # 按维数1拼接
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class DDPG:
    def __init__(self,
                 max_action,
                 user_id,
                 action_dim,
                 state_dim,
                 discount=0.99,
                 batch_size=256,
                 beta=0.0003):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.critic = Critic(state_dim, action_dim, 32).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.actor = Actor(state_dim, action_dim, 32).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.target_critic = Critic(state_dim, action_dim, 32).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.target_actor = Actor(state_dim, action_dim, 32).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # 复制参数到目标网络
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=1e-3)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.memory = ReplayBuffer(8000)
        self.batch_size = batch_size
        self.soft_tau = 1e-2 # 软更新参数
        self.gamma = discount

        self.exploration = 0
        self.exploration_total = 1000
        self.action_space = spaces.MultiDiscrete([max_action * max_action + 1])

    def select_action(self, state):
        self.exploration += 1
        if self.exploration < self.exploration_total:
            type = random.randint(-1, 1)
            if type != 0:
                price = random.randint(1, 100)
                number = random.randint(0, 100)
            else:
                price = 0
                number = 0
            return [type * number, price]
        else:
            state = T.Tensor(np.array([observation])).to(device)
            actions, _ = self.actor.sample_normal(state, reparameterize=False)

            return actions.cpu().detach().numpy().reshape([2]).tolist()  #

    def train(self):
        if len(self.memory) < self.batch_size:  # 当 memory 中不满足一个批量时，不更新策略
            return
        # 从经验回放中(replay memory)中随机采样一个批量的转移(transition)
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        # 转变为张量
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        policy_loss = self.critic(state, self.actor(state))
        policy_loss = -policy_loss.mean()
        next_action = self.target_actor(next_state)
        target_value = self.target_critic(next_state, next_action.detach())
        expected_value = reward + (1.0 - done) * self.gamma * target_value
        expected_value = torch.clamp(expected_value, -np.inf, np.inf)

        value = self.critic(state, action)
        value_loss = nn.MSELoss()(value, expected_value.detach())

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()
        # 软更新
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) +
                param.data * self.soft_tau
            )
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) +
                param.data * self.soft_tau
            )

    def store_transition(self,
                         curr_state,
                         action,
                         next_state,
                         reward,
                         done=False):
        self.memory.push(curr_state, action, reward, next_state, done)

    def save(self, path):
        torch.save(self.actor.state_dict(), path + 'checkpoint.pt')

    def load(self, path):
        self.actor.load_state_dict(torch.load(path + 'checkpoint.pt'))