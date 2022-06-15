import math
from gym import spaces
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torch.optim as optiom
class MLP(nn.Module):
    def __init__(self,state_dim,action_dim,hidden_dim=128):
        super(MLP,self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(state_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,hidden_dim)
        self.fc3 = nn.Linear(hidden_dim,action_dim)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DuelingNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DuelingNet, self).__init__()

        # 隐藏层
        self.hidden = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )

        # 优势函数
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # 价值函数
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x = self.hidden(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean()


class ReplayBuffer:
    def __init__(self,capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self,state,action,reward,state_next,done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None) #开辟内存空间

        self.buffer[self.position] = (state,action,reward,state_next,done)
        self.position = (self.position+1) % self.capacity

    def sample(self,batch_size):
        if len(self.buffer)<batch_size:
            return

        batch_data = random.sample(self.buffer,batch_size)
        state, action, reward, state_next,done = zip(*batch_data)
        return state,action,reward,state_next,done
    def __len__(self):
        return len(self.buffer)

class DQN:
    def __init__(self,state_dim,action_dim,max_action,user_id,discount,batch_size,beta,hidden_dim = 256,alg='DDQN',epsilon_end=0.01,epsilon_start=0.95,epsilon_decay=500,memory_capacity=100000):
        self.action_dim = action_dim
        self.device = torch.device('cuda')
        self.frame_idx = 0
        self.gamma = discount
        self.epsilon = lambda frame_idx :epsilon_end + (epsilon_start-epsilon_end)*math.exp(-1.*frame_idx/epsilon_decay)
        self.batch_size = batch_size
        if alg=="DDQN" or alg=="DQN":
            self.police_net = MLP(state_dim,action_dim,hidden_dim=hidden_dim).to(self.device)
            self.tartget_net = MLP(state_dim,action_dim,hidden_dim=hidden_dim).to(self.device)
        elif alg=="DuelDQN":
            self.police_net = DuelingNet(state_dim, action_dim, hidden_dim=hidden_dim).to(self.device)
            self.tartget_net = DuelingNet(state_dim, action_dim, hidden_dim=hidden_dim).to(self.device)
        for target_param,param in zip(self.tartget_net.parameters(),self.police_net.parameters()):
            target_param.data.copy_(param.data)
        self.optimizer = optiom.Adam(self.police_net.parameters(),lr=beta)
        self.memory = ReplayBuffer(memory_capacity)
        self.alg = alg
        self.action_space = spaces.MultiDiscrete([max_action, max_action])


    def select_action(self,state): #ε-贪心和玻尔兹曼探索（Boltzmann exploration）
        self.frame_idx += 1
        if random.random()>self.epsilon(self.frame_idx): #选择动作
            with torch.no_grad():
                state = torch.tensor([state], device=self.device,dtype=torch.float32)
                q_values = self.police_net(state)

                action = q_values.max(1)[1].item()

        else:
            action = random.randrange(self.action_dim)
        print('aaaa',action)
        return action

    def train(self):
        if len(self.memory)<self.batch_size:

            return
        state_batch, action_batch, state_next_batch,reward_batch, done_batch = self.memory.sample(self.batch_size)
        state_batch = torch.tensor(state_batch,device=self.device,dtype=torch.float32)

        action_batch = torch.tensor(action_batch,device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(reward_batch ,device=self.device,dtype=torch.float32)
        state_next_batch = torch.tensor(state_next_batch,device=self.device,dtype=torch.float32)
        done_batch = torch.tensor(done_batch,device=self.device,dtype=torch.float32)


#DQN-----------
        if self.alg=="DQN":
            q_values = self.police_net(state_batch).gather(dim=1,index=action_batch)
            next_q_values = self.tartget_net(state_next_batch).max(1)[0].detach()
            expect_q_values = reward_batch + self.gamma*next_q_values*(1-done_batch)
# DQN-----------
#DDQN------
        elif self.alg == "DDQN" or "DuelDQN":

            q_values = self.police_net(state_batch).gather(dim=1, index=action_batch)

            next_q_action = self.police_net(state_batch).max(1)[1].detach()


            next_q_target = self.tartget_net(state_next_batch).gather(1,next_q_action.unsqueeze(1)).squeeze(1)

            expect_q_values = reward_batch + self.gamma * next_q_target * (1 - done_batch)


#DDQN------
        loss = nn.MSELoss()(q_values,expect_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.police_net.parameters():
            param.grad.data.clamp_(-1,1)
        self.optimizer.step()
    def store_transition(self,curr_state,action,next_state,reward,done):
        self.memory.push(curr_state, action, next_state, reward, done)
    def save(self,path):
        torch.save(self.tartget_net.state_dict(),path+'dqn.pth')


    def load(self,path):
        self.tartget_net.load_state_dict(torch.load(path+'dqn.pth'))
        for param,param_target in zip(self.police_net.parameters(),self.tartget_net.parameters()):
            param.data.copy_(param_target.data)