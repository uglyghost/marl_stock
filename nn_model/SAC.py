import os
import random
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np

device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')


class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.int)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.int)
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.int)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.int)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, curr_state, action, next_state, reward, done):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = curr_state
        self.new_state_memory[index] = next_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions, fc1_dims=256, fc2_dims=256,
            name='critic', chkpt_dir='./save_model/SAC'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.fc1 = nn.Linear(self.input_dims[0]+n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)

        self.to(device)

    def forward(self, state, action):
        action_value = self.fc1(T.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        q = self.q(action_value)

        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class ValueNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims=256, fc2_dims=256,
            name='value', chkpt_dir='./save_model/SAC'):
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)

        self.to(device)

    def forward(self, state):
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)

        v = self.v(state_value)

        return v

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, max_action, fc1_dims=256,
            fc2_dims=256, n_actions=2, name='actor', chkpt_dir='./save_model/SAC'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        self.max_action = max_action
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        self.to(device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = self.mu(prob)
        sigma = self.sigma(prob)

        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)

        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        if T.any(T.isnan(mu)) or T.any(T.isnan(sigma)):
            print('Nan')
        probabilities = Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        action = T.tanh(actions)*T.tensor(self.max_action).to(device)
        log_probs = probabilities.log_prob(actions)
        log_probs -= T.log(1-action.pow(2)+self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class SAC:
    def __init__(self,
                 max_action,
                 user_id,
                 alpha=0.0003,
                 beta=0.0003,
                 state_dim=400,
                 discount=0.99,
                 action_dim=200,
                 max_size=100,
                 tau=0.005,
                 batch_size=256,
                 reward_scale=2):
        self.gamma = discount
        self.tau = tau
        self.action_dim = action_dim
        self.max_size = max_size
        self.state_dim = [state_dim]
        self.memory = ReplayBuffer(max_size, self.state_dim, action_dim)
        self.batch_size = batch_size
        self.n_actions = action_dim

        self.actor = ActorNetwork(alpha, self.state_dim, n_actions=action_dim,
                                  name='actor_' + str(user_id), max_action=max_action)
        self.critic_1 = CriticNetwork(beta, self.state_dim, n_actions=action_dim,
                                      name='critic_1' + str(user_id))
        self.critic_2 = CriticNetwork(beta, self.state_dim, n_actions=action_dim,
                                      name='critic_2' + str(user_id))
        self.value = ValueNetwork(beta, self.state_dim, name='value' + str(user_id))
        self.target_value = ValueNetwork(beta, self.state_dim, name='target_value' + str(user_id))

        self.scale = reward_scale
        self.update_network_parameters(tau=1)
        self.exploration = 0
        self.exploration_total = 1000

    def reset_memory(self):
        self.target_value.optimizer = optim.Adam(self.target_value.parameters(), lr=3e-4)
        self.value.optimizer = optim.Adam(self.value.parameters(), lr=3e-4)
        self.actor.optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_1.optimizer = optim.Adam(self.critic_1.parameters(), lr=3e-4)
        self.critic_2.optimizer = optim.Adam(self.critic_2.parameters(), lr=3e-4)
        self.memory = ReplayBuffer(self.max_size, self.state_dim, self.action_dim)

    def select_action(self, observation):
        self.exploration += 1
        if self.exploration < self.exploration_total:
            type = random.randint(-1, 1)
            if type != 0:
                price = random.randint(0, 100)
                number = random.randint(1, 100)
            else:
                price = 1
                number = 0
            return [type * number, price]
        else:
            state = T.Tensor(np.array([observation])).to(device)
            actions, _ = self.actor.sample_normal(state, reparameterize=False)

            return actions.cpu().detach().numpy().reshape([2]).tolist()  #

    def store_transition(self, state, action, new_state, reward, done):
        self.memory.store_transition(state, action, new_state, reward, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau * value_state_dict[name].clone() + \
                                     (1 - tau) * target_value_state_dict[name].clone()

        self.target_value.load_state_dict(value_state_dict)

    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    def train(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(device)
        done = T.tensor(done).to(device)
        state_ = T.tensor(new_state, dtype=T.float).to(device)
        state = T.tensor(state, dtype=T.float).to(device)
        action = T.tensor(action, dtype=T.float).to(device)

        value = self.value(state).view(-1)
        value_ = self.target_value(state_).view(-1)
        value_[done] = 0.0
        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        actor_loss = log_probs - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale * reward + self.gamma * value_
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()

        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters()
