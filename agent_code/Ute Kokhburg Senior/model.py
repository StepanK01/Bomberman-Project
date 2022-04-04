import torch as T
import torch.nn as nn
import torch.optim as optim
import numpy as np


class DQN(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions): 
        super(DQN, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        in_channels, s, _ = input_dims[0]
        linear_features_size = input_dims[1]

        def conv2d_size_out(size, kernel_size = 3, stride = 1):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        
        convs = conv2d_size_out(conv2d_size_out(conv2d_size_out(s)))
        linear_input_size = convs**2 * 40 + linear_features_size

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, groups=1),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, groups=1),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Conv2d(in_channels=20, out_channels=40, kernel_size=3, groups=1),
            nn.BatchNorm2d(40),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(linear_input_size, self.fc1_dims),
            nn.ReLU(),
            nn.Linear(self.fc1_dims, self.fc2_dims),
            nn.ReLU(),
            nn.Linear(self.fc2_dims, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(linear_input_size, self.fc1_dims),
            nn.ReLU(),
            nn.Linear(self.fc1_dims, self.fc2_dims),
            nn.ReLU(),
            nn.Linear(self.fc2_dims, n_actions)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, vision_features, linear_features):
        x = self.conv(vision_features)
        x = x.view(x.size(0), -1)
        y = T.cat((x,linear_features),dim=1)
        values = self.value_stream(y)
        advantages = self.advantage_stream(y)
        return values, advantages

class Agent():
    def __init__(self, n_actions, input_dims, lr, log):
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.Q_eval = DQN(self.lr, n_actions=self.n_actions, input_dims=self.input_dims, fc1_dims=256, fc2_dims=128)
        self.Q_target = DQN(self.lr, n_actions=self.n_actions, input_dims=self.input_dims, fc1_dims=256, fc2_dims=128)
        self.Q_target.load_state_dict(self.Q_eval.state_dict())
        self.log = log



    def init_training(self, gamma, epsilon, batch_size, max_mem_size, eps_end, eps_dec, target_sync):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.action_space = [i for i in range(self.n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0 

        self.target_cnt = 0
        self.target_sync = target_sync

        self.state_memory_vision = np.zeros((self.mem_size, *self.input_dims[0]), dtype=np.float32)
        self.new_state_memory_vision = np.zeros((self.mem_size, *self.input_dims[0]), dtype=np.float32)
        self.state_memory_linear = np.zeros((self.mem_size, self.input_dims[1]), dtype=np.float32)
        self.new_state_memory_linear = np.zeros((self.mem_size, self.input_dims[1]), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory_vision[index] = state[0]
        self.new_state_memory_vision[index] = state_[0]
        self.state_memory_linear[index] = state[1]
        self.new_state_memory_linear[index] = state_[1]
        self.reward_memory[index] = reward 
        self.action_memory[index] = action 
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def choose_train_action(self, features):
        if np.random.random() > self.epsilon:            
            vision_feature, linear_feature = features[0].astype(np.float32), features[1].astype(np.float32)
            vision_tensor = T.tensor(np.array([vision_feature])).to(self.Q_eval.device)
            linear_tensor = T.tensor(np.array([linear_feature])).to(self.Q_eval.device)
            _, advantage = self.Q_eval.forward(vision_tensor,linear_tensor)
            action = T.argmax(advantage).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def choose_action(self, features):
        vision_feature, linear_feature = features[0].astype(np.float32), features[1].astype(np.float32)
        vision_tensor = T.tensor(np.array([vision_feature])).to(self.Q_eval.device)
        linear_tensor = T.tensor(np.array([linear_feature])).to(self.Q_eval.device)
        _, advantage = self.Q_eval.forward(vision_tensor,linear_tensor)
        action = T.argmax(advantage).item()
        return action


    def learn(self):
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        if ((self.target_cnt % self.target_sync) == 0):
            self.Q_target.load_state_dict(self.Q_eval.state_dict())

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch_vision = T.tensor(self.state_memory_vision[batch]).to(self.Q_eval.device)
        new_state_batch_vision = T.tensor(self.new_state_memory_vision[batch]).to(self.Q_eval.device)
        state_batch_linear = T.tensor(self.state_memory_linear[batch]).to(self.Q_eval.device)
        new_state_batch_linear = T.tensor(self.new_state_memory_linear[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        action_batch = self.action_memory[batch]

        V_s, A_s = self.Q_eval.forward(state_batch_vision, state_batch_linear)
        V_s_, A_s_ = self.Q_target.forward(new_state_batch_vision, new_state_batch_linear)
        V_s_eval, A_s_eval = self.Q_eval.forward(new_state_batch_vision, new_state_batch_linear)

        q_pred = T.add(V_s,(A_s - A_s.mean(dim=1, keepdim=True)))[batch_index, action_batch]
        q_next = T.add(V_s_,(A_s_ - A_s_.mean(dim=1, keepdim=True)))
        q_eval = T.add(V_s_eval,(A_s_eval - A_s_eval.mean(dim=1, keepdim=True)))

        max_actions = T.argmax(q_eval, dim=1)

        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * q_next[batch_index, max_actions]

        loss = self.Q_eval.loss(q_target, q_pred).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.target_cnt += 1

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min  else self.eps_min

    def load_agent(self, path):
        self.Q_eval.load_state_dict(T.load(path,map_location=self.Q_eval.device))

    def save_agent(self, path):
        T.save(self.Q_eval.state_dict(), path)

