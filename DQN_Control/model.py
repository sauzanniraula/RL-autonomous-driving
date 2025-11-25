import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ... (Keep ConvNet class exactly as it is) ...
class ConvNet(nn.Module):
    def __init__(self, dim, in_channels, num_actions) -> None:
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 8, 4)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 4, 3)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.conv3_bn = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(64*8*8, 256)
        self.fc1_bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 32)
        self.fc2_bn = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, num_actions)
    
    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.fc1_bn(self.fc1(x.reshape(-1, 64*8*8))))
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = self.fc3(x)
        return x

class DQN(object):
    def __init__(
        self,
        num_actions,
        state_dim,
        in_channels,
        device,
        discount=0.9,
        optimizer="Adam",
        optimizer_parameters={'lr':0.01},
        target_update_frequency=1e4,
        initial_eps = 1,
        end_eps = 0.05,
        eps_decay_period = 25e4,
        eval_eps=0.001
    ) -> None:
        self.device = device
        self.Q = ConvNet(state_dim, in_channels, num_actions).to(self.device)
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer = getattr(torch.optim, optimizer)(self.Q.parameters(), **optimizer_parameters)
        
        self.discount = discount
        self.target_update_frequency = target_update_frequency
        
        # Epsilon parameters
        self.initial_eps = initial_eps
        self.end_eps = end_eps
        self.eps_decay_period = eps_decay_period
        self.slope = (self.end_eps - self.initial_eps) / eps_decay_period
        
        self.state_shape = (-1,) + state_dim
        self.eval_eps = eval_eps
        self.num_actions = num_actions
        self.iterations = 0
        self.current_eps = initial_eps

    def select_action(self, state, eval=False):
        # Calculate epsilon based on total iterations
        if not eval:
            self.current_eps = max(self.slope * self.iterations + self.initial_eps, self.end_eps)
        
        eps = self.eval_eps if eval else self.current_eps

        if np.random.uniform(0,1) > eps:
            self.Q.eval()
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                return int(self.Q(state).argmax(1))
        else:
            return np.random.randint(self.num_actions)

    def train(self, replay_buffer):
        self.Q.train()
        state, action, next_state, reward, done = replay_buffer.sample()

        with torch.no_grad():
            best_action = self.Q(next_state).argmax(1, keepdim=True)
            target_Q = reward + (1-done) * self.discount * self.Q_target(next_state).gather(1, best_action)

        current_Q = self.Q(state).gather(1, action)
        Q_loss = F.smooth_l1_loss(current_Q, target_Q)

        self.Q_optimizer.zero_grad()
        Q_loss.backward()
        self.Q_optimizer.step()

        self.iterations += 1
        self.copy_target_update()
        
        return Q_loss.item()
    
    def copy_target_update(self):
        if self.iterations % self.target_update_frequency == 0:
            print('Target network updated')
            print(f'Current Epsilon: {self.current_eps:.4f}')
            self.Q_target.load_state_dict(self.Q.state_dict())

    def save(self, filename):
        # Save model state, optimizer, and current training progress
        checkpoint = {
            'q_state_dict': self.Q.state_dict(),
            'optimizer_state_dict': self.Q_optimizer.state_dict(),
            'iterations': self.iterations,
            'current_eps': self.current_eps
        }
        torch.save(checkpoint, filename + ".pth")

    def load(self, filename):
        # Load the checkpoint dictionary
        print(f"Loading model from {filename}.pth")
        try:
            checkpoint = torch.load(filename + ".pth")
            self.Q.load_state_dict(checkpoint['q_state_dict'])
            self.Q_target = copy.deepcopy(self.Q)
            self.Q_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Restore training progress
            self.iterations = checkpoint.get('iterations', 0)
            self.current_eps = checkpoint.get('current_eps', 1.0)
            print(f"Model loaded. Resuming from Iteration {self.iterations}, Epsilon {self.current_eps:.4f}")
        except Exception as e:
            print(f"Error loading model: {e}")
