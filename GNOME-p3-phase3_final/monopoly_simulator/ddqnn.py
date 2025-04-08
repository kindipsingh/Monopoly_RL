import torch
import torch.nn as nn
import torch.nn.functional as F

class DDQNNetwork(nn.Module):
    """
    A refined DDQN neural network for a Monopoly RL agent.
    
    Architecture:
      - Input: 240-dimensional state encoding (from MonopolyStateEncoder)
      - Hidden Layer 1: 1024 neurons, ReLU activation
      - Hidden Layer 2: 512 neurons, ReLU activation
      - Output: 2922 neurons representing Q-values for each action
      
    Training Details (to be handled externally):
      - Optimizer: Adam (recommended learning rate: 1e-5)
      - Loss Function: Mean Squared Error (MSE)
      - Discount Factor: 0.9999
      - Batch Size: 128
      - Experience Replay Buffer Size: 10^4 entries
      - Target Network Update: Every 500 episodes, update target network parameters to match policy network
    """
    def __init__(self, state_dim=240, action_dim=2950):
        super(DDQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, action_dim)

    def forward(self, state):
        """
        Forward pass through the network.
        
        Parameters:
            state (torch.Tensor): Input tensor of shape (batch_size, state_dim)
            
        Returns:
            torch.Tensor: Q-values for each action, shape (batch_size, action_dim)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.out(x)
        return q_values

