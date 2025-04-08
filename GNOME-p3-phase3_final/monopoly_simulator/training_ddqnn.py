
import os
import sys
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Import DDQN network and state encoder.
from monopoly_simulator.ddqnn import DDQNNetwork
# -----------------------------
# Replay Buffer Implementation
# -----------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Store a transition in the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample a batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# -----------------------------
# DDQN Agent Implementation
# -----------------------------
class DDQNAgent:
    def __init__(self, state_dim=240, action_dim=2950,
                 lr=1e-5, gamma=0.9999, batch_size=128, 
                 replay_capacity=10000, target_update_freq=500):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize the policy network and target network
        self.policy_net = DDQNNetwork(state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.target_net = DDQNNetwork(state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.update_target_network()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(replay_capacity)
        
        # Epsilon parameters for epsilon-greedy action selection.
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        
        self.steps_done = 0

    def select_action(self, state):
        """
        Select an action using epsilon-greedy, restricted to action indices between 2268 and 2922.
        Assumes state is a torch.Tensor of shape (1, state_dim).
        """
        self.steps_done += 1
        lower_bound = 2268
        upper_bound = 2922  # upper_bound is exclusive
        
        if random.random() < self.epsilon:
            # Random action selected from the restricted range.
            return random.randrange(lower_bound, upper_bound)
        else:
            with torch.no_grad():
                q_values = self.policy_net(state.to(self.device))
                # Restrict Q-values to the desired range.
                restricted_q = q_values[:, lower_bound:upper_bound]
                # Get index of max value relative to the restricted slice.
                relative_index = restricted_q.argmax(dim=1).item()
                # Map the relative index to the absolute index.
                return lower_bound + relative_index

    def optimize_model(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample a mini-batch of transitions
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to torch tensors
        states      = torch.FloatTensor(states).to(self.device)
        actions     = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards     = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones       = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Current Q values for actions taken
        current_q = self.policy_net(states).gather(1, actions)
        
        # Compute next Q values using Double DQN:
        next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
        next_q = self.target_net(next_states).gather(1, next_actions)
        
        # Compute target Q values:
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss (Mean Squared Error)
        loss = nn.MSELoss()(current_q, target_q.detach())
        
        # Optimize the network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()

    def update_target_network(self):
        """Copy the policy network parameters to the target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())


# -----------------------------
# Training Loop using Replay Buffer
# -----------------------------
def train_from_replay_buffer(agent, train_steps=1000, update_target_every=500):
    """
    Train the DDQN agent solely using experiences from its replay buffer.
    Assumes that the replay buffer has been pre-populated by game play.
    
    Parameters:
      - agent: the DDQNAgent instance with a populated replay buffer
      - train_steps: total number of training steps to perform
      - update_target_every: update the target network every `update_target_every` steps
    """
    for step in range(1, train_steps + 1):
        loss = agent.optimize_model()
        if loss is not None:
            print(f"Train step {step}: Loss = {loss:.6f}")
        if step % update_target_every == 0:
            agent.update_target_network()
            print(f"Train step {step}: Target network updated.")

if __name__ == "__main__":
    print("Starting DDQN agent training from replay buffer...")
    # Create an agent instance; its replay buffer should be filled externally.
    agent = DDQNAgent()
    # Here one would normally load the pre-populated replay buffer,
    # for example from a file or as accumulated during game play.
    # train_from_replay_buffer uses the experiences already in agent.replay_buffer.
    train_from_replay_buffer(agent, train_steps=1000, update_target_every=500)
    # Optionally, save the final trained model.
    final_model_path = os.path.join("GNOME-p3-phase3_final", "monopoly_simulator", "ddqn_decision_agent_final.pth")
    torch.save(agent.policy_net.state_dict(), final_model_path)
    print(f"Training complete. Final model saved as '{final_model_path}'.")
