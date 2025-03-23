import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Import your DDQN network and state encoder.
# Make sure these paths are correct in your workspace.
from ddqnn import DDQNNetwork
from monopoly_simulator.monopoly_state_encoder import MonopolyStateEncoder, create_initial_state

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
    def __init__(self, state_dim=240, action_dim=2922, 
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
        
        # Epsilon parameters for epsilon-greedy
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        
        self.steps_done = 0
    
    def select_action(self, state):
        """
        Select an action using epsilon-greedy.
        Assumes state is a torch.Tensor of shape (1, state_dim).
        In this example, we choose the action index with max Q-value.
        In practice, you might use action masking using your ActionEncoder.
        """
        self.steps_done += 1
        if random.random() < self.epsilon:
            # Random action (note: in a real simulator, select only from valid actions)
            return random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                q_values = self.policy_net(state.to(self.device))
                # Use torch.argmax for index (action) selection.
                return q_values.argmax(dim=1).item()

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
        # 1. Select best action from policy network
        next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
        # 2. Evaluate with target network
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
# Dummy Environment
# -----------------------------
class DummyMonopolyEnv:
    """
    This dummy environment uses the initial state from your MonopolyStateEncoder.
    Replace step() with your actual environment logic.
    """
    def __init__(self):
        self.encoder = MonopolyStateEncoder()
        self.state = create_initial_state().squeeze().numpy()  # 240-dim vector

    def reset(self):
        self.state = create_initial_state().squeeze().numpy()
        return self.state

    def step(self, action):
        """
        For demonstration, simulate:
         - Next state as a small random perturbation,
         - Reward as a random float,
         - done flag with a small probability.
        In your actual environment, use your game simulation logic.
        """
        next_state = self.state + np.random.normal(0, 0.01, size=self.state.shape)
        reward = np.random.rand()  # dummy reward between 0 and 1
        done = np.random.rand() < 0.05  # 5% chance episode ends at each step
        self.state = next_state
        return next_state, reward, done

# -----------------------------
# Training Loop
# -----------------------------
def train_ddqn(num_episodes=1000, max_steps_per_episode=200):
    env = DummyMonopolyEnv()
    agent = DDQNAgent()
    update_counter = 0
    
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        episode_loss = 0
        for t in range(max_steps_per_episode):
            # Convert state to tensor shape (1, state_dim)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = agent.select_action(state_tensor)
            
            # Interact with the environment
            next_state, reward, done = env.step(action)
            
            # Save transition in replay buffer
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # Optimize model after each step
            loss = agent.optimize_model()
            if loss is not None:
                episode_loss += loss
                update_counter += 1
            
            state = next_state
            if done:
                break
        
        # Update target network every target_update_freq episodes
        if episode % agent.target_update_freq == 0:
            agent.update_target_network()
        
        avg_loss = episode_loss / max(1, update_counter)
        print(f"Episode {episode}: steps={t+1}, avg_loss={avg_loss:.6f}, epsilon={agent.epsilon:.4f}")
        update_counter = 0  # reset update counter per episode

if __name__ == "__main__":
    # Start training
    train_ddqn(num_episodes=1000)