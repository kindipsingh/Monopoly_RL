import random
import torch
import torch.nn as nn
import torch.optim as optim
# Ensure internal logging is imported; assuming rl_logger exists as in other parts of the code
import logging
rl_logger = logging.getLogger("rl_agent_logs")

class DDQNAgent:
    def __init__(self, state_dim=240, action_dim=2922, 
                 lr=1e-5, gamma=0.9999, batch_size=128, 
                 replay_capacity=10000, target_update_freq=500, 
                 background_agent_methods=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize policy and target networks using the internal organization versions 
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

        # background_agent_methods is expected to be a dict where keys are decision function names
        # For example: {'make_pre_roll_move': func1, 'make_post_roll_move': func2, 'skip_turn': func3, ...}
        self.background_agent_methods = background_agent_methods if background_agent_methods is not None else {}

        # Create a default mapping from index to function name using available background methods.
        # Here, we simply enumerate the keys. When a random index is generated, we use modulo arithmetic 
        # to pick from the available function names.
        if self.background_agent_methods:
            self._action_mapping = {i: key for i, key in enumerate(list(self.background_agent_methods.keys()))}
        else:
            self._action_mapping = {}

    def select_action(self, state):
        """
        Select an action using epsilon-greedy.
        For random actions, map the random index to a corresponding function name 
        from the background_agent_methods if possible.
        
        Parameters:
            state: torch.Tensor with shape (1, state_dim)
        
        Returns:
            If in random branch and a mapping exists, returns the function name (string) of the selected action.
            Otherwise returns the action index selected via q-values.
        """
        self.steps_done += 1
        if random.random() < self.epsilon:
            rand_index = random.randrange(self.action_dim)
            # Log the raw random index
            rl_logger.debug(f"DDQN would have selected random action index: {rand_index}")
            # Attempt to map the random index to a valid background agent function
            if self._action_mapping:
                mapped_idx = rand_index % len(self._action_mapping)
                action_name = self._action_mapping.get(mapped_idx, None)
                if action_name and action_name in self.background_agent_methods:
                    rl_logger.debug(f"DDQN selected random action mapped to function: {action_name}")
                    return action_name
                else:
                    rl_logger.debug(f"Random action index {rand_index} does not map to a valid function.")
                    # Return a safe fallback (could also be None or a specific default function)
                    return None
            else:
                rl_logger.debug(f"No action mapping available; returning raw index {rand_index}.")
                return rand_index
        else:
            with torch.no_grad():
                q_values = self.policy_net(state.to(self.device))
                action_val = q_values.argmax(dim=1).item()
                return action_val

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
        
# Dummy definitions for the classes/functions used above (replace with actual implementations)
class DDQNNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DDQNNetwork, self).__init__()
        self.fc = torch.nn.Linear(state_dim, action_dim)
    def forward(self, x):
        return self.fc(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
    def __len__(self):
        return len(self.buffer)
    def sample(self, batch_size):
        # Dummy implementation; replace with actual sampling logic
        return [], [], [], [], []