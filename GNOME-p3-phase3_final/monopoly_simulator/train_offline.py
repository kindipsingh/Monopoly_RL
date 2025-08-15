
import os
import sys
import torch
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monopoly_simulator.ddqn_decision_agent import DDQNAgent
from monopoly_simulator.replay_buffer_module import ReplayBuffer
import torch.nn.functional as F

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
train_logger = logging.getLogger(__name__)
log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'offline_training.log')
file_handler = logging.FileHandler(log_file, mode='w')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
train_logger.addHandler(file_handler)

def train_offline(replay_buffer_path, model_path, num_epochs=100, batch_size=64, learning_rate=1e-4, test_size=0.2, random_state=42):
    """
    Trains a DDQN model offline using a saved replay buffer.

    Args:
        replay_buffer_path (str): Path to the replay buffer file.
        model_path (str): Path to save the trained model.
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for the optimizer.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed for the random number generator.
    """
    train_logger.info("Starting offline training...")

    # Load the replay buffer
    if not os.path.exists(replay_buffer_path):
        train_logger.error(f"Replay buffer not found at {replay_buffer_path}")
        return

    replay_buffer = ReplayBuffer(capacity=10000)  # Capacity doesn't matter here
    replay_buffer.load_from_file(replay_buffer_path)
    train_logger.info(f"Replay buffer loaded with {len(replay_buffer.buffer)} transitions.")

    # Shuffle the data
    np.random.shuffle(replay_buffer.buffer)

    # Split data into training and validation sets
    train_data, val_data = train_test_split(replay_buffer.buffer, test_size=test_size, random_state=random_state)
    train_logger.info(f"Training data size: {len(train_data)}")
    train_logger.info(f"Validation data size: {len(val_data)}")

    # Create separate replay buffers for training and validation
    train_buffer = ReplayBuffer(capacity=len(train_data))
    train_buffer.buffer = train_data
    val_buffer = ReplayBuffer(capacity=len(val_data))
    val_buffer.buffer = val_data

    # Initialize the DDQN agent
    # The state and action dimensions need to be known. Let's assume they are stored somewhere or can be inferred.
    # For now, I'll use placeholders. You should replace these with the correct values.
    state_dim = 240  # Example state dimension
    action_dim = 2934  # Example action dimension
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DDQNAgent(state_dim=state_dim, action_dim=action_dim, lr=learning_rate, batch_size=batch_size)
    
    # Load model if it exists
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        agent.policy_net.load_state_dict(checkpoint['policy_net'])
        agent.target_net.load_state_dict(checkpoint['target_net'])
        train_logger.info(f"Loaded existing model from {model_path}")

        agent.optimizer = torch.optim.Adam(agent.policy_net.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        np.random.shuffle(train_data)
        
        # Get the full training dataset and perform robust filtering
        transitions = train_buffer.sample(len(train_data))
        
        clean_transitions = [t[:5] for t in transitions if t[0].size == 240 and t[3].size == 240]
        
        if len(clean_transitions) < len(transitions):
            train_logger.info(f"Epoch {epoch+1}/{num_epochs}, removed {len(transitions) - len(clean_transitions)} invalid transitions from batch.")

        if not clean_transitions:
            train_logger.warning(f"Epoch {epoch+1}/{num_epochs}, no valid transitions found, skipping.")
            continue
            
        batch = list(zip(*clean_transitions))
        states, actions, rewards, next_states, dones = batch

        state_batch = torch.FloatTensor(np.array([s.reshape(240) for s in states])).to(device)
        processed_actions = []
        for a in actions:
            if isinstance(a, (np.ndarray, list)) and len(a) > 1:
                processed_actions.append(int(np.argmax(a)))  # one-hot -> index
            else:
                processed_actions.append(int(a))  # already index

        action_batch = torch.LongTensor(processed_actions).unsqueeze(1).to(device)
        reward_batch = torch.FloatTensor([float(r[0]) if isinstance(r, (np.ndarray, list)) else float(r) for r in rewards]).to(device)
        next_state_batch = torch.FloatTensor(np.array([ns.reshape(240) for ns in next_states])).to(device)
        # Ensure done flags are single booleans, taking the first element if it's an array
        processed_dones = [bool(d[0]) if isinstance(d, np.ndarray) else bool(d) for d in dones]
        done_batch = torch.tensor(processed_dones, dtype=torch.bool, device=device)
        
        print("state_batch shape:", state_batch.shape)  # should be [B, 240]
        print("policy_net output shape:", agent.policy_net(state_batch).shape)  # [B, 2934]
        print("action_batch shape:", action_batch.shape)  # should be [B, 1]
        print("action_batch[:5]:", action_batch[:5])
  
        # Compute Q(s_t, a)
        q_values = agent.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_q_values = agent.target_net(next_state_batch).max(1)[0].detach()
        expected_q_values = reward_batch + (agent.gamma * next_q_values * (~done_batch))

        # Compute loss
        loss = F.smooth_l1_loss(q_values, expected_q_values.unsqueeze(1))

        # Optimize the model
        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()

        train_logger.info(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {loss.item():.4f}")

        # Validation step
        val_transitions = val_buffer.sample(len(val_data))

        clean_val_transitions = [t[:5] for t in val_transitions if t[0].size == 240 and t[3].size == 240]

        if not clean_val_transitions:
            train_logger.warning(f"Epoch {epoch+1}/{num_epochs}, no valid validation transitions found, skipping.")
            continue

        val_batch = list(zip(*clean_val_transitions))
        val_states, val_actions, val_rewards, val_next_states, val_dones = val_batch

        val_state_batch = torch.FloatTensor(np.array([s.reshape(240) for s in val_states])).to(device)
        processed_val_actions = []
        for a in val_actions:
            if isinstance(a, (np.ndarray, list)) and len(a) > 1:
                processed_val_actions.append(int(np.argmax(a)))  # one-hot -> index
            else:
                processed_val_actions.append(int(a))  # already index
        val_action_batch = torch.LongTensor(processed_val_actions).unsqueeze(1).to(device)
        val_reward_batch = torch.FloatTensor([float(r[0]) if isinstance(r, (np.ndarray, list)) else float(r) for r in val_rewards]).to(device)
        val_next_state_batch = torch.FloatTensor(np.array([ns.reshape(240) for ns in val_next_states])).to(device)
        # Ensure done flags are single booleans, taking the first element if it's an array
        processed_val_dones = [bool(d[0]) if isinstance(d, np.ndarray) else bool(d) for d in val_dones]
        val_done_batch = torch.tensor(processed_val_dones, dtype=torch.bool, device=device)

        
        val_q_values = agent.policy_net(val_state_batch).gather(1, val_action_batch)
        val_next_q_values = agent.target_net(val_next_state_batch).max(1)[0].detach()
        val_expected_q_values = val_reward_batch + (agent.gamma * val_next_q_values * (~val_done_batch))
        
        val_loss = F.smooth_l1_loss(val_q_values, val_expected_q_values.unsqueeze(1))
        train_logger.info(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss.item():.4f}")


    # Save the trained model
    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    torch.save({
        'policy_net': agent.policy_net.state_dict(),
        'target_net': agent.target_net.state_dict(),
        'optimizer': agent.optimizer.state_dict()
    }, model_path)
    train_logger.info(f"Trained model saved to {model_path}")

if __name__ == "__main__":
    # These paths should be configured as needed
    replay_buffer_file = "replay_buffer.pkl"
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    trained_model_file = os.path.join(project_root, "models", "ddqn_model_final.pth")
    
    # Check if the replay buffer exists in the expected location
    if not os.path.exists(replay_buffer_file):
        # If not, try to locate it in the parent directory, assuming a standard project structure
        alt_replay_buffer_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), replay_buffer_file)
        if os.path.exists(alt_replay_buffer_file):
            replay_buffer_file = alt_replay_buffer_file
        else:
            print(f"Error: Replay buffer not found at {replay_buffer_file} or {alt_replay_buffer_file}")
            sys.exit(1)

    train_offline(replay_buffer_file, trained_model_file)
