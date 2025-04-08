
import os
import sys
import argparse
import logging
import numpy as np
import torch
import pickle
from tqdm import tqdm
from collections import deque
import random
import time

# Import required modules from the Monopoly simulator
from monopoly_simulator import gameplay_socket_phase3 as gameplay
from monopoly_simulator import background_agent_v3_1
from monopoly_simulator.agent import Agent
from monopoly_simulator.ddqn_decision_agent import DDQNDecisionAgent
from monopoly_simulator.logging_info import log_file_create
from monopoly_simulator.monopoly_state_encoder import MonopolyStateEncoder
from monopoly_simulator.action_encoding import ActionEncoder
from monopoly_simulator.action_mapping_builder import create_action_mask
from monopoly_simulator.ddqnn import DDQNNetwork

# Configure logging
current_file = os.path.abspath(__file__)
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(current_file)))
log_dir = os.path.join(base_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'train_ddqn_agent.log')

logger = logging.getLogger('train_ddqn_agent')
logger.setLevel(logging.INFO)

if not logger.handlers:
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

class ActionTracker:
    """
    Tracks actions taken by the DDQN agent during gameplay to enable proper experience collection.
    """
    def __init__(self):
        self.last_state = None
        self.last_action_idx = None
        self.last_player = None
        self.current_reward = 0.0
        self.episode_rewards = []
        self.total_actions = 0
        
    def reset(self):
        self.last_state = None
        self.last_action_idx = None
        self.last_player = None
        self.current_reward = 0.0
        self.episode_rewards.append(self.current_reward)
        self.current_reward = 0.0

def create_custom_network(state_dim=240, action_dim=2950, hidden_sizes=[1024, 512]):
    """
    Create a custom DDQNNetwork with specified architecture.
    
    Args:
        state_dim: Dimension of the state vector
        action_dim: Dimension of the action space
        hidden_sizes: List of hidden layer sizes
    
    Returns:
        A DDQNNetwork instance with the specified architecture
    """
    network = DDQNNetwork(state_dim=state_dim, action_dim=action_dim)
    logger.info(f"Created custom DDQNNetwork with architecture:")
    logger.info(f"  Input dim: {state_dim}")
    logger.info(f"  Hidden layers: {hidden_sizes}")
    logger.info(f"  Output dim: {action_dim}")
    return network

def state_logger_factory(replay_buffer, ddqn_agent, encoder, action_tracker):
    """
    Creates a state logger function that captures state transitions for the replay buffer.
    
    Args:
        replay_buffer: The replay buffer to store transitions
        ddqn_agent: The DDQN agent being trained
        encoder: The state encoder to convert game states to vectors
        action_tracker: Tracks actions for proper experience collection
    
    Returns:
        A function that can be passed to simulate_game_instance
    """
    def state_logger(game_elements):
        # Find our agent's player
        current_player = None
        for player in game_elements['players']:
            if player.player_name == 'player_1':  # Assuming DDQN agent is player_1
                current_player = player
                break
        
        if not current_player:
            return
            
        # Encode current state
        current_state_vector = encoder.encode_state(game_elements)
        
        # If we have a previous state and action, we can create a transition
        if action_tracker.last_state is not None and action_tracker.last_action_idx is not None:
            # Calculate reward based on the change in game state
            reward = ddqn_agent._calculate_reward(current_player, game_elements)
            action_tracker.current_reward += reward
            
            # Check if episode is done
            done = ddqn_agent._is_episode_done(game_elements)
            
            # Add transition to replay buffer
            replay_buffer.push(
                action_tracker.last_state.numpy(),
                action_tracker.last_action_idx,
                reward,
                current_state_vector.numpy(),
                done
            )
            
            action_tracker.total_actions += 1
            
            if action_tracker.total_actions % 100 == 0:
                logger.debug(f"Added {action_tracker.total_actions} transitions to replay buffer")
            
            # If training mode is enabled, periodically train the agent
            if ddqn_agent.training_mode and len(replay_buffer) >= ddqn_agent.ddqn_agent.batch_size:
                loss = ddqn_agent.train()
                if loss is not None and action_tracker.total_actions % 100 == 0:
                    logger.debug(f"Training loss: {loss:.6f}")
            
            # If the episode is done, reset the action tracker
            if done:
                action_tracker.reset()
        
        # Update last state for next transition
        action_tracker.last_state = current_state_vector
        action_tracker.last_player = current_player
        
    return state_logger

def action_logger_factory(ddqn_agent, action_tracker):
    """
    Creates an action logger function that captures actions taken by the DDQN agent.
    
    Args:
        ddqn_agent: The DDQN agent being trained
        action_tracker: Tracks actions for proper experience collection
    
    Returns:
        A function that can be called to log actions
    """
    def action_logger(player, current_gameboard, action_idx, game_phase):
        if player.player_name == 'player_1':  # Only track actions for our DDQN agent
            action_tracker.last_action_idx = action_idx
            logger.debug(f"Logged action {action_idx} for player {player.player_name} in phase {game_phase}")
    
    return action_logger

def patch_ddqn_decision_agent(ddqn_agent, action_logger):
    """
    Patches the DDQN decision agent to log actions for experience collection.
    
    Args:
        ddqn_agent: The DDQN agent to patch
        action_logger: The action logger function
    """
    original_make_decision = ddqn_agent._make_decision
    
    def patched_make_decision(player, current_gameboard, game_phase):
        try:
            # Create state vector and tensor
            state_vector = ddqn_agent.state_encoder.encode_state(current_gameboard)
            state_tensor = state_vector.to(ddqn_agent.device)
            
            # Build the full action mapping
            action_encoder = ActionEncoder()
            full_mapping = action_encoder.build_full_action_mapping(player, current_gameboard)
            
            # Get valid actions
            action_mask = create_action_mask(player, current_gameboard, game_phase)
            mask_tensor = torch.BoolTensor(action_mask).to(ddqn_agent.device)
            
            # Find indices of valid actions
            valid_action_indices = np.where(action_mask)[0]
            
            if len(valid_action_indices) == 0:
                logger.warning(f"No valid actions found for player {player.player_name} in phase {game_phase}")
                # Default to skip_turn or conclude_actions based on game phase
                if game_phase == "post_roll":
                    action_name = "concluded_actions"
                else:
                    action_name = "skip_turn"
                logger.info(f"Defaulting to {action_name}")
                return (action_name, {})
            
            # Select action using epsilon-greedy
            if ddqn_agent.training_mode and random.random() < ddqn_agent.ddqn_agent.epsilon:
                # Random action from valid actions
                action_idx = np.random.choice(valid_action_indices)
                logger.debug(f"Selected random action {action_idx} (epsilon={ddqn_agent.ddqn_agent.epsilon:.2f})")
            else:
                # Use DDQN to select action
                with torch.no_grad():
                    q_values = ddqn_agent.ddqn_agent.policy_net(state_tensor)
                    # Mask invalid actions by setting their Q-values to -inf
                    masked_q_values = q_values.clone()
                    masked_q_values[0, ~mask_tensor] = float('-inf')
                    # Select action with highest Q-value
                    action_idx = masked_q_values.argmax(dim=1).item()
                    logger.debug(f"Selected action {action_idx} with Q-value {q_values[0, action_idx]:.4f}")
            
            # Log the action
            action_logger(player, current_gameboard, action_idx, game_phase)
            
            # Decode the selected action
            mapping = action_encoder.decode_action(player, current_gameboard, action_idx)
            action_name = mapping.get("action")
            parameters = mapping.get("parameters", {})
            
            logger.debug(f"Decoded action: {action_name} with parameters: {parameters}")
            
            # Process parameters for special cases (same as original _make_decision)
            if "to_player" in parameters and isinstance(parameters["to_player"], str):
                target_name = parameters["to_player"]
                target_player = None
                for p in current_gameboard.get("players", []):
                    if getattr(p, "player_name", None) == target_name:
                        target_player = p
                        break
                if target_player is None:
                    logger.error(f"Target player '{target_name}' not found in the current gameboard.")
                    if game_phase == "post_roll":
                        return ("concluded_actions", {})
                    else:
                        return ("skip_turn", {})
                parameters["to_player"] = target_player
            
            # For trade offers, convert the offer object
            if action_name == "make_trade_offer":
                offer = parameters.get("offer")
                if offer is not None:
                    # Copy the offer object to ensure original data is preserved during conversion
                    raw_offer = offer.copy()
                    logger.debug(f"Trade offer before conversion: {raw_offer}")
                    # Convert cash values
                    from monopoly_simulator import action_validator
                    converted_offer = action_validator.convert_cash_values(raw_offer, current_gameboard, logger)
                    # Convert property names to objects
                    converted_offer = action_validator.convert_offer_properties(converted_offer, current_gameboard, logger)
                    parameters["offer"] = converted_offer
                    logger.debug(f"Trade offer after conversion: {parameters['offer']}")
            
            # Validate and convert asset parameters for property-related actions
            if action_name == "sell_property":
                from monopoly_simulator import action_validator
                parameters = action_validator.validate_sell_property(parameters, current_gameboard, logger)
            elif action_name =="make_sell_property_offer":
                from monopoly_simulator import action_validator
                parameters = action_validator.validate_make_sell_property_offer(parameters, current_gameboard, logger)
            elif action_name == "sell_house_hotel":
                from monopoly_simulator import action_validator
                parameters = action_validator.validate_sell_house_hotel_asset(parameters, current_gameboard, logger)
            elif action_name in ["improve_property", "reverse_improve_property"]:
                from monopoly_simulator import action_validator
                parameters = action_validator.validate_improve_property(parameters, current_gameboard, logger)
            elif action_name == "free_mortgage":
                from monopoly_simulator import action_validator
                parameters = action_validator.validate_free_mortgage(parameters, current_gameboard, logger)
            elif action_name == "mortgage_property":
                from monopoly_simulator import action_validator
                parameters = action_validator.validate_free_mortgage(parameters, current_gameboard, logger)
            
            # Important: Replace "current_gameboard" string with the actual current_gameboard
            if "current_gameboard" in parameters and parameters["current_gameboard"] == "current_gameboard":
                parameters["current_gameboard"] = current_gameboard
            
            # Return the action name and parameters
            return (action_name, parameters)
        
        except Exception as e:
            logger.error(f"Error in patched_make_decision: {str(e)}", exc_info=True)
            # Return a safe default action based on game phase
            if game_phase == "post_roll":
                return ("concluded_actions", {})
            else:
                return ("skip_turn", {})
    
    # Replace the original method with our patched version
    ddqn_agent._make_decision = patched_make_decision
    logger.info("Patched DDQN decision agent to log actions")

def visualize_network_architecture(ddqn_agent, output_dir):
    """
    Visualize the network architecture and save to a file.
    
    Args:
        ddqn_agent: The DDQN agent containing the networks
        output_dir: Directory to save the visualization
    """
    try:
        import torch.nn as nn
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get the policy network
        policy_net = ddqn_agent.ddqn_agent.policy_net
        
        # Count parameters
        total_params = sum(p.numel() for p in policy_net.parameters())
        trainable_params = sum(p.numel() for p in policy_net.parameters() if p.requires_grad)
        
        # Create a visualization of the network architecture
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Define the layers
        layers = []
        layer_sizes = []
        
        # Input layer
        layers.append("Input")
        layer_sizes.append(policy_net.fc1.in_features)
        
        # Hidden layers
        for name, module in policy_net.named_children():
            if isinstance(module, nn.Linear):
                layers.append(f"{name}")
                layer_sizes.append(module.out_features)
        
        # Plot the architecture
        y_positions = np.arange(len(layers))
        ax.barh(y_positions, layer_sizes, align='center')
        ax.set_yticks(y_positions)
        ax.set_yticklabels(layers)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Number of Neurons')
        ax.set_title('DDQN Network Architecture')
        
        # Add parameter count
        plt.figtext(0.5, 0.01, f'Total Parameters: {total_params:,} (Trainable: {trainable_params:,})', 
                   ha='center', fontsize=10)
        
        # Save the figure
        output_path = os.path.join(output_dir, 'network_architecture.png')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Network architecture visualization saved to {output_path}")
        
        # Also save a text summary
        summary_path = os.path.join(output_dir, 'network_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("DDQN Network Architecture Summary\n")
            f.write("================================\n\n")
            f.write(f"Total Parameters: {total_params:,}\n")
            f.write(f"Trainable Parameters: {trainable_params:,}\n\n")
            
            f.write("Layer Structure:\n")
            for i, (layer, size) in enumerate(zip(layers, layer_sizes)):
                f.write(f"  {i}: {layer} - {size} neurons\n")
        
        logger.info(f"Network summary saved to {summary_path}")
        
    except ImportError as e:
        logger.warning(f"Could not visualize network architecture: {str(e)}")
    except Exception as e:
        logger.error(f"Error visualizing network architecture: {str(e)}", exc_info=True)

def train_agent(num_games=100, save_interval=10, learning_rate=1e-5, gamma=0.99, 
                batch_size=64, replay_capacity=10000, target_update_freq=5,
                epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995,
                custom_network=False, state_dim=240, action_dim=2950):
    """
    Train the DDQN agent by playing multiple games and collecting experiences.
    
    Args:
        num_games: Number of games to play for training
        save_interval: How often to save the model (in games)
        learning_rate: Learning rate for the optimizer
        gamma: Discount factor for future rewards
        batch_size: Batch size for training
        replay_capacity: Capacity of the replay buffer
        target_update_freq: How often to update the target network (in games)
        epsilon_start: Starting value for epsilon (exploration rate)
        epsilon_end: Minimum value for epsilon
        epsilon_decay: Decay rate for epsilon after each game
        custom_network: Whether to use a custom network architecture
        state_dim: Dimension of the state vector (if using custom network)
        action_dim: Dimension of the action space (if using custom network)
    """
    # Create output directories
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Initialize the DDQN agent
    ddqn_agent = DDQNDecisionAgent(
        name="DDQN_Player",
        lr=learning_rate,
        gamma=gamma,
        batch_size=batch_size,
        replay_capacity=replay_capacity,
        target_update_freq=target_update_freq,
        state_dim=state_dim,
        action_dim=action_dim
    )
    
    # If using custom network, replace the default networks
    if custom_network:
        logger.info("Using custom network architecture")
        # Create custom networks for policy and target
        policy_net = create_custom_network(state_dim, action_dim)
        target_net = create_custom_network(state_dim, action_dim)
        
        # Replace the networks in the DDQN agent
        ddqn_agent.ddqn_agent.policy_net = policy_net
        ddqn_agent.ddqn_agent.target_net = target_net
        
        # Update the target network to match the policy network
        ddqn_agent.update_target_network()
    
    # Visualize the network architecture
    visualize_network_architecture(ddqn_agent, models_dir)
    
    # Set epsilon parameters
    ddqn_agent.ddqn_agent.epsilon = epsilon_start
    ddqn_agent.ddqn_agent.epsilon_min = epsilon_end
    ddqn_agent.ddqn_agent.epsilon_decay = epsilon_decay
    
    # Set to training mode
    ddqn_agent.set_training_mode(True)
    
    # Create state encoder
    encoder = MonopolyStateEncoder()
    
    # Create action tracker
    action_tracker = ActionTracker()
    
    # Create action logger
    action_logger_func = action_logger_factory(ddqn_agent, action_tracker)
    
    # Patch the DDQN decision agent to log actions
    patch_ddqn_decision_agent(ddqn_agent, action_logger_func)
    
    # Load existing replay buffer if available
    replay_buffer_path = os.path.join(base_dir, "monopoly_simulator", "replay_buffer.pkl")
    ddqn_agent.load_replay_buffer(replay_buffer_path)
    
    # Create state logger
    game_state_logger = state_logger_factory(
        ddqn_agent.ddqn_agent.replay_buffer, 
        ddqn_agent,
        encoder,
        action_tracker
    )
    
    # Track metrics
    wins = 0
    losses = 0
    draws = 0
    training_losses = []
    
    # Main training loop
    for game_num in tqdm(range(num_games), desc="Training Games"):
        logger.info(f"Starting game {game_num+1}/{num_games}")
        
        # Reset action tracker for new game
        action_tracker.reset()
        
        # Play a game
        seed = np.random.randint(0, 10000)
        logger.info(f"Game seed: {seed}")
        
        # Set up background agents
        background_agent = Agent(**background_agent_v3_1.decision_agent_methods)
        
        # Play the game
        start_time = time.time()
        winner = gameplay.play_game_in_tournament_socket_phase3(
            game_seed=seed,
            agent1=ddqn_agent,
            agent2=background_agent_v3_1,
            agent3=background_agent_v3_1,
            agent4=background_agent_v3_1,
            state_logger=game_state_logger
        )
        end_time = time.time()
        
        # Update statistics
        if winner == 'player_1':
            wins += 1
        elif winner is None:
            draws += 1
        else:
            losses += 1
        
        # Calculate win rate
        win_rate = wins / (game_num + 1)
        game_duration = end_time - start_time
        
        logger.info(f"Game {game_num+1} complete in {game_duration:.1f}s. Winner: {winner}. Win rate: {win_rate:.2f}")
        logger.info(f"Epsilon: {ddqn_agent.ddqn_agent.epsilon:.4f}, Replay buffer size: {len(ddqn_agent.ddqn_agent.replay_buffer)}")
        
        # Perform additional training after each game
        if len(ddqn_agent.ddqn_agent.replay_buffer) >= ddqn_agent.ddqn_agent.batch_size:
            # Train for multiple batches after each game
            num_batches = min(10, len(ddqn_agent.ddqn_agent.replay_buffer) // ddqn_agent.ddqn_agent.batch_size)
            batch_losses = []
            for _ in range(num_batches):
                loss = ddqn_agent.train()
                if loss is not None:
                    batch_losses.append(loss)
            
            if batch_losses:
                avg_loss = sum(batch_losses) / len(batch_losses)
                training_losses.append(avg_loss)
                logger.info(f"Post-game training: avg loss over {num_batches} batches: {avg_loss:.6f}")
        
        # Decay epsilon
        if ddqn_agent.ddqn_agent.epsilon > ddqn_agent.ddqn_agent.epsilon_min:
            ddqn_agent.ddqn_agent.epsilon *= ddqn_agent.ddqn_agent.epsilon_decay
            logger.debug(f"Decayed epsilon to {ddqn_agent.ddqn_agent.epsilon:.4f}")
        
        # Update target network periodically
        if (game_num + 1) % target_update_freq == 0:
            ddqn_agent.update_target_network()
            logger.info(f"Updated target network after game {game_num+1}")
        
        # Save model periodically
        if (game_num + 1) % save_interval == 0:
            model_path = os.path.join(models_dir, f"ddqn_model_game_{game_num+1}.pth")
            ddqn_agent.save_model(model_path)
            logger.info(f"Saved model to {model_path}")
            
            # Also save the replay buffer
            ddqn_agent.persist_replay_buffer()
            logger.info(f"Saved replay buffer after game {game_num+1}")
            
            # Plot training losses if available
            if training_losses:
                try:
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(10, 6))
                    plt.plot(training_losses)
                    plt.title('Training Loss')
                    plt.xlabel('Training Batch')
                    plt.ylabel('Loss')
                    plt.grid(True)
                    loss_plot_path = os.path.join(models_dir, f"training_loss_game_{game_num+1}.png")
                    plt.savefig(loss_plot_path)
                    plt.close()
                    logger.info(f"Saved training loss plot to {loss_plot_path}")
                except Exception as e:
                    logger.warning(f"Could not plot training losses: {str(e)}")
    
    # Save final model
    final_model_path = os.path.join(models_dir, "ddqn_model_final.pth")
    ddqn_agent.save_model(final_model_path)
    logger.info(f"Training complete. Final model saved to {final_model_path}")
    
    # Save final replay buffer
    ddqn_agent.persist_replay_buffer()
    
    # Print final statistics
    logger.info(f"Training completed. Final statistics:")
    logger.info(f"  Games played: {num_games}")
    logger.info(f"  Wins: {wins} ({wins/num_games:.2f})")
    logger.info(f"  Losses: {losses} ({losses/num_games:.2f})")
    logger.info(f"  Draws: {draws} ({draws/num_games:.2f})")
    logger.info(f"  Final epsilon: {ddqn_agent.ddqn_agent.epsilon:.4f}")
    logger.info(f"  Replay buffer size: {len(ddqn_agent.ddqn_agent.replay_buffer)}")
    logger.info(f"  Total actions: {action_tracker.total_actions}")
    
    return ddqn_agent

def evaluate_agent(model_path, num_games=10):
    """
    Evaluate a trained DDQN agent by playing games without training.
    
    Args:
        model_path: Path to the saved model
        num_games: Number of games to play for evaluation
    """
    # Initialize the DDQN agent
    ddqn_agent = DDQNDecisionAgent(name="DDQN_Evaluator")
    
    # Load the trained model
    ddqn_agent.load_model(model_path)
    
    # Set to evaluation mode
    ddqn_agent.set_training_mode(False)
    
    # Create action tracker for statistics
    action_tracker = ActionTracker()
    
    # Create action logger
    action_logger_func = action_logger_factory(ddqn_agent, action_tracker)
    
    # Patch the DDQN decision agent to log actions
    patch_ddqn_decision_agent(ddqn_agent, action_logger_func)
    
    # Track metrics
    wins = 0
    losses = 0
    draws = 0
    game_durations = []
    
    # Evaluation loop
    for game_num in tqdm(range(num_games), desc="Evaluation Games"):
        logger.info(f"Starting evaluation game {game_num+1}/{num_games}")
        
        # Reset action tracker for new game
        action_tracker.reset()
        
        # Play a game
        seed = np.random.randint(0, 10000)
        logger.info(f"Game seed: {seed}")
        
        # Set up background agents
        background_agent = Agent(**background_agent_v3_1.decision_agent_methods)
        
        # Play the game
        start_time = time.time()
        winner = gameplay.play_game_in_tournament_socket_phase3(
            game_seed=seed,
            agent1=ddqn_agent,
            agent2=background_agent_v3_1,
            agent3=background_agent_v3_1,
            agent4=background_agent_v3_1
        )
        end_time = time.time()
        game_duration = end_time - start_time
        game_durations.append(game_duration)
        
        # Update statistics
        if winner == 'player_1':
            wins += 1
        elif winner is None:
            draws += 1
        else:
            losses += 1
        
        # Calculate win rate
        win_rate = wins / (game_num + 1)
        
        logger.info(f"Evaluation game {game_num+1} complete in {game_duration:.1f}s. Winner: {winner}. Win rate: {win_rate:.2f}")
    
    # Print final statistics
    avg_duration = sum(game_durations) / len(game_durations) if game_durations else 0
    logger.info(f"Evaluation completed. Final statistics:")
    logger.info(f"  Games played: {num_games}")
    logger.info(f"  Wins: {wins} ({wins/num_games:.2f})")
    logger.info(f"  Losses: {losses} ({losses/num_games:.2f})")
    logger.info(f"  Draws: {draws} ({draws/num_games:.2f})")
    logger.info(f"  Average game duration: {avg_duration:.1f}s")
    logger.info(f"  Total actions: {action_tracker.total_actions}")

def analyze_replay_buffer(buffer_path):
    """
    Analyze the contents of a replay buffer.
    
    Args:
        buffer_path: Path to the saved replay buffer
    """
    if not os.path.exists(buffer_path):
        logger.error(f"Replay buffer not found at {buffer_path}")
        return
    
    with open(buffer_path, "rb") as f:
        buffer = pickle.load(f)
    
    logger.info(f"Replay buffer analysis:")
    logger.info(f"  Size: {len(buffer)}")
    
    if len(buffer) == 0:
        logger.info("  Buffer is empty")
        return
    
    # Analyze rewards
    rewards = [transition[2] for transition in buffer]
    avg_reward = sum(rewards)