import os
import sys
import argparse
import logging
import random
import time
from collections import deque

import numpy as np
import torch
import pickle
from tqdm import tqdm

# Import required modules from the Monopoly simulator
from monopoly_simulator import gameplay_socket_phase3 as gameplay
from monopoly_simulator import background_agent_v3_1, background_agent_v4_1
from monopoly_simulator.agent import Agent
from monopoly_simulator.ddqn_decision_agent import DDQNDecisionAgent
from monopoly_simulator import ddqn_decision_agent
from monopoly_simulator.logging_info import log_file_create
from monopoly_simulator.monopoly_state_encoder import MonopolyStateEncoder
from monopoly_simulator.action_encoding import ActionEncoder
from monopoly_simulator.action_mapping_builder import create_action_mask
from monopoly_simulator.ddqnn import DDQNNetwork

# Configure logging
current_file = os.path.abspath(__file__)
base_dir = os.path.dirname(os.path.dirname(current_file))
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


def create_custom_network(state_dim=240, action_dim=2934, hidden_sizes=[1024, 512]):
    """
    Create a custom DDQNNetwork with specified architecture.

    Args:
        state_dim (int): Dimension of the state vector.
        action_dim (int): Dimension of the action space.
        hidden_sizes (list): List of hidden layer sizes.

    Returns:
        DDQNNetwork: An instance with the specified architecture.
    """
    network = DDQNNetwork(state_dim=state_dim, action_dim=action_dim)
    logger.info("Created custom DDQNNetwork with architecture:")
    logger.info(f"  Input dim: {state_dim}")
    logger.info(f"  Hidden layers: {hidden_sizes}")
    logger.info(f"  Output dim: {action_dim}")
    return network


def patch_ddqn_decision_agent(ddqn_agent_instance, action_logger):
    """
    Patches the DDQN decision agent to log actions for experience collection.

    Args:
        ddqn_agent_instance (DDQNDecisionAgent): The DDQN agent instance to patch.
        action_logger (callable): The action logger callback function.
    """
    original_make_decision = ddqn_agent_instance._make_decision

    def patched_make_decision(player, current_gameboard, game_phase):
        try:
            # Create state vector and tensor
            state_vector = ddqn_agent_instance.state_encoder.encode_state(current_gameboard)
            state_tensor = state_vector.to(ddqn_agent_instance.device)

            # Build full action mapping
            action_encoder = ActionEncoder()
            full_mapping = action_encoder.build_full_action_mapping(player, current_gameboard)

            # Get valid actions using the mask
            action_mask = create_action_mask(player, current_gameboard, game_phase)
            mask_tensor = torch.BoolTensor(action_mask).to(ddqn_agent_instance.device)
            valid_action_indices = np.where(action_mask)[0]

            if len(valid_action_indices) == 0:
                logger.warning(f"No valid actions found for player {player.player_name} in phase {game_phase}")
                action_name = "concluded_actions" if game_phase == "post_roll" else "skip_turn"
                logger.info(f"Defaulting to {action_name}")
                return (action_name, {})

            # Select action using epsilon-greedy scheme
            if ddqn_agent_instance.training_mode and random.random() < ddqn_agent_instance.ddqn_agent.epsilon:
                action_idx = np.random.choice(valid_action_indices)
                logger.debug(f"Selected random action {action_idx} (epsilon={ddqn_agent_instance.ddqn_agent.epsilon:.2f})")
            else:
                with torch.no_grad():
                    q_values = ddqn_agent_instance.ddqn_agent.policy_net(state_tensor)
                    masked_q_values = q_values.clone()
                    masked_q_values[0, ~mask_tensor] = float('-inf')
                    action_idx = masked_q_values.argmax(dim=1).item()
                    logger.debug(f"Selected action {action_idx} with Q-value {q_values[0, action_idx]:.4f}")

            # Log the action
            action_logger(player, current_gameboard, action_idx, game_phase)

            # Decode the selected action
            mapping = action_encoder.decode_action(player, current_gameboard, action_idx)
            action_name = mapping.get("action")
            parameters = mapping.get("parameters", {})
            logger.debug(f"Decoded action: {action_name} with parameters: {parameters}")

            # Process special parameters such as "to_player"
            if "to_player" in parameters and isinstance(parameters["to_player"], str):
                target_name = parameters["to_player"]
                target_player = next((p for p in current_gameboard.get("players", [])
                                      if getattr(p, "player_name", None) == target_name), None)
                if target_player is None:
                    logger.error(f"Target player '{target_name}' not found in the current gameboard.")
                    return ("concluded_actions", {}) if game_phase == "post_roll" else ("skip_turn", {})
                parameters["to_player"] = target_player

            # Handle trade offers
            if action_name == "make_trade_offer":
                offer = parameters.get("offer")
                if offer is not None:
                    raw_offer = offer.copy()
                    logger.debug(f"Trade offer before conversion: {raw_offer}")
                    from monopoly_simulator import action_validator
                    converted_offer = action_validator.convert_cash_values(raw_offer, current_gameboard, logger)
                    converted_offer = action_validator.convert_offer_properties(converted_offer, current_gameboard, logger)
                    parameters["offer"] = converted_offer
                    logger.debug(f"Trade offer after conversion: {parameters['offer']}")

            # Validate property-related actions
            from monopoly_simulator import action_validator
            if action_name == "sell_property":
                parameters = action_validator.validate_sell_property(parameters, current_gameboard, logger)
            elif action_name == "make_sell_property_offer":
                parameters = action_validator.validate_make_sell_property_offer(parameters, current_gameboard, logger)
            elif action_name == "sell_house_hotel":
                parameters = action_validator.validate_sell_house_hotel_asset(parameters, current_gameboard, logger)
            elif action_name in ["improve_property", "reverse_improve_property"]:
                parameters = action_validator.validate_improve_property(parameters, current_gameboard, logger)
            elif action_name in ["free_mortgage", "mortgage_property"]:
                parameters = action_validator.validate_free_mortgage(parameters, current_gameboard, logger)

            # Replace marker strings if necessary
            if parameters.get("current_gameboard") == "current_gameboard":
                parameters["current_gameboard"] = current_gameboard

            return (action_name, parameters)

        except Exception as e:
            logger.error(f"Error in patched_make_decision: {str(e)}", exc_info=True)
            return ("concluded_actions", {}) if game_phase == "post_roll" else ("skip_turn", {})

    ddqn_agent_instance._make_decision = patched_make_decision
    logger.info("Patched DDQN decision agent to log actions")


class ActionTracker:
    """
    Tracks actions during game simulations.
    Records each action along with a timestamp, player identifier, action index, and game phase.
    """
    def __init__(self):
        self.total_actions = 0
        self.logged_actions = []

    def log(self, player, action_idx, game_phase):
        """
        Log an action performed by a player.
        """
        timestamp = time.time()
        player_id = getattr(player, "player_name", str(player))
        entry = {
            'timestamp': timestamp,
            'player': player_id,
            'action_idx': action_idx,
            'game_phase': game_phase
        }
        self.logged_actions.append(entry)
        self.total_actions += 1

    def reset(self):
        """Reset logged actions (typically for a new game)."""
        self.logged_actions = []

    def get_stats(self):
        """Return statistics about the logged actions."""
        return {
            'total_actions': self.total_actions,
            'actions': self.logged_actions
        }


def action_logger_factory(ddqn_agent_instance, action_tracker):
    """
    Return an action logger callback that logs actions to the provided ActionTracker.
    """
    def log_action(player, current_gameboard, action_idx, game_phase):
        action_tracker.log(player, action_idx, game_phase)
        action_logger = logging.getLogger("action_logger")
        player_id = getattr(player, "player_name", str(player))
        action_logger.info(f"Logged action for player {player_id}: action index {action_idx} during phase {game_phase}")
    return log_action


class StateLogger:
    """
    Logs state transitions during simulation.
    Logs transitions (state, action, reward, next_state, done) into the replay buffer and locally.
    """
    def __init__(self, replay_buffer, ddqn_agent_instance, encoder, action_tracker):
        self.replay_buffer = replay_buffer
        self.ddqn_agent = ddqn_agent_instance
        self.encoder = encoder
        self.action_tracker = action_tracker
        self.logged_transitions = []

    def log(self, state, action, reward, next_state, done):
        """
        Log a transition to the replay buffer and record it locally.
        """
        self.replay_buffer.add(state, action, reward, next_state, done)
        self.logged_transitions.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })
        state_logger = logging.getLogger("state_logger")
        state_logger.debug(f"Logged transition: action {action}, reward {reward}, done {done}")


def state_logger_factory(replay_buffer, ddqn_agent_instance, encoder, action_tracker):
    """
    Factory function returning a state logger callable.
    """
    state_logger_instance = StateLogger(replay_buffer, ddqn_agent_instance, encoder, action_tracker)
    return state_logger_instance.log


def train_agent(num_games=2, save_interval=1, learning_rate=1e-5, gamma=0.99,
                batch_size=64, replay_capacity=10000, target_update_freq=5,
                epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995,
                custom_network=False, state_dim=240, action_dim=2934):
    """
    Train the DDQN agent by playing multiple games and collecting experiences.

    Args:
        num_games (int): Number of games for training.
        save_interval (int): Interval (in games) for saving the model.
        learning_rate (float): Learning rate for the optimizer.
        gamma (float): Discount factor for future rewards.
        batch_size (int): Batch size for training.
        replay_capacity (int): Capacity of the replay buffer.
        target_update_freq (int): Frequency (in games) to update the target network.
        epsilon_start (float): Starting value for epsilon (exploration rate).
        epsilon_end (float): Minimum epsilon value.
        epsilon_decay (float): Decay rate for epsilon.
        custom_network (bool): Whether to use a custom network architecture.
        state_dim (int): Dimension of the state vector.
        action_dim (int): Dimension of the action space.
    
    Returns:
        DDQNDecisionAgent: The trained agent.
    """
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    ddqn_agent_instance = DDQNDecisionAgent(
        name="DDQN_Player",
        lr=learning_rate,
        gamma=gamma,
        batch_size=batch_size,
        replay_capacity=replay_capacity,
        target_update_freq=target_update_freq,
        state_dim=state_dim,
        action_dim=action_dim
    )

    if custom_network:
        logger.info("Using custom network architecture")
        policy_net = create_custom_network(state_dim, action_dim)
        target_net = create_custom_network(state_dim, action_dim)
        ddqn_agent_instance.ddqn_agent.policy_net = policy_net
        ddqn_agent_instance.ddqn_agent.target_net = target_net
        ddqn_agent_instance.update_target_network()

    try:
        visualize_network_architecture(ddqn_agent_instance, models_dir)
    except Exception as e:
        logger.warning(f"Visualization skipped: {str(e)}")

    ddqn_agent_instance.ddqn_agent.epsilon = epsilon_start
    ddqn_agent_instance.ddqn_agent.epsilon_min = epsilon_end
    ddqn_agent_instance.ddqn_agent.epsilon_decay = epsilon_decay
    ddqn_agent_instance.set_training_mode(True)

    encoder = MonopolyStateEncoder()
    action_tracker = ActionTracker()
    action_logger_func = action_logger_factory(ddqn_agent_instance, action_tracker)
    patch_ddqn_decision_agent(ddqn_agent_instance, action_logger_func)

    # Load previous replay buffers if available
    import glob
    tournament_dir = os.path.join(base_dir, "single_tournament")
    replay_files = glob.glob(os.path.join(tournament_dir, "replay_buffer_seed_*.pkl"))
    combined_buffer = []
    combined_episode_rewards = []

    if replay_files:
        logger.info(f"Found {len(replay_files)} replay buffer files in {tournament_dir}")
        for rf in replay_files:
            logger.info(f"Loading replay buffer file: {rf}")
            with open(rf, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                combined_buffer.extend(data.get('buffer', []))
                combined_episode_rewards.extend(data.get('episode_rewards', []))
            else:
                combined_buffer.extend(data)
        ddqn_agent_instance.ddqn_agent.replay_buffer.buffer = combined_buffer
        ddqn_agent_instance.ddqn_agent.replay_buffer.episode_rewards = combined_episode_rewards
        ddqn_agent_instance.ddqn_agent.replay_buffer.position = len(combined_buffer) % ddqn_agent_instance.ddqn_agent.replay_buffer.capacity
        logger.info(f"Combined replay buffer loaded from {len(replay_files)} files with {len(combined_buffer)} transitions")
    else:
        logger.info(f"No replay buffer file found in {tournament_dir}")

    game_state_logger = state_logger_factory(
        ddqn_agent_instance.ddqn_agent.replay_buffer, 
        ddqn_agent_instance,
        encoder,
        action_tracker
    )

    # Define agent combinations exactly as in the test harness
    agent_combination_1 = [[background_agent_v3_1, background_agent_v3_1, ddqn_decision_agent, background_agent_v4_1]]

    wins = 0
    losses = 0
    draws = 0
    training_losses = []

    for game_num in tqdm(range(num_games), desc="Training Games"):
        logger.info(f"Starting game {game_num+1}/{num_games}")
        action_tracker.reset()
        seed = np.random.randint(0, 10000)
        logger.info(f"Game seed: {seed}")

        start_time = time.time()
        winner = gameplay.play_game_in_tournament_socket_phase3(
            game_seed=seed,
            agent1=agent_combination_1[0][0],
            agent2=agent_combination_1[0][1],
            agent3=agent_combination_1[0][2],
            agent4=agent_combination_1[0][3]
        )
        end_time = time.time()

        if winner == 'player_3':
            wins += 1
        elif winner is None:
            draws += 1
        else:
            losses += 1

        win_rate = wins / (game_num + 1)
        game_duration = end_time - start_time
        logger.info(f"Game {game_num+1} complete in {game_duration:.1f}s. Winner: {winner}. Win rate: {win_rate:.2f}")
        logger.info(f"Epsilon: {ddqn_agent_instance.ddqn_agent.epsilon:.4f}, Replay buffer size: {len(ddqn_agent_instance.ddqn_agent.replay_buffer)}")

        if len(ddqn_agent_instance.ddqn_agent.replay_buffer) >= ddqn_agent_instance.ddqn_agent.batch_size:
            num_batches = min(10, len(ddqn_agent_instance.ddqn_agent.replay_buffer) // ddqn_agent_instance.ddqn_agent.batch_size)
            batch_losses = []
            for _ in range(num_batches):
                loss = ddqn_agent_instance.train()
                if loss is not None:
                    batch_losses.append(loss)
            if batch_losses:
                avg_loss = sum(batch_losses) / len(batch_losses)
                training_losses.append(avg_loss)
                logger.info(f"Post-game training: avg loss over {num_batches} batches: {avg_loss:.6f}")

        if ddqn_agent_instance.ddqn_agent.epsilon > ddqn_agent_instance.ddqn_agent.epsilon_min:
            ddqn_agent_instance.ddqn_agent.epsilon *= ddqn_agent_instance.ddqn_agent.epsilon_decay
            logger.debug(f"Decayed epsilon to {ddqn_agent_instance.ddqn_agent.epsilon:.4f}")

        if (game_num + 1) % target_update_freq == 0:
            ddqn_agent_instance.update_target_network()
            logger.info(f"Updated target network after game {game_num+1}")

        if (game_num + 1) % save_interval == 0:
            model_path = os.path.join(models_dir, f"ddqn_model_game_{game_num+1}.pth")
            ddqn_agent_instance.save_model(model_path)
            logger.info(f"Saved model to {model_path}")
            ddqn_agent_instance.persist_replay_buffer()
            logger.info(f"Saved replay buffer after game {game_num+1}")
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

    final_model_path = os.path.join(models_dir, "ddqn_model_final.pth")
    ddqn_agent_instance.save_model(final_model_path)
    logger.info(f"Training complete. Final model saved to {final_model_path}")
    ddqn_agent_instance.persist_replay_buffer()

    logger.info("Training completed. Final statistics:")
    logger.info(f"  Games played: {num_games}")
    logger.info(f"  Wins: {wins} ({wins/num_games:.2f})")
    logger.info(f"  Losses: {losses} ({losses/num_games:.2f})")
    logger.info(f"  Draws: {draws} ({draws/num_games:.2f})")
    logger.info(f"  Final epsilon: {ddqn_agent_instance.ddqn_agent.epsilon:.4f}")
    logger.info(f"  Replay buffer size: {len(ddqn_agent_instance.ddqn_agent.replay_buffer)}")
    logger.info(f"  Total actions: {action_tracker.total_actions}")

    return ddqn_agent_instance


def evaluate_agent(model_path, num_games=10):
    """
    Evaluate a trained DDQN agent by playing games without training.

    Args:
        model_path (str): Path to the saved model.
        num_games (int): Number of games for evaluation.
    """
    ddqn_agent_instance = DDQNDecisionAgent(name="DDQN_Evaluator")
    ddqn_agent_instance.load_model(model_path)
    ddqn_agent_instance.set_training_mode(False)
    action_tracker = ActionTracker()
    action_logger_func = action_logger_factory(ddqn_agent_instance, action_tracker)
    patch_ddqn_decision_agent(ddqn_agent_instance, action_logger_func)

    # Define agent combination for evaluation as in the test harness
    agent_combination_1 = [[background_agent_v3_1, background_agent_v3_1, ddqn_decision_agent, background_agent_v4_1]]

    wins = 0
    losses = 0
    draws = 0
    game_durations = []

    for game_num in tqdm(range(num_games), desc="Evaluation Games"):
        logger.info(f"Starting evaluation game {game_num+1}/{num_games}")
        action_tracker.reset()
        seed = np.random.randint(0, 10000)
        logger.info(f"Game seed: {seed}")

        background_agent = Agent(**background_agent_v3_1.decision_agent_methods)
        start_time = time.time()
        winner = gameplay.play_game_in_tournament_socket_phase3(
            game_seed=seed,
            agent1=agent_combination_1[0][0],
            agent2=agent_combination_1[0][1],
            agent3=agent_combination_1[0][2],
            agent4=agent_combination_1[0][3]
        )
        end_time = time.time()
        duration = end_time - start_time
        game_durations.append(duration)

        if winner == 'player_3':
            wins += 1
        elif winner is None:
            draws += 1
        else:
            losses += 1

        win_rate = wins / (game_num + 1)
        logger.info(f"Evaluation game {game_num+1} complete in {duration:.1f}s. Winner: {winner}. Win rate: {win_rate:.2f}")

    avg_duration = sum(game_durations) / len(game_durations) if game_durations else 0
    logger.info("Evaluation completed. Final statistics:")
    logger.info(f"  Games played: {num_games}")
    logger.info(f"  Wins: {wins} ({wins/num_games:.2f})")
    logger.info(f"  Losses: {losses} ({losses/num_games:.2f})")
    logger.info(f"  Draws: {draws} ({draws/num_games:.2f})")
    logger.info(f"  Average game duration: {avg_duration:.1f}s")
    logger.info(f"  Total actions: {action_tracker.total_actions}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Evaluate DDQN Agent")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "evaluate"],
                        help="Execution mode: train or evaluate")
    parser.add_argument("--num_games", type=int, default=100,
                        help="Number of games for training")
    parser.add_argument("--eval_games", type=int, default=10,
                        help="Number of games for evaluation")
    parser.add_argument("--save_interval", type=int, default=10,
                        help="Interval (in games) to save the model")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate for training")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Training batch size")
    parser.add_argument("--replay_capacity", type=int, default=10000,
                        help="Capacity of the replay buffer")
    parser.add_argument("--target_update_freq", type=int, default=5,
                        help="Frequency (in games) to update target network")
    parser.add_argument("--epsilon_start", type=float, default=1.0,
                        help="Starting epsilon for exploration")
    parser.add_argument("--epsilon_end", type=float, default=0.1,
                        help="Minimum epsilon")
    parser.add_argument("--epsilon_decay", type=float, default=0.995,
                        help="Epsilon decay rate")
    parser.add_argument("--custom_network", action="store_true",
                        help="Use custom network architecture")
    args = parser.parse_args()

    if args.mode == "train":
        trained_agent = train_agent(
            num_games=args.num_games,
            save_interval=args.save_interval,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            batch_size=args.batch_size,
            replay_capacity=args.replay_capacity,
            target_update_freq=args.target_update_freq,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay=args.epsilon_decay,
            custom_network=args.custom_network
        )
        evaluate_agent(os.path.join(base_dir, "models", "ddqn_model_final.pth"), num_games=args.eval_games)
    elif args.mode == "evaluate":
        evaluate_agent(os.path.join(base_dir, "models", "ddqn_model_final.pth"), num_games=args.eval_games)