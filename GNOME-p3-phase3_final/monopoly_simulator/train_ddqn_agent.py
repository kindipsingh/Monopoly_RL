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
logger.setLevel(logging.DEBUG)
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
    """
    network = DDQNNetwork(state_dim=state_dim, action_dim=action_dim)
    logger.info("Created custom DDQNNetwork with architecture:")
    logger.info(f"  Input dim: {state_dim}")
    logger.info(f"  Hidden layers: {hidden_sizes}")
    logger.info(f"  Output dim: {action_dim}")
    return network

def train_agent(num_games=2, save_interval=1, learning_rate=1e-5, gamma=0.99,
                batch_size=64, replay_capacity=10000, target_update_freq=5,
                epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995,
                custom_network=False, state_dim=240, action_dim=2934):
    """
    Train the DDQN agent by playing multiple games and collecting experiences.
    All RL training and backprop is handled inside gameplay_socket_phase3.py.
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

    ddqn_agent_instance.ddqn_agent.epsilon = epsilon_start
    ddqn_agent_instance.ddqn_agent.epsilon_min = epsilon_end
    ddqn_agent_instance.ddqn_agent.epsilon_decay = epsilon_decay
    ddqn_agent_instance.set_training_mode(True)

    agent_combination_1 = [[background_agent_v3_1, background_agent_v3_1, ddqn_decision_agent, background_agent_v4_1]]

    wins = 0
    losses = 0
    draws = 0

    for game_num in tqdm(range(num_games), desc="Training Games"):
        logger.info(f"Starting game {game_num+1}/{num_games}")
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

        if (game_num + 1) % save_interval == 0:
            model_path = os.path.join(models_dir, f"ddqn_model_game_{game_num+1}.pth")
            ddqn_agent_instance.save_model(model_path)
            logger.info(f"Saved model to {model_path}")
            ddqn_agent_instance.persist_replay_buffer()
            logger.info(f"Saved replay buffer after game {game_num+1}")

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

    return ddqn_agent_instance

def evaluate_agent(model_path, num_games=10):
    """
    Evaluate a trained DDQN agent by playing games without training.
    """
    ddqn_agent_instance = DDQNDecisionAgent(name="DDQN_Evaluator")
    ddqn_agent_instance.load_model(model_path)
    ddqn_agent_instance.set_training_mode(False)

    agent_combination_1 = [[background_agent_v3_1, background_agent_v3_1, ddqn_decision_agent, background_agent_v4_1]]

    wins = 0
    losses = 0
    draws = 0
    game_durations = []

    for game_num in tqdm(range(num_games), desc="Evaluation Games"):
        logger.info(f"Starting evaluation game {game_num+1}/{num_games}")
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Evaluate DDQN Agent")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "evaluate"],
                        help="Execution mode: train or evaluate")
    parser.add_argument("--num_games", type=int, default=100,
                        help="Number of games for training")
    parser.add_argument("--eval_games", type=int, default=1,
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