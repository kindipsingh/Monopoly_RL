import torch
import numpy as np
import random
import os
from typing import Dict, List, Tuple, Any
from monopoly_simulator.agent import Agent
from monopoly_simulator import action_choices
from monopoly_simulator.action_encoding import ActionEncoder
from monopoly_simulator.monopoly_state_encoder import MonopolyStateEncoder
from monopoly_simulator.logging_info import log_file_create
from monopoly_simulator.ddqnn import DDQNNetwork
from monopoly_simulator.training_ddqnn import DDQNAgent
import logging

# Set up the RL agent logger
rl_logger = logging.getLogger('rl_agent_logs')
rl_logger.setLevel(logging.DEBUG)

# Create a file handler
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'rl_agent.log')
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create a formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
rl_logger.addHandler(file_handler)
rl_logger.addHandler(console_handler)

# Keep the original logger for compatibility
logger = logging.getLogger('monopoly_simulator.logging_info.ddqn_decision_agent')

# Global instance of the DDQN agent that will be used by the standalone functions
global_ddqn_agent = None

class DDQNDecisionAgent(Agent):
    """
    A wrapper class that implements the decision agent interface for the Monopoly simulator
    using a DDQN (Double Deep Q-Network) agent for decision making.
    
    This agent uses:
    - MonopolyStateEncoder to encode the game state into a 240-dimensional vector
    - DDQN network to predict Q-values for each possible action
    - ActionEncoder to decode the selected action index into a valid game action
    """
    def __init__(self, name="DDQN Agent", state_dim=240, action_dim=2922, 
                 lr=1e-5, gamma=0.9999, batch_size=128, 
                 replay_capacity=10000, target_update_freq=500):
        """
        Initialize the DDQN Decision Agent.
        
        Parameters:
            name (str): The name of the agent
            state_dim (int): Dimension of the state space (from MonopolyStateEncoder)
            action_dim (int): Dimension of the action space (from ActionEncoder)
            lr (float): Learning rate for the optimizer
            gamma (float): Discount factor for future rewards
            batch_size (int): Batch size for training
            replay_capacity (int): Capacity of the replay buffer
            target_update_freq (int): Frequency to update the target network
        """
        # Initialize the Agent superclass with the decision methods
        super().__init__(
            handle_negative_cash_balance=self.handle_negative_cash_balance,
            make_pre_roll_move=self.make_pre_roll_move,
            make_out_of_turn_move=self.make_out_of_turn_move,
            make_post_roll_move=self.make_post_roll_move,
            make_buy_property_decision=self.make_buy_property_decision,
            make_bid=self.make_bid,
            type="decision_agent_methods"
        )
        
        self.name = name
        rl_logger.info(f"Initializing {self.name} with state_dim={state_dim}, action_dim={action_dim}")
        
        # Initialize the DDQN agent
        self.ddqn_agent = DDQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=lr,
            gamma=gamma,
            batch_size=batch_size,
            replay_capacity=replay_capacity,
            target_update_freq=target_update_freq
        )
        
        # Initialize the state encoder and action encoder
        self.state_encoder = MonopolyStateEncoder()
        self.action_encoder = ActionEncoder()
        
        # For tracking game history
        self.current_state = None
        self.last_action_idx = None
        self.last_state_vector = None
        self.last_action_mask = None
        self.game_history = []
        self.training_mode = True
        
        # For tracking replay buffer statistics
        self.replay_buffer_size = 0
        self.total_rewards = 0
        self.num_decisions = 0
        self.episode_rewards = []
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rl_logger.info(f"Using device: {self.device}")
        
        # Set the global agent reference
        global global_ddqn_agent
        global_ddqn_agent = self
        
        rl_logger.info(f"{self.name} initialization complete")
    
    def make_pre_roll_move(self, player, current_gameboard, allowable_moves, code):
        """
        Make a decision before rolling the dice.
        
        Parameters:
            player: The player object
            current_gameboard: The current state of the game
            allowable_moves: List of allowable moves
            code: Decision code
            
        Returns:
            The selected action
        """
        rl_logger.debug(f"Player {player.player_name} making pre-roll move")
        return self._make_decision(player, current_gameboard, "pre_roll")
    
    def make_out_of_jail_decision(self, player, current_gameboard, allowable_moves, code):
        """
        Make a decision when in jail.
        
        Parameters:
            player: The player object
            current_gameboard: The current state of the game
            allowable_moves: List of allowable moves
            code: Decision code
            
        Returns:
            The selected action
        """
        rl_logger.debug(f"Player {player.player_name} making jail decision")
        # For jail decisions, we'll use the pre_roll phase since it's part of the pre-roll phase
        return self._make_decision(player, current_gameboard, "pre_roll")
    
    def make_out_of_turn_move(self, player, current_gameboard, allowable_moves, code):
        """
        Make a decision when it's not the player's turn.
        
        Parameters:
            player: The player object
            current_gameboard: The current state of the game
            allowable_moves: List of allowable moves
            code: Decision code
            
        Returns:
            The selected action
        """
        rl_logger.debug(f"Player {player.player_name} making out-of-turn move")
        return self._make_decision(player, current_gameboard, "out_of_turn")
    
    def make_post_roll_move(self, player, current_gameboard, allowable_moves, code):
        """
        Make a decision after rolling the dice.
        
        Parameters:
            player: The player object
            current_gameboard: The current state of the game
            allowable_moves: List of allowable moves
            code: Decision code
            
        Returns:
            The selected action
        """
        rl_logger.debug(f"Player {player.player_name} making post-roll move")
        return self._make_decision(player, current_gameboard, "post_roll")
    
    def make_buy_property_decision(self, player, current_gameboard, property_positions, price):
        """
        Make a decision about buying a property.
        
        Parameters:
            player: The player object
            current_gameboard: The current state of the game
            property_positions: List of property positions
            price: The price of the property
            
        Returns:
            True to buy, False to pass
        """
        rl_logger.debug(f"Player {player.player_name} deciding whether to buy property at position {property_positions} for ${price}")
        # For buy property decisions, we'll use the post_roll phase
        decision = self._make_decision(player, current_gameboard, "post_roll")
        
        # Extract the buy decision from the decision dictionary
        if isinstance(decision, dict) and 'action' in decision and decision['action'] == 'buy_property':
            rl_logger.info(f"Player {player.player_name} decided to BUY property at position {property_positions} for ${price}")
            return True
        else:
            rl_logger.info(f"Player {player.player_name} decided NOT to buy property at position {property_positions} for ${price}")
            return False
    
    def make_bid(self, player, current_gameboard, property_position, current_bid):
        """
        Make a bid for a property.
        
        Parameters:
            player: The player object
            current_gameboard: The current state of the game
            property_position: Position of the property
            current_bid: Current highest bid
            
        Returns:
            The bid amount
        """
        rl_logger.debug(f"Player {player.player_name} deciding on bid for property at position {property_position}, current bid: ${current_bid}")
        # For simplicity, we'll just pass on bidding for now
        # In a real implementation, you would use the DDQN to decide on a bid amount
        return 0
    
    def handle_negative_cash_balance(self, player, current_gameboard):
        """
        Handle negative cash balance.
        
        Parameters:
            player: The player object
            current_gameboard: The current state of the game
            
        Returns:
            List of actions to take
        """
        rl_logger.info(f"Player {player.player_name} handling negative cash balance: ${player.current_cash}")
        # For simplicity, we'll just mortgage properties in a fixed order
        # In a real implementation, you would use the DDQN to decide which properties to mortgage
        actions = []
        for asset in player.assets:
            if not asset.is_mortgaged:
                actions.append(('mortgage', asset))
                rl_logger.debug(f"Player {player.player_name} mortgaging asset: {asset.name}")
                if player.current_cash >= 0:
                    break
        
        rl_logger.info(f"Player {player.player_name} taking {len(actions)} actions to handle negative cash balance")
        return actions
    
    def _make_decision(self, player, current_gameboard, game_phase):
        """
        Make a decision based on the current game state using the DDQN agent.
        
        Parameters:
            player: The player object
            current_gameboard: The current state of the game
            game_phase: The current game phase (pre_roll, post_roll, out_of_turn)
            
        Returns:
            The selected action
        """
        try:
            rl_logger.debug(f"Making decision for player {player.player_name} in phase {game_phase}")
            
            # Use the state encoder to encode the current game state
            state_vector = self.state_encoder.encode_state(current_gameboard)
            
            # Get the action mask from the action encoder
            _, action_mask = self.action_encoder.encode(player, current_gameboard, game_phase)
            
            # Convert to tensors
            state_tensor = state_vector.to(self.device)
            mask_tensor = torch.BoolTensor(action_mask).to(self.device)
            
            # Store the current state if we have a previous state and action
            if self.current_state is not None and self.last_action_idx is not None:
                # Calculate reward
                reward = self._calculate_reward(player, current_gameboard)
                
                # Check if the episode is done
                done = self._is_episode_done(current_gameboard)
                
                # Add the transition to the replay buffer
                self.ddqn_agent.replay_buffer.push(
                    self.current_state,
                    self.last_action_idx,
                    reward,
                    state_vector.squeeze().numpy(),
                    done
                )
                
                # Update replay buffer statistics
                self.replay_buffer_size = len(self.ddqn_agent.replay_buffer.buffer)
                self.total_rewards += reward
                self.num_decisions += 1
                
                # Log the reward and replay buffer update
                rl_logger.info(f"Reward: {reward:.4f}, Replay Buffer Size: {self.replay_buffer_size}")
                rl_logger.debug(f"State: {self.current_state[:5]}..., Action: {self.last_action_idx}, Next State: {state_vector.squeeze().numpy()[:5]}..., Done: {done}")
                
                # Add to game history
                self.game_history.append({
                    'state': self.current_state,
                    'action': self.last_action_idx,
                    'reward': reward,
                    'next_state': state_vector.squeeze().numpy(),
                    'done': done
                })
                
                # If episode is done, log episode statistics
                if done:
                    self.episode_rewards.append(self.total_rewards)
                    rl_logger.info(f"Episode complete. Total rewards: {self.total_rewards:.4f}, Average reward: {self.total_rewards/max(1, self.num_decisions):.4f}")
                    rl_logger.info(f"Replay buffer size: {self.replay_buffer_size}")
                    # Reset episode statistics
                    self.total_rewards = 0
                    self.num_decisions = 0
            
            # Update current state
            self.current_state = state_vector.squeeze().numpy()
            self.last_state_vector = state_vector
            self.last_action_mask = action_mask
            
            if self.training_mode:
                # Select action using epsilon-greedy policy
                if random.random() < self.ddqn_agent.epsilon:
                    # Random action from valid actions
                    valid_indices = np.where(action_mask)[0]
                    action_idx = np.random.choice(valid_indices) if len(valid_indices) > 0 else 0
                    rl_logger.debug(f"Random action selected: {action_idx}")
                else:
                    # Get Q-values from policy network
                    with torch.no_grad():
                        q_values = self.ddqn_agent.policy_net(state_tensor)
                        
                    # Mask invalid actions with large negative values
                    masked_q_values = q_values.clone()
                    masked_q_values[0, ~mask_tensor] = float('-inf')
                    
                    # Select action with highest Q-value
                    action_idx = masked_q_values.argmax(dim=1).item()
                    rl_logger.debug(f"Greedy action selected: {action_idx}, Q-value: {masked_q_values[0, action_idx].item():.4f}")
            else:
                # In evaluation mode, always select the best action
                with torch.no_grad():
                    q_values = self.ddqn_agent.policy_net(state_tensor)
                    
                    # Mask invalid actions
                    masked_q_values = q_values.clone()
                    masked_q_values[0, ~mask_tensor] = float('-inf')
                    
                    # Select action with highest Q-value
                    action_idx = masked_q_values.argmax(dim=1).item()
                    rl_logger.debug(f"Evaluation action selected: {action_idx}, Q-value: {masked_q_values[0, action_idx].item():.4f}")
            
            # Store the selected action
            self.last_action_idx = action_idx
            
            # For now, return a simple action based on the game phase
            # In a real implementation, you would decode the action index into a proper decision
            if game_phase == "pre_roll":
                if action_choices.use_get_out_of_jail_card in player.compute_allowable_pre_roll_actions(current_gameboard):
                    rl_logger.info(f"Player {player.player_name} using get out of jail card")
                    return action_choices.use_get_out_of_jail_card
                elif action_choices.pay_jail_fine in player.compute_allowable_pre_roll_actions(current_gameboard):
                    rl_logger.info(f"Player {player.player_name} paying jail fine")
                    return action_choices.pay_jail_fine
                rl_logger.info(f"Player {player.player_name} skipping turn (pre-roll)")
                return action_choices.skip_turn
            elif game_phase == "post_roll":
                if action_choices.buy_property in player.compute_allowable_post_roll_actions(current_gameboard):
                    rl_logger.info(f"Player {player.player_name} buying property")
                    return action_choices.buy_property
                rl_logger.info(f"Player {player.player_name} concluding actions (post-roll)")
                return action_choices.concluded_actions
            else:  # out_of_turn
                rl_logger.info(f"Player {player.player_name} skipping turn (out-of-turn)")
                return action_choices.skip_turn
                
        except Exception as e:
            logger.error(f"Error in _make_decision: {str(e)}")
            rl_logger.error(f"Error in _make_decision: {str(e)}", exc_info=True)
            # Fallback to a simple decision
            if game_phase == "pre_roll":
                return action_choices.skip_turn
            elif game_phase == "post_roll":
                return action_choices.concluded_actions
            else:  # out_of_turn
                return action_choices.skip_turn
    
    def _calculate_reward(self, player, current_gameboard):
        """
        Calculate the reward for the current state.
        
        Parameters:
            player: The player object
            current_gameboard: The current state of the game
            
        Returns:
            The reward value
        """
        # Calculate net worth
        net_worth = player.current_cash
        for asset in player.assets:
            net_worth += asset.price
            if hasattr(asset, 'num_houses'):
                net_worth += asset.num_houses * asset.house_price
            if hasattr(asset, 'num_hotels') and asset.num_hotels > 0:
                net_worth += asset.num_hotels * asset.house_price * 5
        
        # Simple reward based on net worth
        reward = net_worth / 10000.0
        
        # Additional reward for winning
        if 'winner' in current_gameboard and current_gameboard['winner'] == player.player_name:
            reward += 100
            rl_logger.info(f"Player {player.player_name} WON! Adding winning bonus of 100 to reward")
        
        # Penalty for losing
        if player.status == 'lost':
            reward -= 50
            rl_logger.info(f"Player {player.player_name} LOST! Adding losing penalty of -50 to reward")
        
        # Log detailed reward calculation
        rl_logger.info(f"Reward calculation for {player.player_name}:")
        rl_logger.info(f"  Cash: ${player.current_cash}")
        rl_logger.info(f"  Assets: ${net_worth - player.current_cash}")
        rl_logger.info(f"  Net worth: ${net_worth}")
        rl_logger.info(f"  Final reward: {reward}")
        
        return reward
    
    def _is_episode_done(self, current_gameboard):
        """
        Check if the episode is done.
        
        Parameters:
            current_gameboard: The current state of the game
            
        Returns:
            True if the episode is done, False otherwise
        """
        # Check if the game has a winner
        if 'winner' in current_gameboard and current_gameboard['winner'] is not None:
            rl_logger.info(f"Episode done: Game has a winner - {current_gameboard['winner']}")
            return True
        
        # Check if the player has lost
        for p in current_gameboard['players']:
            if p.player_name == self.name and p.status == 'lost':
                rl_logger.info(f"Episode done: Player {self.name} has lost")
                return True
        
        return False
    
    def train(self):
        """
        Train the DDQN agent.
        
        Returns:
            The loss value
        """
        loss = self.ddqn_agent.optimize_model()
        if loss is not None:
            rl_logger.info(f"Training loss: {loss:.6f}")
        return loss
    
    def update_target_network(self):
        """
        Update the target network.
        """
        self.ddqn_agent.update_target_network()
        rl_logger.info("Target network updated")
    
    def save_model(self, path):
        """
        Save the model to a file.
        
        Parameters:
            path: The path to save the model
        """
        torch.save({
            'policy_net': self.ddqn_agent.policy_net.state_dict(),
            'target_net': self.ddqn_agent.target_net.state_dict(),
            'optimizer': self.ddqn_agent.optimizer.state_dict(),
            'epsilon': self.ddqn_agent.epsilon
        }, path)
        rl_logger.info(f"Model saved to {path}")
    
    def load_model(self, path):
        """
        Load the model from a file.
        
        Parameters:
            path: The path to load the model from
        """
        checkpoint = torch.load(path)
        self.ddqn_agent.policy_net.load_state_dict(checkpoint['policy_net'])
        self.ddqn_agent.target_net.load_state_dict(checkpoint['target_net'])
        self.ddqn_agent.optimizer.load_state_dict(checkpoint['optimizer'])
        self.ddqn_agent.epsilon = checkpoint['epsilon']
        rl_logger.info(f"Model loaded from {path}")
    
    def set_training_mode(self, training_mode):
        """
        Set the training mode.
        
        Parameters:
            training_mode: Whether to enable training mode
        """
        self.training_mode = training_mode
        rl_logger.info(f"Training mode set to {training_mode}")
        
    def trade_proposal(self, player, current_gameboard):
        """
        Generate a trade proposal.
        
        Parameters:
            player: The player object
            current_gameboard: The current state of the game
            
        Returns:
            A trade proposal or None
        """
        rl_logger.debug(f"Player {player.player_name} considering trade proposal")
        # For simplicity, we'll just return None for now
        # In a real implementation, you would use the DDQN to generate a trade proposal
        return None
    
    def get_replay_buffer_stats(self):
        """
        Get statistics about the replay buffer.
        
        Returns:
            A dictionary with replay buffer statistics
        """
        stats = {
            'size': len(self.ddqn_agent.replay_buffer.buffer),
            'capacity': self.ddqn_agent.replay_buffer.capacity,
            'utilization': len(self.ddqn_agent.replay_buffer.buffer) / self.ddqn_agent.replay_buffer.capacity,
            'avg_reward': self.total_rewards / max(1, self.num_decisions),
            'episode_rewards': self.episode_rewards
        }
        rl_logger.info(f"Replay buffer stats: {stats}")
        return stats

# Initialize the global agent
ddqn_agent_instance = DDQNDecisionAgent()

# Standalone functions that will be used by the decision agent methods dictionary
def handle_negative_cash_balance(player, current_gameboard):
    """Wrapper for the DDQN agent's handle_negative_cash_balance method."""
    if global_ddqn_agent:
        return global_ddqn_agent.handle_negative_cash_balance(player, current_gameboard)
    return []

def make_pre_roll_move(player, current_gameboard, allowable_moves, code):
    """Wrapper for the DDQN agent's make_pre_roll_move method."""
    if global_ddqn_agent:
        return global_ddqn_agent.make_pre_roll_move(player, current_gameboard, allowable_moves, code)
    return action_choices.skip_turn

def make_out_of_turn_move(player, current_gameboard, allowable_moves, code):
    """Wrapper for the DDQN agent's make_out_of_turn_move method."""
    if global_ddqn_agent:
        return global_ddqn_agent.make_out_of_turn_move(player, current_gameboard, allowable_moves, code)
    return action_choices.skip_turn

def make_post_roll_move(player, current_gameboard, allowable_moves, code):
    """Wrapper for the DDQN agent's make_post_roll_move method."""
    if global_ddqn_agent:
        return global_ddqn_agent.make_post_roll_move(player, current_gameboard, allowable_moves, code)
    return action_choices.concluded_actions

def make_buy_property_decision(player, current_gameboard, property_positions, price):
    """Wrapper for the DDQN agent's make_buy_property_decision method."""
    if global_ddqn_agent:
        return global_ddqn_agent.make_buy_property_decision(player, current_gameboard, property_positions, price)
    return False

def make_bid(player, current_gameboard, property_position, current_bid):
    """Wrapper for the DDQN agent's make_bid method."""
    if global_ddqn_agent:
        return global_ddqn_agent.make_bid(player, current_gameboard, property_position, current_bid)
    return 0

# Build the decision agent methods dictionary
def _build_decision_agent_methods_dict():
    """
    This function builds the decision agent methods dictionary.
    :return: The decision agent dict. Keys should be exactly as stated in this example, but the functions can be anything
    as long as you use/expect the exact function signatures we have indicated in this document.
    """
    ans = dict()
    ans['handle_negative_cash_balance'] = handle_negative_cash_balance
    ans['make_pre_roll_move'] = make_pre_roll_move
    ans['make_out_of_turn_move'] = make_out_of_turn_move
    ans['make_post_roll_move'] = make_post_roll_move
    ans['make_buy_property_decision'] = make_buy_property_decision
    ans['make_bid'] = make_bid
    ans['type'] = "decision_agent_methods"
    return ans

# Create the decision agent methods dictionary
decision_agent_methods = _build_decision_agent_methods_dict()

# Function to check replay buffer status
def check_replay_buffer():
    """
    Print statistics about the replay buffer.
    """
    if global_ddqn_agent:
        stats = global_ddqn_agent.get_replay_buffer_stats()
        print(f"Replay Buffer Statistics:")
        print(f"  Size: {stats['size']} / {stats['capacity']} ({stats['utilization']*100:.1f}% full)")
        print(f"  Average Reward: {stats['avg_reward']:.4f}")
        if stats['episode_rewards']:
            print(f"  Last Episode Reward: {stats['episode_rewards'][-1]:.4f}")
            print(f"  Average Episode Reward: {sum(stats['episode_rewards'])/len(stats['episode_rewards']):.4f}")
        
        # Print a sample from the buffer if it's not empty
        if stats['size'] > 0:
            sample = global_ddqn_agent.ddqn_agent.replay_buffer.buffer[-1]  # Get the most recent transition
            print(f"Most recent transition:")
            print(f"  State: {sample[0][:5]}...")  # Print first 5 elements of state
            print(f"  Action: {sample[1]}")
            print(f"  Reward: {sample[2]:.4f}")
            print(f"  Next State: {sample[3][:5]}...")  # Print first 5 elements of next state
            print(f"  Done: {sample[4]}")
    else:
        print("No DDQN agent initialized yet.")
