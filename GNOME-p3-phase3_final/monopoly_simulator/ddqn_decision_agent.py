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
from monopoly_simulator import action_validator
from monopoly_simulator.flag_config import flag_config_dict
from monopoly_simulator.flag_config import flag_config_dict

##############################################################
# Logging configuration
##############################################################
current_file = globals().get('__file__', os.getcwd())
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(current_file)))
log_dir = os.path.join(base_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'rl_agent.log')

rl_logger = logging.getLogger('rl_agent_logs')
rl_logger.setLevel(logging.DEBUG)

if not rl_logger.handlers:
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    rl_logger.addHandler(file_handler)
    rl_logger.addHandler(console_handler)
##############################################################

logger = logging.getLogger('monopoly_simulator.logging_info.ddqn_decision_agent')

global_ddqn_agent = None

class DDQNDecisionAgent(Agent):
    """
    A wrapper class that implements the decision agent interface for the Monopoly simulator
    using a DDQN (Double Deep Q-Network) agent for decision making.
    
    This agent uses:
    - MonopolyStateEncoder to encode the game state into a 240-dimensional vector
    - DDQN network to predict Q-values for each possible action
    - ActionEncoder to decode the selected action index into a valid game action along with required parameters
    """
    def __init__(self, name="DDQN Agent", state_dim=240, action_dim=2950,
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
        
        self.ddqn_agent = DDQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=lr,
            gamma=gamma,
            batch_size=batch_size,
            replay_capacity=replay_capacity,
            target_update_freq=target_update_freq
        )
        
        self.state_encoder = MonopolyStateEncoder()
        self.action_encoder = ActionEncoder()
        
        self.current_state = None
        self.last_action_idx = None
        self.last_state_vector = None
        self.last_action_mask = None
        self.game_history = []
        self.training_mode = True
        
        self.replay_buffer_size = 0
        self.total_rewards = 0
        self.num_decisions = 0
        self.episode_rewards = []
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rl_logger.info(f"Using device: {self.device}")
        
        global global_ddqn_agent
        global_ddqn_agent = self
        
        rl_logger.info(f"{self.name} initialization complete")
    
    def make_pre_roll_move(self, player, current_gameboard, allowable_moves, code):
        rl_logger.debug(f"Player {player.player_name} making pre-roll move")
        return self._make_decision(player, current_gameboard, "pre_roll")
    
    def make_out_of_jail_decision(self, player, current_gameboard, allowable_moves, code):
        rl_logger.debug(f"Player {player.player_name} making jail decision")
        return self._make_decision(player, current_gameboard, "pre_roll")
    
    def make_out_of_turn_move(self, player, current_gameboard, allowable_moves, code):
        rl_logger.debug(f"Player {player.player_name} making out-of-turn move")
        return self._make_decision(player, current_gameboard, "out_of_turn")
    
    def make_post_roll_move(self, player, current_gameboard, allowable_moves, code):
        rl_logger.debug(f"Player {player.player_name} making post-roll move")
        return self._make_decision(player, current_gameboard, "post_roll")
    
    def make_buy_property_decision(self, player, current_gameboard, property_positions, price):
        rl_logger.debug(f"Player {player.player_name} deciding whether to buy property at position {property_positions} for ${price}")
        decision = self._make_decision(player, current_gameboard, "post_roll")
        
        if isinstance(decision, dict) and 'parameters' in decision and decision['parameters'].get("action_name") == "buy_property":
            rl_logger.info(f"Player {player.player_name} decided to BUY property at position {property_positions} for ${price}")
            return True
        else:
            rl_logger.info(f"Player {player.player_name} decided NOT to buy property at position {property_positions} for ${price}")
            return False
    
    def make_bid(self, player, current_gameboard, property_position, current_bid):
        rl_logger.debug(f"Player {player.player_name} deciding on bid for property at position {property_position}, current bid: ${current_bid}")
        return 0
    
    def handle_negative_cash_balance(self, player, current_gameboard):
        rl_logger.info(f"Player {player.player_name} handling negative cash balance: ${player.current_cash}")
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
        
        This method:
        1. Encodes the current game state
        2. Builds the full action mapping
        3. Creates an action mask for valid actions
        4. Selects an action from valid actions
        5. Returns a tuple of (action_function/name, parameters) that gameplay_socket_phase3.py expects
        
        Returns:
            A tuple of (action_function/name, parameters) where:
            - action_function/name: Either a callable function or a string action name
            - parameters: A dictionary of parameters required for the action
        """
        try:
            rl_logger.debug(f"Making decision for player {player.player_name} in phase {game_phase} (background agent style)")
            
            # Create state vector and tensor
            state_vector = self.state_encoder.encode_state(current_gameboard)
            state_tensor = state_vector.to(self.device)
            
            # Build the full action mapping
            action_encoder = ActionEncoder()
            full_mapping = action_encoder.build_full_action_mapping(player, current_gameboard)
            
            # Import and use the create_action_mask function to get valid actions
            from monopoly_simulator.action_mapping_builder import create_action_mask
            action_mask = create_action_mask(player, current_gameboard, game_phase)
            mask_tensor = torch.BoolTensor(action_mask).to(self.device)
            
            # Find indices of valid actions
            valid_action_indices = np.where(action_mask)[0]
            
            if len(valid_action_indices) == 0:
                rl_logger.warning(f"No valid actions found for player {player.player_name} in phase {game_phase}")
                # Default to skip_turn or conclude_actions based on game phase
                if game_phase == "post_roll":
                    action_name = "concluded_actions"
                else:
                    action_name = "skip_turn"
                rl_logger.info(f"Defaulting to {action_name}")
                return (action_name, {})
            else:
                # Check if there are valid actions other than skip_turn and concluded_actions
                non_skip_actions = []
                for idx in valid_action_indices:
                    action_name = full_mapping[idx]["action"]
                    if action_name not in ["skip_turn", "concluded_actions"]:
                        non_skip_actions.append(idx)
                
                # If there are non-skip actions available, prioritize them
                if len(non_skip_actions) > 0:
                    rl_logger.info(f"Found {len(non_skip_actions)} non-skip actions. Prioritizing these.")
                    action_idx = np.random.choice(non_skip_actions)
                    rl_logger.debug(f"Randomly selected non-skip action index: {action_idx}")
                else:
                    # Otherwise, select from all valid actions
                    action_idx = np.random.choice(valid_action_indices)
                    rl_logger.debug(f"Randomly selected valid action index: {action_idx}")
                
                # Later, you can replace this with DDQN selection from valid actions:
                # action_idx = self.ddqn_agent.select_action(state_tensor, mask_tensor)
            
            rl_logger.debug(f"Action_idx value: {action_idx}")
            
            # Decode the selected action
            mapping = action_encoder.decode_action(player, current_gameboard, action_idx)
            action_name = mapping.get("action")
            parameters = mapping.get("parameters", {})
            
            rl_logger.info(f"Decoded action: {action_name} with parameters: {parameters}")
            
            # Process parameters for special cases
            if "to_player" in parameters and isinstance(parameters["to_player"], str):
                target_name = parameters["to_player"]
                target_player = None
                for p in current_gameboard.get("players", []):
                    if getattr(p, "player_name", None) == target_name:
                        target_player = p
                        break
                if target_player is None:
                    rl_logger.error(f"Target player '{target_name}' not found in the current gameboard.")
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
                    rl_logger.debug(f"Trade offer before conversion: {raw_offer}")
                    # Convert cash values
                    converted_offer = action_validator.convert_cash_values(raw_offer, current_gameboard, rl_logger)
                    # Convert property names to objects
                    converted_offer = action_validator.convert_offer_properties(converted_offer, current_gameboard, rl_logger)
                    parameters["offer"] = converted_offer
                    rl_logger.debug(f"Trade offer after conversion: {parameters['offer']}")
            
            # Validate and convert asset parameters for property-related actions
            if action_name == "sell_property":
                parameters = action_validator.validate_sell_property(parameters, current_gameboard, rl_logger)
            elif action_name == "sell_house_hotel":
                parameters = action_validator.validate_sell_house_hotel_asset(parameters, current_gameboard, rl_logger)
            elif action_name in ["improve_property", "reverse_improve_property"]:
                parameters = action_validator.validate_improve_property(parameters, current_gameboard, rl_logger)
            elif action_name == "free_mortgage":
                parameters = action_validator.validate_free_mortgage(parameters, current_gameboard, rl_logger)
            elif action_name == "mortgage_property":
                parameters = action_validator.validate_free_mortgage(parameters, current_gameboard, rl_logger)
            
            # Important: Replace "current_gameboard" string with the actual current_gameboard
            if "current_gameboard" in parameters and parameters["current_gameboard"] == "current_gameboard":
                parameters["current_gameboard"] = current_gameboard
            
            # Return the action name and parameters - this is what gameplay_socket_phase3.py expects
            return (action_name, parameters)
        
        except Exception as e:
            logger.error(f"Error in _make_decision: {str(e)}")
            rl_logger.error(f"Error in _make_decision: {str(e)}", exc_info=True)
            # Return a safe default action based on game phase
            if game_phase == "post_roll":
                return ("concluded_actions", {})
            else:
                return ("skip_turn", {})

    def _calculate_reward(self, player, current_gameboard):
        net_worth = player.current_cash
        for asset in player.assets:
            net_worth += asset.price
            if hasattr(asset, 'num_houses'):
                net_worth += asset.num_houses * asset.house_price
            if hasattr(asset, 'num_hotels') and asset.num_hotels > 0:
                net_worth += asset.num_hotels * asset.house_price * 5
        
        reward = net_worth / 10000.0
        
        if 'winner' in current_gameboard and current_gameboard['winner'] == player.player_name:
            reward += 100
            rl_logger.info(f"Player {player.player_name} WON! Adding winning bonus of 100 to reward")
        
        if player.status == 'lost':
            reward -= 50
            rl_logger.info(f"Player {player.player_name} LOST! Adding losing penalty of -50 to reward")
        
        rl_logger.info(f"Reward calculation for {player.player_name}:")
        rl_logger.info(f"  Cash: ${player.current_cash}")
        rl_logger.info(f"  Assets: ${net_worth - player.current_cash}")
        rl_logger.info(f"  Net worth: ${net_worth}")
        rl_logger.info(f"  Final reward: {reward}")
        
        return reward
    
    def _is_episode_done(self, current_gameboard):
        if 'winner' in current_gameboard and current_gameboard['winner'] is not None:
            rl_logger.info(f"Episode done: Game has a winner - {current_gameboard['winner']}")
            return True
        
        for p in current_gameboard['players']:
            if p.player_name == self.name and p.status == 'lost':
                rl_logger.info(f"Episode done: Player {self.name} has lost")
                return True
        
        return False
    
    def train(self):
        loss = self.ddqn_agent.optimize_model()
        if loss is not None:
            rl_logger.info(f"Training loss: {loss:.6f}")
        return loss
    
    def update_target_network(self):
        self.ddqn_agent.update_target_network()
        rl_logger.info("Target network updated")
    
    def save_model(self, path):
        torch.save({
            'policy_net': self.ddqn_agent.policy_net.state_dict(),
            'target_net': self.ddqn_agent.target_net.state_dict(),
            'optimizer': self.ddqn_agent.optimizer.state_dict(),
            'epsilon': self.ddqn_agent.epsilon
        }, path)
        rl_logger.info(f"Model saved to {path}")
    
    def load_model(self, path):
        checkpoint = torch.load(path)
        self.ddqn_agent.policy_net.load_state_dict(checkpoint['policy_net'])
        self.ddqn_agent.target_net.load_state_dict(checkpoint['target_net'])
        self.ddqn_agent.optimizer.load_state_dict(checkpoint['optimizer'])
        self.ddqn_agent.epsilon = checkpoint['epsilon']
        rl_logger.info(f"Model loaded from {path}")
    
    def set_training_mode(self, training_mode):
        self.training_mode = training_mode
        rl_logger.info(f"Training mode set to {training_mode}")
        
    def trade_proposal(self, player, current_gameboard):
        rl_logger.debug(f"Player {player.player_name} considering trade proposal")
        return None
    
    def get_replay_buffer_stats(self):
        stats = {
            'size': len(self.ddqn_agent.replay_buffer.buffer),
            'capacity': self.ddqn_agent.replay_buffer.capacity,
            'utilization': len(self.ddqn_agent.replay_buffer.buffer) / self.ddqn_agent.replay_buffer.capacity,
            'avg_reward': self.total_rewards / max(1, self.num_decisions),
            'episode_rewards': self.episode_rewards
        }
        rl_logger.info(f"Replay buffer stats: {stats}")
        return stats

ddqn_agent_instance = DDQNDecisionAgent()

def handle_negative_cash_balance(player, current_gameboard):
    if global_ddqn_agent:
        return global_ddqn_agent.handle_negative_cash_balance(player, current_gameboard)
    return []

def make_pre_roll_move(player, current_gameboard, allowable_moves, code):
    if global_ddqn_agent:
        return global_ddqn_agent.make_pre_roll_move(player, current_gameboard, allowable_moves, code)
    return action_choices.skip_turn

def make_out_of_turn_move(player, current_gameboard, allowable_moves, code):
    if global_ddqn_agent:
        return global_ddqn_agent.make_out_of_turn_move(player, current_gameboard, allowable_moves, code)
    return action_choices.skip_turn

def make_post_roll_move(player, current_gameboard, allowable_moves, code):
    if global_ddqn_agent:
        return global_ddqn_agent.make_post_roll_move(player, current_gameboard, allowable_moves, code)
    return action_choices.concluded_actions

def make_buy_property_decision(player, current_gameboard, property_positions, price):
    if global_ddqn_agent:
        return global_ddqn_agent.make_buy_property_decision(player, current_gameboard, property_positions, price)
    return False

def make_bid(player, current_gameboard, property_position, current_bid):
    if global_ddqn_agent:
        return global_ddqn_agent.make_bid(player, current_gameboard, property_position, current_bid)
    return 0

def _build_decision_agent_methods_dict():
    ans = dict()
    ans['handle_negative_cash_balance'] = handle_negative_cash_balance
    ans['make_pre_roll_move'] = make_pre_roll_move
    ans['make_out_of_turn_move'] = make_out_of_turn_move
    ans['make_post_roll_move'] = make_post_roll_move
    ans['make_buy_property_decision'] = make_buy_property_decision
    ans['make_bid'] = make_bid
    ans['type'] = "decision_agent_methods"
    return ans

decision_agent_methods = _build_decision_agent_methods_dict()

def check_replay_buffer():
    if global_ddqn_agent:
        stats = global_ddqn_agent.get_replay_buffer_stats()
        print(f"Replay Buffer Statistics:")
        print(f"  Size: {stats['size']} / {stats['capacity']} ({stats['utilization']*100:.1f}% full)")
        print(f"  Average Reward: {stats['avg_reward']:.4f}")
        if stats['episode_rewards']:
            print(f"  Last Episode Reward: {stats['episode_rewards'][-1]:.4f}")
            print(f"  Average Episode Reward: {sum(stats['episode_rewards'])/len(stats['episode_rewards']):.4f}")
        
        if stats['size'] > 0:
            sample = global_ddqn_agent.ddqn_agent.replay_buffer.buffer[-1]
            print(f"Most recent transition:")
            print(f"  State: {sample[0][:5]}...")
            print(f"  Action: {sample[1]}")
            print(f"  Reward: {sample[2]:.4f}")
            print(f"  Next State: {sample[3][:5]}...")
            print(f"  Done: {sample[4]}")
    else:
        print("No DDQN agent initialized yet.")
