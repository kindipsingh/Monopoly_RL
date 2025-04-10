import pickle
import numpy as np
import logging
import os
import json
from typing import Dict, List, Tuple, Any, Optional

logger = logging.getLogger(__name__)

def store_player_decision(player, decision, action_index=None):
    """
    Store the decision dictionary returned by ActionEncoder's make_decision method.
    This should be called immediately after make_decision is called.
    
    Parameters:
        player: The Player instance
        decision: The decision dictionary returned by make_decision
        action_index: Optional explicit action index if known
    """
    if player is None:
        logger.warning("Cannot store decision: Player object is None.")
        return
    
    # Store the entire decision dictionary
    player.last_decision = decision
    
    # If an explicit action index was provided, store it
    if action_index is not None:
        decision['action_index'] = action_index
        
        # Also store in agent memory if available
        if hasattr(player, 'agent'):
            if not hasattr(player.agent, '_agent_memory') or player.agent._agent_memory is None:
                player.agent._agent_memory = {}
            player.agent._agent_memory['last_action_index'] = action_index
            player.agent.last_action_index = action_index
    
    # Extract action name if possible
    action_name = None
    if 'action_vector' in decision and 'action_mask' in decision:
        # Find the index of the selected action
        for i, (val, mask) in enumerate(zip(decision['action_vector'], decision['action_mask'])):
            if val == 1 and mask:
                # We found the selected action, now map it back to a name
                try:
                    from monopoly_simulator.action_encoding import ActionEncoder
                    encoder = ActionEncoder()
                    full_mapping = encoder.build_full_action_mapping(player, None)  # We don't have gameboard here
                    if i < len(full_mapping):
                        action_name = full_mapping[i].get('action')
                except Exception as e:
                    logger.error(f"Error mapping action index to name: {str(e)}")
    
    # Store action name if found
    if action_name and hasattr(player, 'agent'):
        player.agent.last_action_name = action_name
        if hasattr(player.agent, '_agent_memory') and player.agent._agent_memory is not None:
            player.agent._agent_memory['previous_action'] = action_name
    
    logger.debug(f"Stored decision for player {getattr(player, 'player_name', 'unknown')}")

def numpy_to_serializable(obj):
    """Convert numpy arrays and data types to serializable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    else:
        return obj

class ReplayBuffer:
    """
    A simple replay buffer to store transitions for reinforcement learning.
    """
    def __init__(self, capacity: int = 100000):
        """
        Initialize a replay buffer.
        
        Args:
            capacity (int): Maximum number of transitions to store
        """
        self.buffer = []
        self.capacity = capacity
        self.position = 0
        self.episode_rewards = []
        self.current_episode_reward = 0.0
        
    def add(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer.
        
        Args:
            state: The current state
            action: The action taken (can be an index, a player object, or a decision dictionary)
            reward: The reward received
            next_state: The next state
            done: Whether the episode is done
        """
        # Extract the action index based on what type of action we received
        action_idx = 0  # Default
        
        if isinstance(action, (int, np.integer)):
            # Action is already an index
            action_idx = action
        elif isinstance(action, dict) and 'action_index' in action:
            # Action is a decision dictionary with an action_index
            action_idx = action['action_index']
        elif hasattr(action, 'player_name'):  # This is likely a player object
            # Get the action index from the player
            action_idx = get_action_index(action, next_state)  # Using next_state as the game_elements
        
        # Add to buffer
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action_idx, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
        # Track rewards
        self.current_episode_reward += reward
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0.0
        
        logger.debug(f"Added transition to buffer with action {action_idx}. Buffer size: {len(self.buffer)}/{self.capacity}")
        
    def sample(self, batch_size):
        """
        Sample a batch of transitions.
        
        Args:
            batch_size (int): Size of batch to sample
            
        Returns:
            Tuple of state, action, reward, next_state, done batches
        """
        batch_size = min(batch_size, len(self.buffer))
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def get_stats(self):
        """
        Get statistics about the replay buffer.
        
        Returns:
            Dict containing buffer statistics
        """
        return {
            'size': len(self.buffer),
            'capacity': self.capacity,
            'avg_reward': sum([x[2] for x in self.buffer]) / max(1, len(self.buffer)) if self.buffer else 0,
            'episode_rewards': self.episode_rewards
        }
    
    def save_to_file(self, file_path, save_summary=True, detailed_summary=False):
        """
        Save the replay buffer to a file with optional summary JSON files.
        
        Args:
            file_path (str): Path to save the buffer
            save_summary (bool): Whether to save summary JSON files
            detailed_summary (bool): Whether to include full state vectors in the summary
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save the buffer as a pickle file
        with open(file_path, 'wb') as f:
            pickle.dump({
                'buffer': self.buffer,
                'capacity': self.capacity,
                'position': self.position,
                'episode_rewards': self.episode_rewards
            }, f)
        logger.info(f"Replay buffer saved to {file_path}")
        
        # Optionally save summary JSON files
        if save_summary:
            # Save a compact summary
            compact_summary_path = file_path + '.summary.json'
            self.save_compact_summary(compact_summary_path)
            
            # Optionally save a detailed summary
            if detailed_summary:
                detailed_summary_path = file_path + '.full_summary.json'
                self.save_detailed_summary(detailed_summary_path)
                logger.info(f"Detailed summary saved to {detailed_summary_path}")
    
    def save_compact_summary(self, file_path):
        """
        Save a compact summary of the replay buffer to a JSON file.
        This includes summary statistics but not the full state vectors.
        
        Args:
            file_path (str): Path to save the summary
        """
        summary = {
            'size': len(self.buffer),
            'capacity': self.capacity,
            'utilization': len(self.buffer) / self.capacity if self.capacity > 0 else 0,
            'episode_rewards': [float(r) for r in self.episode_rewards],
            'avg_reward': 0
        }
        
        # Calculate average reward
        if len(self.buffer) > 0:
            rewards = [exp[2] for exp in self.buffer]
            summary['avg_reward'] = float(sum(rewards) / len(rewards))
        
        # Include compact entries with just summaries of the state vectors
        entries = []
        for i, (s, a, r, ns, d) in enumerate(self.buffer):
            if s is None or ns is None:
                continue
                
            entry = {
                'index': i,
                'state_summary': f"Array of shape {s.shape if hasattr(s, 'shape') else 'unknown'}, mean: {np.mean(s) if hasattr(s, 'mean') else 'unknown'}",
                'action': numpy_to_serializable(a),
                'reward': numpy_to_serializable(r),
                'next_state_summary': f"Array of shape {ns.shape if hasattr(ns, 'shape') else 'unknown'}, mean: {np.mean(ns) if hasattr(ns, 'mean') else 'unknown'}",
                'done': numpy_to_serializable(d)
            }
            entries.append(entry)
        
        summary['all_entries'] = entries
        
        with open(file_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Compact summary saved to {file_path}")
    
    def save_detailed_summary(self, file_path):
        """
        Save a detailed summary of the replay buffer to a JSON file.
        This includes the full state vectors, which can make the file very large.
        
        Args:
            file_path (str): Path to save the summary
        """
        summary = {
            'size': len(self.buffer),
            'capacity': self.capacity,
            'utilization': len(self.buffer) / self.capacity if self.capacity > 0 else 0,
            'episode_rewards': [float(r) for r in self.episode_rewards],
            'avg_reward': 0
        }
        
        # Calculate average reward
        if len(self.buffer) > 0:
            rewards = [exp[2] for exp in self.buffer]
            summary['avg_reward'] = float(sum(rewards) / len(rewards))
        
        # Include ALL entries with COMPLETE state and next_state arrays
        entries = []
        for i, (s, a, r, ns, d) in enumerate(self.buffer):
            if s is None or ns is None:
                continue
                
            entry = {
                'index': i,
                'state': numpy_to_serializable(s),  # Full state array
                'action': numpy_to_serializable(a),
                'reward': numpy_to_serializable(r),
                'next_state': numpy_to_serializable(ns),  # Full next_state array
                'done': numpy_to_serializable(d)
            }
            entries.append(entry)
        
        summary['all_entries'] = entries
        
        with open(file_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Detailed summary saved to {file_path}")
    
    def load_from_file(self, file_path):
        """
        Load the replay buffer from a file.
        
        Args:
            file_path (str): Path to the saved buffer
        """
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                try:
                    data = pickle.load(f)
                    if isinstance(data, dict):
                        self.buffer = data.get('buffer', [])
                        self.capacity = data.get('capacity', self.capacity)
                        self.position = data.get('position', 0)
                        self.episode_rewards = data.get('episode_rewards', [])
                    else:
                        # Handle case where buffer was saved as a plain list
                        self.buffer = data
                        self.position = len(self.buffer) % self.capacity
                except Exception as e:
                    logger.error(f"Error loading replay buffer: {e}")
            
            logger.info(f"Replay buffer loaded from {file_path} with {len(self.buffer)} transitions")
        else:
            logger.warning(f"No replay buffer found at {file_path}")

def calculate_reward(player, current_gameboard):
    """
    Calculate a comprehensive reward based on the player's current state in the game.
    This function evaluates multiple aspects of the player's performance:
    - Net worth (cash + property values)
    - Property ownership and monopolies
    - Cash balance
    - Game state (winning/losing)
    
    Parameters:
        player: A Player instance representing the current player
        current_gameboard: A dict representing the current game board state
        
    Returns:
        float: The calculated reward value
    """
    # Check if player is None or doesn't have necessary attributes
    if player is None:
        logger.warning("Player object is None. Returning default reward of 0.")
        return 0.0
    
    # Import ddqn_decision_agent to use its reward calculation
    try:
        from monopoly_simulator import ddqn_decision_agent
        
        # Try to use the DDQN agent's reward calculation first
        if hasattr(ddqn_decision_agent, 'ddqn_agent_instance') and ddqn_decision_agent.ddqn_agent_instance:
            try:
                return ddqn_decision_agent.ddqn_agent_instance._calculate_reward(player, current_gameboard)
            except Exception as e:
                logger.warning(f"Error using DDQN agent's reward calculation: {e}. Falling back to default.")
    except ImportError:
        logger.warning("Could not import ddqn_decision_agent module. Using default reward calculation.")
    
    # Verify player has necessary attributes
    if not hasattr(player, 'current_cash'):
        logger.warning(f"Player {getattr(player, 'player_name', 'unknown')} has no current_cash attribute. Returning default reward of 0.")
        return 0.0
    
    # Base reward: Net worth calculation
    net_worth = player.current_cash
    property_value = 0
    
    # Calculate property values including houses and hotels
    # Safely check if player.assets exists and is iterable
    player_assets = getattr(player, 'assets', [])
    if player_assets is None:
        player_assets = []
        logger.warning(f"Player {getattr(player, 'player_name', 'unknown')} has assets attribute set to None. Treating as empty list.")
    
    try:
        for asset in player_assets:
            if hasattr(asset, 'price'):
                property_value += asset.price
                if hasattr(asset, 'num_houses') and hasattr(asset, 'price_per_house'):
                    property_value += asset.num_houses * asset.price_per_house
                if hasattr(asset, 'num_hotels') and asset.num_hotels > 0 and hasattr(asset, 'price_per_house'):
                    property_value += asset.num_hotels * asset.price_per_house * 5
    except TypeError as e:
        logger.error(f"Error iterating through player assets: {e}. Using only cash value for net worth.")
        # Continue with just cash as net worth
    
    # Calculate total net worth
    total_net_worth = net_worth + property_value
    
    # Base reward normalized by a factor to keep it manageable
    base_reward = total_net_worth / 10000.0
    
    # Additional rewards for strategic achievements
    strategic_reward = 0.0
    
    # Reward for monopolies (owning all properties of a color group)
    # Safely check if player has full_color_sets_possessed attribute
    monopoly_reward = 0.0
    if hasattr(player, 'full_color_sets_possessed'):
        try:
            monopoly_reward = len(player.full_color_sets_possessed) * 5.0
        except TypeError:
            logger.warning(f"Player {getattr(player, 'player_name', 'unknown')} has full_color_sets_possessed that is not iterable. Assuming 0 monopolies.")
            monopoly_reward = 0.0
    strategic_reward += monopoly_reward
    
    # Reward for property development (houses and hotels)
    development_reward = 0.0
    try:
        for asset in player_assets:
            if hasattr(asset, 'num_houses') and asset.num_houses > 0:
                development_reward += asset.num_houses * 0.5
            if hasattr(asset, 'num_hotels') and asset.num_hotels > 0:
                development_reward += asset.num_hotels * 3.0
    except TypeError:
        logger.warning("Unable to calculate development reward due to asset iteration issue.")
    strategic_reward += development_reward
    
    # Reward for maintaining cash reserves (liquidity)
    liquidity_reward = 0.0
    if player.current_cash > 500:
        liquidity_reward = min(player.current_cash / 5000.0, 5.0)  # Cap at 5.0
    strategic_reward += liquidity_reward
    
    # Penalty for mortgaged properties
    mortgage_penalty = 0.0
    try:
        for asset in player_assets:
            if hasattr(asset, 'is_mortgaged') and asset.is_mortgaged:
                mortgage_penalty += 0.5
    except TypeError:
        logger.warning("Unable to calculate mortgage penalty due to asset iteration issue.")
    strategic_reward -= mortgage_penalty
    
    # Game state rewards/penalties
    game_state_reward = 0.0
    
    # Check if player has player_name attribute
    player_name = getattr(player, 'player_name', None)
    
    # Major reward for winning
    if player_name and 'winner' in current_gameboard and current_gameboard['winner'] == player_name:
        game_state_reward += 100.0
        logger.info(f"Player {player_name} WON! Adding winning bonus of 100 to reward")
    
    # Major penalty for losing
    if hasattr(player, 'status') and player.status == 'lost':
        game_state_reward -= 50.0
        logger.info(f"Player {getattr(player, 'player_name', 'unknown')} LOST! Adding losing penalty of -50 to reward")
    
    # Calculate final reward
    final_reward = base_reward + strategic_reward + game_state_reward
    
    logger.debug(f"Reward for {getattr(player, 'player_name', 'unknown')}: {final_reward:.2f} (Base: {base_reward:.2f}, Strategic: {strategic_reward:.2f}, Game State: {game_state_reward:.2f})")
    
    return final_reward

def get_action_index(player, game_elements):
    """
    Get the action index directly from the player's last action.
    
    This function retrieves the action index that was selected in the DDQNDecisionAgent's
    _make_decision method.
    
    Parameters:
        player: A Player instance that performed an action
        game_elements: The current game state passed from simulate_game_instance
        
    Returns:
        int: The index of the action in the full action space, or 0 if not found
    """
    # Check if player is None
    if player is None:
        logger.warning("Player object is None. Returning default action index 0.")
        return 0
    
    try:
        # First check if the player has a last_action_idx attribute directly
        if hasattr(player, 'last_action_idx') and player.last_action_idx is not None:
            logger.debug(f"Using player's last_action_idx: {player.last_action_idx}")
            return player.last_action_idx
        
        # Check if the player's agent has the last_action_idx
        if hasattr(player, 'agent') and hasattr(player.agent, 'last_action_idx'):
            logger.debug(f"Using agent's last_action_idx: {player.agent.last_action_idx}")
            return player.agent.last_action_idx
        
        # Check if we can find it in the agent's memory
        if hasattr(player, 'agent') and hasattr(player.agent, '_agent_memory'):
            if player.agent._agent_memory and 'last_action_idx' in player.agent._agent_memory:
                logger.debug(f"Using last_action_idx from agent memory: {player.agent._agent_memory['last_action_idx']}")
                return player.agent._agent_memory['last_action_idx']
        
        # If we can't find it, return 0 as default
        logger.warning(f"Could not find action index for player {getattr(player, 'player_name', 'unknown')}, using default (0)")
        return 0
    except Exception as e:
        logger.error(f"Error getting action index: {str(e)}")
        return 0
def get_action_mapping():
    """
    Get a simplified mapping from action names to action indices.
    
    This is used as a fallback when the full action mapping from ActionEncoder is not available.
    
    Returns:
        dict: Mapping from action names to indices
    """
    # This is a simplified version of the action space defined in ActionEncoder
    return {
        "skip_turn": 0,
        "concluded_actions": 1,
        "buy_property": 2,
        "mortgage_property": 3,
        "free_mortgage": 4,
        "improve_property": 5,
        "reverse_improve_property": 6,
        "accept_trade_offer": 7,
        "decline_trade_offer": 8,
        "make_trade_offer": 9,
        "make_sell_property_offer": 10,
        "sell_property": 11,
        "sell_house_hotel": 12,
        "pay_jail_fine": 13,
        "use_get_out_of_jail_card": 14,
        "roll_die": 15
    }

def add_to_replay_buffer(replay_buffer, state, player, reward, next_state, done):
    """
    Add a transition to the replay buffer with the correct action index.
    
    Parameters:
        replay_buffer: The replay buffer to add the transition to
        state: The current state
        player: The player who took the action
        reward: The reward received
        next_state: The next state
        done: Whether the episode is done
        
    Returns:
        int: The action index that was used
    """
    # Get the action index from the player
    action_idx = get_action_index(player, next_state)
    
    # Add the transition to the replay buffer
    replay_buffer.add(state, action_idx, reward, next_state, done)
    
    player_name = getattr(player, 'player_name', 'unknown')
    logger.debug(f"Added transition to replay buffer with action {action_idx} for player {player_name}")
    return action_idx

def is_episode_done(player, current_gameboard):
    """
    Check if the current episode is done.
    
    Parameters:
        player: A Player instance
        current_gameboard: A dict representing the current game board state
        
    Returns:
        bool: True if the episode is done, False otherwise
    """
    # Check if player is None
    if player is None:
        logger.warning("Player object is None in is_episode_done. Returning True to end episode.")
        return True
    
    # Episode is done if there's a winner or the player has lost
    if 'winner' in current_gameboard and current_gameboard['winner'] is not None:
        return True
    
    if hasattr(player, 'status') and player.status == 'lost':
        return True
    
    # Check if only one active player remains
    try:
        if 'players' in current_gameboard and current_gameboard['players'] is not None:
            active_players = sum(1 for p in current_gameboard['players'] if hasattr(p, 'status') and p.status != 'lost')
            if active_players <= 1:
                return True
    except (TypeError, AttributeError) as e:
        logger.error(f"Error checking active players: {e}")
        # Default to not done if we can't determine
        return False
    
    return False
