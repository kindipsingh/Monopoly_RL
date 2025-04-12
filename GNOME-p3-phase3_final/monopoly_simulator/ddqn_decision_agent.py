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
from monopoly_simulator import agent_helper_functions_v2 as agent_helper_functions
import pickle
import os

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
        
        replay_buffer_path = os.path.join("GNOME-p3-phase3_final", "monopoly_simulator", "replay_buffer.pkl")
        self.load_replay_buffer(replay_buffer_path)
        
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
    
    def make_buy_property_decision(self, player, current_gameboard, asset):
        """
        The agent decides to buy the property if:
        (i) it can 'afford' it. Our definition of afford is that we must have at least go_increment cash balance after
        the purchase.
        (ii) we can obtain a full color set through the purchase, and still have positive cash balance afterwards (though
        it may be less than go_increment).
        :param player: A Player instance. You should expect this to be the player that is 'making' the decision (i.e. the player
        instantiated with the functions specified by this decision agent).
        :param current_gameboard: A dict. The global data structure representing the current game board.
        :return: A Boolean. If True, then you decided to purchase asset from the bank, otherwise False. We allow you to
        purchase the asset even if you don't have enough cash; however, if you do you will end up with a negative
        cash balance and will have to handle that if you don't want to lose the game at the end of your move (see notes
        in handle_negative_cash_balance)
        """
        decision = False
        ####
        #if not hasattr(asset, 'price'):
        #    return(decision)
        ###
        rl_logger.info(f"Make decision is being set to {decision} ")

        if player.current_cash - asset.price >= current_gameboard['go_increment']:  # case 1: can we afford it?
            logger.debug(player.player_name+ ': I will attempt to buy '+ asset.name+ ' from the bank.')
            decision = True
        elif asset.price <= player.current_cash and agent_helper_functions.will_property_complete_set(player,asset,current_gameboard):
            logger.debug(player.player_name+ ': I will attempt to buy '+ asset.name+ ' from the bank.')
            decision = True

        
        return decision


    def make_bid(self, player, current_gameboard, asset, current_bid):
        """
        Decide the amount you wish to bid for asset in auction, given the current_bid that is currently going. If you don't
        return a bid that is strictly higher than current_bid you will be removed from the auction and won't be able to
        bid anymore. Note that it is not necessary that you are actually on the location on the board representing asset, since
        you will be invited to the auction automatically once a player who lands on a bank-owned asset rejects buying that asset
        (this could be you or anyone else).
        :param player: A Player instance. You should expect this to be the player that is 'making' the decision (i.e. the player
        instantiated with the functions specified by this decision agent).
        :param current_gameboard: A dict. The global data structure representing the current game board.
        :param asset: An purchaseable instance of Location (i.e. real estate, utility or railroad)
        :param current_bid: The current bid that is going in the auction. If you don't bid higher than this amount, the bank
        will remove you from the auction proceedings. You could also always return 0 to voluntarily exit the auction.
        :return: An integer that indicates what you wish to bid for asset
        """
        decision = False
        ####
        #if not hasattr(asset, 'price'):
        #    return(decision)
        ###
        if current_bid < asset.price:
            new_bid = current_bid + (asset.price-current_bid)/2
            if new_bid < player.current_cash:
                return new_bid
            else:   # We are aware that this can be simplified with a simple return 0 statement at the end. However in the final baseline agent
                    # the return 0's would be replaced with more sophisticated rules. Think of them as placeholders.
                return 0 # this will lead to a rejection of the bid downstream automatically
        elif current_bid < player.current_cash and agent_helper_functions.will_property_complete_set(player,asset,current_gameboard):
                # We are prepared to bid more than the price of the asset only if it doesn't result in insolvency, and
                    # if we can get a monopoly this way
            return current_bid+(player.current_cash-current_bid)/4
        else:
            return 0 # no reason to bid
    
    def handle_negative_cash_balance(self, player, current_gameboard):
        """
        You have a negative cash balance at the end of your move (i.e. your post-roll phase is over) and you must handle
        this issue before we move to the next player's pre-roll. If you do not succeed in restoring your cash balance to
        0 or positive, bankruptcy proceeds will begin and you will lost the game.
        The background agent tries a number of things to get itself out of a financial hole. First, it checks whether
        mortgaging alone can save it. If not, then it begins selling unimproved properties in ascending order of price, the idea being
        that it might as well get rid of cheap properties. This may not be the most optimal move but it is reasonable.
        If it ends up selling all unimproved properties and is still insolvent, it starts selling improvements, followed
        by a sale of the (now) unimproved properties.
        :param player: A Player instance. You should expect this to be the player that is 'making' the decision (i.e. the player
        instantiated with the functions specified by this decision agent).
        :param current_gameboard: A dict. The global data structure representing the current game board.
        :return: -1 if you do not try to address your negative cash balance, or 1 if you tried and believed you succeeded.
        Note that even if you do return 1, we will check to see whether you have non-negative cash balance. The rule of thumb
        is to return 1 as long as you 'try', or -1 if you don't try (in which case you will be declared bankrupt and lose the game)
        """
        if player.current_cash >= 0:   # prelim check to see if player has negative cash balance
            return (None, flag_config_dict['successful_action'])

        mortgage_potentials = list()
        max_sum = 0
        sorted_player_assets_list = self._set_to_sorted_list_assets(player.assets)
        for a in sorted_player_assets_list:
            if a.is_mortgaged:
                continue
            elif a.loc_class=='real_estate' and (a.num_houses>0 or a.num_hotels>0):
                continue
            else:
                mortgage_potentials.append((a,a.mortgage))
                max_sum += a.mortgage
        if mortgage_potentials and max_sum+player.current_cash >= 0: # if the second condition is not met, no point in mortgaging
            sorted_potentials = sorted(mortgage_potentials, key=lambda x: x[1])  # sort by mortgage in ascending order
            for p in sorted_potentials:
                if player.current_cash >= 0:
                    return (None, flag_config_dict['successful_action']) # we're done

                params = dict()
                params['player'] = player.player_name
                params['asset'] = p[0].name
                params['current_gameboard'] = "current_gameboard"
                logger.debug(player.player_name+ ': I am attempting to mortgage property '+ params['asset'])
                player.agent._agent_memory['previous_action'] = "mortgage_property"
                return ("mortgage_property", params)


        # if we got here, it means we're still in trouble. Next move is to sell unimproved properties. We don't check if
        # the total will cover our debts, since we're desperate at this point.

        # following sale potentials doesnot include properties from monopolized color groups
        sale_potentials = list()
        sorted_player_assets_list = self._set_to_sorted_list_assets(player.assets)
        for a in sorted_player_assets_list:
            if a.color in player.full_color_sets_possessed:
                continue
            elif a.is_mortgaged:
                sale_potentials.append((a, (a.price*current_gameboard['bank'].property_sell_percentage)-((1+current_gameboard['bank'].mortgage_percentage)*a.mortgage)))
            elif a.loc_class=='real_estate' and (a.num_houses>0 or a.num_hotels>0):
                continue
            else:
                sale_potentials.append((a,a.price*current_gameboard['bank'].property_sell_percentage))

        if sale_potentials: # if the second condition is not met, no point in mortgaging
            sorted_potentials = sorted(sale_potentials, key=lambda x: x[1])  # sort by mortgage in ascending order
            for p in sorted_potentials:
                if player.current_cash >= 0:
                    return (None, flag_config_dict['successful_action']) # we're done

                params = dict()
                params['player'] = player.player_name
                params['asset'] = p[0].name
                params['current_gameboard'] = "current_gameboard"
                logger.debug(player.player_name + ': I am attempting to sell property '+ p[0].name + ' to the bank')
                player.agent._agent_memory['previous_action'] = "sell_property"
                return ("sell_property", params)


        # if selling properties from non monopolized color groups doesnot relieve the player from debt, then only we start thinking about giving up monopolized groups.
        # If we come across a unimproved property which belongs to a monopoly, we still have to loop through the other properties from the same color group and
        # sell the houses and hotels first because we cannot sell this property when the color group has improved properties
        # We first check if selling houses and hotels one by one on the other improved properties of the same color group relieves the player of his debt. If it does
        # then we return without selling the current property else we sell the property and the player loses monopoly of that color group.
        sale_potentials = list()
        sorted_player_assets_list = self._set_to_sorted_list_assets(player.assets)
        for a in sorted_player_assets_list:
            if a.is_mortgaged:
                sale_potentials.append((a, (a.price*current_gameboard['bank'].property_sell_percentage)-((1+current_gameboard['bank'].mortgage_percentage)*a.mortgage)))
            elif a.loc_class=='real_estate' and (a.num_houses>0 or a.num_hotels>0):
                continue
            else:
                sale_potentials.append((a,a.price*current_gameboard['bank'].property_sell_percentage))

        if sale_potentials:
            sorted_potentials = sorted(sale_potentials, key=lambda x: x[1])  # sort by sell value in ascending order
            for p in sorted_potentials:
                if player.current_cash >= 0:
                    return (None, flag_config_dict['successful_action']) # we're done

                sorted_player_assets_list = self._set_to_sorted_list_assets(player.assets)
                for prop in sorted_player_assets_list:
                    if prop!=p[0] and prop.color==p[0].color and p[0].color in player.full_color_sets_possessed:
                        if hasattr(prop, 'num_hotels'):  # add by Peter, for composite novelty
                            if prop.num_hotels>0:
                                if player.current_cash >= 0:
                                    return (None, flag_config_dict['successful_action'])
                                params = dict()
                                params['player'] = player.player_name
                                params['asset'] = prop.name
                                params['current_gameboard'] = "current_gameboard"
                                params['sell_house'] = False
                                params['sell_hotel'] = True
                                logger.debug(player.player_name+ ': I am attempting to sell hotel on '+ prop.name + ' to the bank')
                                player.agent._agent_memory['previous_action'] = "sell_house_hotel"
                                return ("sell_house_hotel", params)

                            elif prop.num_houses>0:
                                if player.current_cash >= 0:
                                    return (None, flag_config_dict['successful_action'])
                                params = dict()
                                params['player'] = player.player_name
                                params['asset'] = prop.name
                                params['current_gameboard'] = "current_gameboard"
                                params['sell_house'] = True
                                params['sell_hotel'] = False
                                logger.debug(player.player_name+ ': I am attempting to sell house on '+ prop.name + ' to the bank')
                                player.agent._agent_memory['previous_action'] = "sell_house_hotel"
                                return ("sell_house_hotel", params)
                            else:
                                continue

                params = dict()
                params['player'] = player.player_name
                params['asset'] = p[0].name
                params['current_gameboard'] = "current_gameboard"
                logger.debug(player.player_name + ': I am attempting to sell property '+ p[0].name + ' to the bank')
                player.agent._agent_memory['previous_action'] = "sell_property"
                return ("sell_property", params)



        #we reach here if the player still hasnot cleared his debt. The above loop has now resulted in some more non monopolized properties.
        #Hence we have to go through the process of looping through these properties once again to decide on the potential properties that can be mortgaged or sold.

        mortgage_potentials = list()
        max_sum = 0
        sorted_player_assets_list = self._set_to_sorted_list_assets(player.assets)
        for a in sorted_player_assets_list:
            if a.is_mortgaged:
                continue
            elif a.loc_class=='real_estate' and (a.num_houses>0 or a.num_hotels>0):
                continue
            else:
                mortgage_potentials.append((a,a.mortgage))
                max_sum += a.mortgage
        if mortgage_potentials and max_sum+player.current_cash >= 0: # if the second condition is not met, no point in mortgaging
            sorted_potentials = sorted(mortgage_potentials, key=lambda x: x[1])  # sort by mortgage in ascending order
            for p in sorted_potentials:
                if player.current_cash >= 0:
                    return (None, flag_config_dict['successful_action']) # we're done

                params = dict()
                params['player'] = player.player_name
                params['asset'] = p[0].name
                params['current_gameboard'] = "current_gameboard"
                logger.debug(player.player_name+ ': I am attempting to mortgage property '+ params['asset'])
                player.agent._agent_memory['previous_action'] = "mortgage_property"
                return ("mortgage_property", params)

        # following sale potentials loops through the properties that have become unmonopolized due to the above loops and
        # doesnot include properties from monopolized color groups
        sale_potentials = list()
        sorted_player_assets_list = self._set_to_sorted_list_assets(player.assets)
        for a in sorted_player_assets_list:
            if a.color in player.full_color_sets_possessed:
                continue
            elif a.is_mortgaged:
                sale_potentials.append((a, (a.price*current_gameboard['bank'].property_sell_percentage)-((1+current_gameboard['bank'].mortgage_percentage)*a.mortgage)))
            elif a.loc_class=='real_estate' and (a.num_houses>0 or a.num_hotels>0):
                continue
            else:
                sale_potentials.append((a, a.price*current_gameboard['bank'].property_sell_percentage))

        if sale_potentials: # if the second condition is not met, no point in mortgaging
            sorted_potentials = sorted(sale_potentials, key=lambda x: x[1])  # sort by mortgage in ascending order
            for p in sorted_potentials:
                if player.current_cash >= 0:
                    return (None, flag_config_dict['successful_action']) # we're done

                params = dict()
                params['player'] = player.player_name
                params['asset'] = p[0].name
                params['current_gameboard'] = "current_gameboard"
                logger.debug(player.player_name + ': I am attempting to sell property '+ p[0].name + ' to the bank')
                player.agent._agent_memory['previous_action'] = "sell_property"
                return ("sell_property", params)

        count = 0
        # if we're STILL not done, then the only option is to start selling houses and hotels from the remaining improved monopolized properties, if we have 'em
        while (player.num_total_houses > 0 or player.num_total_hotels > 0) and count <3: # often times, a sale may not succeed due to uniformity requirements. We keep trying till everything is sold,
            # or cash balance turns non-negative.
            count += 1 # there is a slim chance that it is impossible to sell an improvement unless the player does something first (e.g., replace 4 houses with a hotel).
            # The count ensures we terminate at some point, regardless.
            sorted_assets_list = self._set_to_sorted_list_assets(player.assets)

            for a in sorted_assets_list:
                if a.loc_class == 'real_estate' and a.num_houses > 0:
                    if player.current_cash >= 0:
                        return (None, flag_config_dict['successful_action']) # we're done

                    params = dict()
                    params['player'] = player.player_name
                    params['asset'] = a.name
                    params['current_gameboard'] = "current_gameboard"
                    params['sell_house'] = True
                    params['sell_hotel'] = False
                    logger.debug(player.player_name+ ': I am attempting to sell house on '+ a.name + ' to the bank')
                    player.agent._agent_memory['previous_action'] = "sell_house_hotel"
                    return ("sell_house_hotel", params)

                elif a.loc_class == 'real_estate' and a.num_hotels > 0:
                    if player.current_cash >= 0:
                        return (None, flag_config_dict['successful_action']) # we're done
                    params = dict()
                    params['player'] = player.player_name
                    params['asset'] = a.name
                    params['current_gameboard'] = "current_gameboard"
                    params['sell_house'] = False
                    params['sell_hotel'] = True
                    logger.debug(player.player_name+ ': I am attempting to sell house on '+ a.name + ' to the bank')
                    player.agent._agent_memory['previous_action'] = "sell_house_hotel"
                    return ("sell_house_hotel", params)

        # final straw
        final_sale_assets = player.assets.copy()
        sorted_player_assets_list = self._set_to_sorted_list_assets(final_sale_assets)
        for a in sorted_player_assets_list:
            if player.current_cash >= 0:
                return (None, flag_config_dict['successful_action'])  # we're done
            params = dict()
            params['player'] = player.player_name
            params['asset'] = a.name
            params['current_gameboard'] = "current_gameboard"
            logger.debug(player.player_name + ': I am attempting to sell property '+ a.name + ' to the bank')
            player.agent._agent_memory['previous_action'] = "sell_property"
            return ("sell_property", params)

        return (None, flag_config_dict['successful_action']) # if we didn't succeed in establishing solvency, it will get caught by the simulator. Since we tried, we return 1.
    def _set_to_sorted_list_mortgaged_assets(self,player_mortgaged_assets):
        player_m_assets_list = list()
        player_m_assets_dict = dict()
        for item in player_mortgaged_assets:
            player_m_assets_dict[item.name] = item
        for sorted_key in sorted(player_m_assets_dict):
            player_m_assets_list.append(player_m_assets_dict[sorted_key])
        return player_m_assets_list


    def _set_to_sorted_list_assets(self,player_assets):
        player_assets_list = list()
        player_assets_dict = dict()
        for item in player_assets:
            player_assets_dict[item.name] = item
        for sorted_key in sorted(player_assets_dict):
            player_assets_list.append(player_assets_dict[sorted_key])
        return player_assets_list
    
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
             # IMPORTANT: Store the action_idx in the player object and agent memory
            player.last_action_idx = action_idx
            rl_logger.debug(f"{player.last_action_idx}")
        
            # Also store in agent memory if available
            if hasattr(player, 'agent') and player.agent is not None:
                player.agent.last_action_idx = action_idx
                rl_logger.debug(f"1")
                if not hasattr(player.agent, '_agent_memory') or player.agent._agent_memory is None:
                    player.agent._agent_memory = {}
                    rl_logger.debug(f"2")
                player.agent._agent_memory['last_action_idx'] = action_idx
            
            rl_logger.debug(f"{player.agent._agent_memory['last_action_idx']}")
        
            # Store in the DDQN agent for later training
            self.last_action_idx = action_idx
        
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
            elif action_name =="make_sell_property_offer":
                parameters = action_validator.validate_make_sell_property_offer(parameters, current_gameboard, rl_logger)
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
        # Base reward: Net worth calculation
        net_worth = player.current_cash
        property_value = 0
        
        # Calculate property values including houses and hotels
        for asset in player.assets:
            property_value += asset.price
            if hasattr(asset, 'num_houses'):
                property_value += asset.num_houses * asset.price_per_house
            if hasattr(asset, 'num_hotels') and asset.num_hotels > 0:
                property_value += asset.num_hotels * asset.price_per_house * 5
        
        # Calculate total net worth
        total_net_worth = net_worth + property_value
        
        # Base reward normalized by a factor to keep it manageable
        base_reward = total_net_worth / 10000.0
        
        # Additional rewards for strategic achievements
        strategic_reward = 0.0
        
        # Reward for monopolies (owning all properties of a color group)
        monopoly_reward = len(player.full_color_sets_possessed) * 5.0
        strategic_reward += monopoly_reward
        
        # Reward for property development (houses and hotels)
        development_reward = 0.0
        for asset in player.assets:
            if hasattr(asset, 'num_houses') and asset.num_houses > 0:
                development_reward += asset.num_houses * 0.5
            if hasattr(asset, 'num_hotels') and asset.num_hotels > 0:
                development_reward += asset.num_hotels * 3.0
        strategic_reward += development_reward
        
        # Reward for maintaining cash reserves (liquidity)
        liquidity_reward = 0.0
        if player.current_cash > 500:
            liquidity_reward = min(player.current_cash / 5000.0, 5.0)  # Cap at 5.0
        strategic_reward += liquidity_reward
        
        # Penalty for mortgaged properties
        mortgage_penalty = 0.0
        for asset in player.assets:
            if asset.is_mortgaged:
                mortgage_penalty += 0.5
        strategic_reward -= mortgage_penalty
        
        # Game state rewards/penalties
        game_state_reward = 0.0
        
        # Major reward for winning
        if 'winner' in current_gameboard and current_gameboard['winner'] == player.player_name:
            game_state_reward += 10
            rl_logger.info(f"Player {player.player_name} WON! Adding winning bonus of 100 to reward")
        
        # Major penalty for losing
        if player.status == 'lost':
            game_state_reward -= 10
            rl_logger.info(f"Player {player.player_name} LOST! Adding losing penalty of -50 to reward")
        
        # Calculate final reward
        final_reward = base_reward + strategic_reward + game_state_reward
        
        # Log detailed reward breakdown
        rl_logger.info(f"Reward calculation for {player.player_name}:")
        rl_logger.info(f"  Cash: ${player.current_cash}")
        rl_logger.info(f"  Property value: ${property_value}")
        rl_logger.info(f"  Net worth: ${total_net_worth}")
        rl_logger.info(f"  Base reward: {base_reward:.2f}")
        rl_logger.info(f"  Strategic reward: {strategic_reward:.2f}")
        rl_logger.info(f"    - Monopoly reward: {monopoly_reward:.2f}")
        rl_logger.info(f"    - Development reward: {development_reward:.2f}")
        rl_logger.info(f"    - Liquidity reward: {liquidity_reward:.2f}")
        rl_logger.info(f"    - Mortgage penalty: -{mortgage_penalty:.2f}")
        rl_logger.info(f"  Game state reward: {game_state_reward:.2f}")
        rl_logger.info(f"  Final reward: {final_reward:.2f}")
        
        return final_reward
    
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
    
    def save_replay_buffer(self, file_path):
        """Persist the agent's replay buffer to disk."""
        with open(file_path, "wb") as f:
            pickle.dump(self.ddqn_agent.replay_buffer.buffer, f)
        rl_logger.info(f"Replay buffer saved to {file_path}")

    def load_replay_buffer(self, file_path):
        """Load and replace the agent's replay buffer from disk if it exists."""
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                self.ddqn_agent.replay_buffer.buffer = pickle.load(f)
            rl_logger.info(f"Replay buffer loaded from {file_path}")
        else:
            rl_logger.info(f"No saved replay buffer found at {file_path}")

    def persist_replay_buffer(self):
        """Helper to easily save the replay buffer at a known location."""
        replay_buffer_path = os.path.join(base_dir, "monopoly_simulator", "replay_buffer.pkl")
        self.save_replay_buffer(replay_buffer_path)

ddqn_agent_instance = DDQNDecisionAgent()

def handle_negative_cash_balance(player, current_gameboard):
    if global_ddqn_agent:
        rl_logger.info(f"handle negative cash balance called")
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

def make_buy_property_decision(player, current_gameboard, asset):
    if global_ddqn_agent:
        rl_logger.info(f"make buy property decision called")
        decision=global_ddqn_agent.make_buy_property_decision(player, current_gameboard, asset)
        rl_logger.info(f"make buy property decision finally is {decision}")
        return decision
    return False

def make_bid(player, current_gameboard, asset, current_bid):
    if global_ddqn_agent:
        rl_logger.info(f"make bid called")
        return global_ddqn_agent.make_bid(player, current_gameboard, asset, current_bid)
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