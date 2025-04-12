from monopoly_simulator import initialize_game_elements
# from monopoly_simulator.action_choices import roll_die
from monopoly_simulator import action_choices
import numpy as np
from monopoly_simulator import card_utility_actions
from monopoly_simulator import background_agent_v3_1
from monopoly_simulator import ddqn_decision_agent
from monopoly_simulator import background_agent_v1_2
from monopoly_simulator import read_write_current_state
import json
from monopoly_simulator import novelty_generator
from monopoly_simulator import diagnostics
from monopoly_simulator.agent import Agent
import xlsxwriter
from monopoly_simulator.flag_config import flag_config_dict
from monopoly_simulator.logging_info import log_file_create
import os
import time
import sys
from monopoly_simulator.server_agent_serial import ServerAgent
#from monopoly_simulator.random_novelty_generation import *
import logging
import copy
### for reimport in set_up_board
import importlib
import monopoly_simulator
from monopoly_simulator import agent_helper_functions_v2 as agent_helper_functions
from monopoly_simulator import location
from monopoly_simulator import player
from monopoly_simulator.player import Player
###
from action_vector_logger import ActionVectorLogger, integrate_with_gameplay
from monopoly_simulator import action_choices, diagnostics, card_utility_actions, read_write_current_state
from importlib import resources
from monopoly_simulator import action_choices, diagnostics, card_utility_actions, read_write_current_state
from monopoly_simulator.monopoly_state_encoder import MonopolyStateEncoder
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
file_handler = logging.FileHandler("gameplay_socket_phase3.log")
file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

encoded_logger = logging.getLogger("encoded_state")
encoded_logger.setLevel(logging.DEBUG)

# Remove any existing handlers if present (to avoid duplicate logs).
if encoded_logger.hasHandlers():
    encoded_logger.handlers.clear()

# Create the log directory if it doesn't exist
log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "single_tournament"))
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Create a FileHandler for seed_6_encoded.log that opens in write mode.
encoded_handler = logging.FileHandler(os.path.join(log_dir, "seed_6_encoded.log"), mode="a")
encoded_handler.setLevel(logging.DEBUG)

# Set a formatter and add it to the handler.
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
encoded_handler.setFormatter(formatter)
encoded_logger.addHandler(encoded_handler)

# Test the logger.
encoded_logger.debug("Encoded logger is now configured to rewrite file on each run.")

# #
# nov = "trade_within_incoming_group"
# meta_seed = 15
# num_games = 40
# novelty_index = 12
# #ignore_list = list(range(1,13))
# ignore_list = []
# wrong_para_test = False
# #
def cleanup_loggers():
    for handler in logger.handlers:
        handler.flush()
        handler.close()
    for handler in encoded_logger.handlers:
        handler.flush()
        handler.close()
    logger.debug("Loggers cleaned up successfully")

def write_history_to_file(game_board, workbook):
    worksheet = workbook.add_worksheet()
    col = 0
    for key in game_board['history']:
        if key == 'param':
            col += 1
            row = 0
            worksheet.write(row, col, key)
            worksheet.write(row, col + 1, 'current_player')
            for item in game_board['history'][key]:
                worksheet.write(row + 1, col, str(item))
                try:
                    worksheet.write(row + 1, col + 1, item['player'].player_name)
                except:
                    pass
                row += 1
            col += 1
        else:
            col += 1
            row = 0
            worksheet.write(row, col, key)
            for item in game_board['history'][key]:
                worksheet.write(row + 1, col, str(item))
                row += 1
    workbook.close()
    print("History logged into history_log.xlsx file.")


def disable_history(game_elements):
    game_elements['history'] = dict()
    game_elements['history']['function'] = list()
    game_elements['history']['param'] = list()
    game_elements['history']['return'] = list()
    game_elements['history']['time_step'] = list()


def simulate_game_instance(game_elements, history_log_file=None, np_seed=7, state_logger=None):
    """
    Simulate a game instance with replay buffer collection for the RL agent (player_3).
    
    :param game_elements: Dictionary representing the current game state.
    :param history_log_file: Optional Excel workbook for logging game history.
    :param np_seed: Numpy seed for randomness.
    :param state_logger: Callback function for logging the current state. Expects game_elements.copy() as input.
    :return: The winner's name if the game terminates naturally.
    """
    # Import the replay buffer module
    from monopoly_simulator.replay_buffer_module import ReplayBuffer, calculate_reward, get_action_index, is_episode_done, add_to_replay_buffer, store_player_decision
    from monopoly_simulator.ddqn_decision_agent import ddqn_agent_instance
    from monopoly_simulator.action_encoding import ActionEncoder
    
    # Initialize replay buffer for player_3 (RL agent)
    rl_agent_name = 'player_3'
    replay_buffer = ReplayBuffer(capacity=100000)  # Adjust capacity as needed
    
    # Store the replay buffer in game_elements so it can be accessed by other components
    game_elements['replay_buffer'] = replay_buffer
    
    # Initialize the action encoder for building the full action mapping
    action_encoder = ActionEncoder()
    
    # Variables to track state and action for the RL agent
    current_state = None
    episode_done = False
    
    # Create a function to track individual actions for the RL agent
    def track_action(player, action_name, game_elements):
        nonlocal current_state
        
        if player.player_name != rl_agent_name or current_state is None:
            return
            
        # Encode the state after the action
        next_state = encoder.encode_state(game_elements).numpy()[0]
        
        # Calculate reward
        reward = calculate_reward(player, game_elements)
        
        # Check if episode is done
        done = is_episode_done(player, game_elements)
        
        # Get the action index from the full mapping
        action_idx = get_action_index(player, game_elements)
        
        # Store the action in the player for debugging
        player.last_action_name = action_name
        
        # Add experience to replay buffer
        replay_buffer.add(current_state, action_idx, reward, next_state, done)
        logger.debug(f"Added individual action experience to replay buffer for {rl_agent_name}, action: {action_name} (idx: {action_idx}), reward: {reward:.2f}")
        
        # Update current state for next action
        current_state = next_state
        
        # If episode is done, reset current state
        if done:
            current_state = None
            nonlocal episode_done
            episode_done = True
    
    # Monkey patch the _execute_action method of Player to track actions
    original_execute_action = player.Player._execute_action
    
    def execute_action_with_tracking(self, action_to_execute, parameters, current_gameboard):
        # Get the action name (function name)
        action_name = action_to_execute.__name__ if callable(action_to_execute) else action_to_execute
        
        # Log the action before execution
        logger.debug(f"Player {self.player_name} executing action: {action_name}")
        
        # If this is the RL agent and we have a current state, capture the state before action
        if self.player_name == rl_agent_name and not episode_done:
            nonlocal current_state
            if current_state is None:
                current_state = encoder.encode_state(current_gameboard).numpy()[0]
        
        # Execute the original action
        result = original_execute_action(self, action_to_execute, parameters, current_gameboard)
        
        # Track the action for the RL agent if it's a meaningful action
        if self.player_name == rl_agent_name and not episode_done:
            # Build the full action mapping to get the correct index
            track_action(self, action_name, current_gameboard)
        
        return result
    
    # Apply the monkey patch
    player.Player._execute_action = execute_action_with_tracking
    
    logger.debug("size of board " + str(len(game_elements['location_sequence'])))
    np.random.seed(np_seed)
    np.random.shuffle(game_elements['players'])
    game_elements['seed'] = np_seed
    game_elements['card_seed'] = np_seed
    game_elements['choice_function'] = np.random.choice
    count_json = 0
    num_die_rolls = 0
    tot_time = 0

    logger.debug('players will play in the following order: ' + '->'.join([p.player_name for p in game_elements['players']]))
    logger.debug('Beginning play. Rolling first die...')
    current_player_index = 0
    num_active_players = len(game_elements['players'])
    winner = None
    workbook = None
    if history_log_file:
        workbook = xlsxwriter.Workbook(history_log_file)
    game_elements['start_time'] = time.time()
    game_elements['time_step_indicator'] = 0
    timeout = game_elements['start_time'] + 60 * 60 * 3

    # Initialize the action vector logger
    vector_logger = integrate_with_gameplay(game_elements, history_log_file)
    
    # Initialize state encoder for replay buffer
    encoder = MonopolyStateEncoder()

    while num_active_players > 1 and time.time() <= timeout:
        current_player = game_elements['players'][current_player_index]
        while current_player.status == 'lost':
            current_player_index = (current_player_index + 1) % len(game_elements['players'])
            current_player = game_elements['players'][current_player_index]
        current_player.status = 'current_move'
        
        # Encode state for logging
        encoded_state = encoder.encode_state(game_elements)
        encoded_logger.debug("\n=== Current State Vector ===\n%s", encoded_state.numpy())
        
        if state_logger:
            state_logger(copy.deepcopy(game_elements))
        
        # Record current state for RL agent (player_3)
        if current_player.player_name == rl_agent_name:
            current_state = encoded_state.numpy()[0]
            logger.debug(f"Recorded state for {rl_agent_name}")
        
        # Log allowable pre-roll actions for debugging
        pre_roll_allowable = current_player.compute_allowable_pre_roll_actions(game_elements)
        logger.debug(f"Pre-roll allowable actions for {current_player.player_name}: {pre_roll_allowable}")
           
        # Pre-roll phase for current player
        skip_turn = 0
        pre_roll_code = current_player.make_pre_roll_moves(game_elements)
        
        # After pre-roll moves, log the pre-roll action vector
        vector_logger.log_pre_roll(current_player, game_elements)
        
        # If this was the RL agent's turn and the last action was a conclude action, record it
        if current_player.player_name == rl_agent_name and current_state is not None:
            # Only add the conclude action if it wasn't already added by the tracking function
            if not hasattr(current_player, 'last_action_name') or current_player.last_action_name != 'concluded_actions':
                track_action(current_player, 'concluded_actions', game_elements)
        
        if pre_roll_code == 2:
            skip_turn += 1
        
        # Out-of-turn moves
        out_of_turn_player_index = current_player_index + 1
        out_of_turn_count = 0
        while skip_turn != num_active_players and out_of_turn_count <= 5:
            out_of_turn_count += 1
            out_of_turn_player = game_elements['players'][out_of_turn_player_index % len(game_elements['players'])]
            if out_of_turn_player.status == 'lost':
                out_of_turn_player_index += 1
                continue
                
            # Ensure out-of-turn player has allowable action methods
            if not hasattr(out_of_turn_player, 'compute_allowable_out_of_turn_actions'):
                def default_oot_actions(game_elements):
                    actions = ['skip_turn']
                    logger.debug(f"Default allowable out-of-turn actions for {out_of_turn_player.player_name}: {actions}")
                    return actions
                out_of_turn_player.compute_allowable_out_of_turn_actions = default_oot_actions
                logger.debug(f"Assigned default compute_allowable_out_of_turn_actions for {out_of_turn_player.player_name}")
            
            # Record state for RL agent if it's their turn
            if out_of_turn_player.player_name == rl_agent_name:
                current_state = encoder.encode_state(game_elements).numpy()[0]
                logger.debug(f"Recorded out-of-turn state for {rl_agent_name}")
            
            # Log allowable out-of-turn actions for debugging
            oot_allowable = out_of_turn_player.compute_allowable_out_of_turn_actions(game_elements)
            logger.debug(f"Out-of-turn allowable actions for {out_of_turn_player.player_name}: {oot_allowable}")
            
            oot_code = out_of_turn_player.make_out_of_turn_moves(game_elements)
            
            # Log the action vector for out-of-turn moves
            vector_logger.log_out_of_turn(out_of_turn_player, game_elements)
            
            # If this was the RL agent's turn and the last action was a skip action, record it
            if out_of_turn_player.player_name == rl_agent_name and current_state is not None:
                # Only add the skip action if it wasn't already added by the tracking function
                if not hasattr(out_of_turn_player, 'last_action_name') or out_of_turn_player.last_action_name != 'skip_turn':
                    track_action(out_of_turn_player, 'skip_turn', game_elements)
            
            if state_logger:
                state_logger(game_elements.copy())
            if oot_code == 2:
                skip_turn += 1
            else:
                skip_turn = 0
            out_of_turn_player_index += 1

        # Dice roll phase
        r = action_choices.roll_die(game_elements['dies'], np.random.choice, game_elements)
        for i in range(len(r)):
            game_elements['die_sequence'][i].append(r[i])
        num_die_rolls += 1
        game_elements['current_die_total'] = sum(r)
        logger.debug('dice have come up: ' + str(r))
        
        # If this is the RL agent's turn, record the roll_die action
        if current_player.player_name == rl_agent_name and current_state is not None:
            track_action(current_player, 'roll_die', game_elements)
        
        if state_logger:
            state_logger(copy.deepcopy(game_elements))
            
        # Movement and consequence phase if not in jail
        if not current_player.currently_in_jail:
            check_for_go = True
            game_elements['move_player_after_die_roll'](current_player, sum(r), game_elements, check_for_go)
            if state_logger:
                state_logger(copy.deepcopy(game_elements))
            current_player.process_move_consequences(game_elements)
            if state_logger:
                state_logger(copy.deepcopy(game_elements))
                
            # Ensure player has allowable post-roll action methods
            if not hasattr(current_player, 'compute_allowable_post_roll_actions'):
                current_player.compute_allowable_post_roll_actions = lambda ge: ['buy_property']
                logger.debug(f"Assigned default compute_allowable_post_roll_actions for {current_player.player_name}")
            
            # Record state for RL agent if it's their turn
            if current_player.player_name == rl_agent_name:
                current_state = encoder.encode_state(game_elements).numpy()[0]
                logger.debug(f"Recorded post-roll state for {rl_agent_name}")
            
            # Log allowable post-roll actions for debugging
            post_roll_allowable = current_player.compute_allowable_post_roll_actions(game_elements)
            logger.debug(f"Post-roll allowable actions for {current_player.player_name}: {post_roll_allowable}")
            
            current_player.make_post_roll_moves(game_elements)
            
            # Log the post-roll action vector after moves
            vector_logger.log_post_roll(current_player, game_elements)
            
            # If this was the RL agent's turn and the last action was a conclude action, record it
            if current_player.player_name == rl_agent_name and current_state is not None:
                # Only add the conclude action if it wasn't already added by the tracking function
                if not hasattr(current_player, 'last_action_name') or current_player.last_action_name != 'concluded_actions':
                    track_action(current_player, 'concluded_actions', game_elements)
            
            if state_logger:
                state_logger(copy.deepcopy(game_elements))
        else:
            card_utility_actions.set_currently_in_jail_to_false(current_player, game_elements)

        # Negative cash handling
        if current_player.current_cash < 0:
            code = current_player.handle_negative_cash_balance(game_elements)
            if state_logger:
                state_logger(game_elements.copy())
            if code == card_utility_actions.flag_config_dict['failure_code'] or current_player.current_cash < 0:
                current_player.begin_bankruptcy_proceedings(game_elements)
                num_active_players -= 1
                diagnostics.print_asset_owners(game_elements)
                diagnostics.print_player_cash_balances(game_elements)
                
                # If the RL agent went bankrupt, record this as a terminal state
                if current_player.player_name == rl_agent_name and current_state is not None:
                    # Get next state after bankruptcy
                    next_state = encoder.encode_state(game_elements).numpy()[0]
                    
                    # Large negative reward for bankruptcy
                    reward = calculate_reward(current_player, game_elements)
                    
                    # This is a terminal state
                    done = True
                    
                    # Add experience to replay buffer using the proper action index
                    action_idx = get_action_index(current_player, game_elements)
                    replay_buffer.add(current_state, action_idx, reward, next_state, done)
                    logger.debug(f"Added bankruptcy experience to replay buffer for {rl_agent_name}, action index: {action_idx}, reward: {reward:.2f}")
                    
                    # Reset current state
                    current_state = None
                    episode_done = True
                
                if num_active_players == 1:
                    for p in game_elements['players']:
                        if p.status != 'lost':
                            winner = p
                            p.status = 'won'
                            game_elements['winner'] = p.player_name
                            
                            # If the RL agent won, record this as a terminal state with high reward
                            if p.player_name == rl_agent_name and current_state is not None:
                                # Get next state after winning
                                next_state = encoder.encode_state(game_elements).numpy()[0]
                                
                                # Calculate reward for winning
                                reward = calculate_reward(p, game_elements)
                                
                                # This is a terminal state
                                done = True
                                
                                # Add experience to replay buffer using the proper action index
                                action_idx = get_action_index(p, game_elements)
                                replay_buffer.add(current_state, action_idx, reward, next_state, done)
                                logger.debug(f"Added winning experience to replay buffer for {rl_agent_name}, action index: {action_idx}, reward: {reward:.2f}")
                                
                                # Reset current state
                                current_state = None
                                episode_done = True
            else:
                current_player.status = 'waiting_for_move'
        else:
            current_player.status = 'waiting_for_move'

        current_player_index = (current_player_index + 1) % len(game_elements['players'])
        tot_time = time.time() - game_elements['start_time']
        count_json += 1

    logger.debug('Liquid Cash remaining with Bank = ' + str(game_elements['bank'].total_cash_with_bank))
    if workbook:
        read_write_current_state.write_history_to_file(game_elements, workbook)
    for handler in vector_logger.logger.handlers:
        handler.close()
        vector_logger.logger.removeHandler(handler)

    # Restore the original _execute_action method
    player.Player._execute_action = original_execute_action

    diagnostics.print_asset_owners(game_elements)
    diagnostics.print_player_cash_balances(game_elements)
    diagnostics.print_player_net_worths(game_elements)
    logger.debug("Game ran for " + str(tot_time) + " seconds.")
    
    # Save the replay buffer to a file
    buffer_stats = replay_buffer.get_stats()
    logger.info(f"Replay buffer stats: size={buffer_stats['size']}, capacity={buffer_stats['capacity']}")
    logger.info(f"Average reward: {buffer_stats['avg_reward']:.4f}")
    if buffer_stats['episode_rewards']:
        logger.info(f"Episode rewards: {buffer_stats['episode_rewards']}")
    
    # Save the replay buffer to a file
    buffer_filepath = f"../single_tournament/replay_buffer_seed_{np_seed}.pkl"
    replay_buffer.save_to_file(buffer_filepath)
    logger.info(f"Replay buffer saved to {buffer_filepath}")
    
    if winner:
        logger.debug('We have a winner: ' + winner.player_name)
        return winner.player_name
    else:
        winner = card_utility_actions.check_for_winner(game_elements)
        if winner is not None:
            logger.debug('We have a winner: ' + winner.player_name)
            return winner.player_name
        else:
            logger.debug('Game has no winner, do not know what went wrong!!!')
            return None

def set_up_board(game_schema_file_path, player_decision_agents):
    """
    ## reimporting for avoid influence between tournaments
    importlib.reload(agent_helper_functions)
    importlib.reload(action_choices)
    importlib.reload(card_utility_actions)
    #del sys.modules['monopoly_simulator.location']
    #from monopoly_simulator.location import Location, RealEstateLocation, RailroadLocation, UtilityLocation

    import monopoly_simulator.location
    importlib.reload(monopoly_simulator.location)
    import monopoly_simulator;
    importlib.reload(monopoly_simulator);
    from monopoly_simulator import location

    import monopoly_simulator.player
    importlib.reload(monopoly_simulator.player)
    import monopoly_simulator
    importlib.reload(monopoly_simulator)
    from monopoly_simulator import player

    #del sys.modules['monopoly_simulator.player']
    #from monopoly_simulator.player import Player
    #print(sys.modules)
    #del sys.modules['monopoly_simulator.location']
    #del sys.modules['monopoly_simulator.player']
    #import monopoly_simulator.location
    #from monopoly_simulator.player import Player
    #p = Player()
    #p.process_move_consequences = getattr(sys.modules[__name__], 'Player.process_move_consequences')
    #importlib.reload(player)
    #print(action_choices.buy_property.__module__)
    #print(Player.process_move_consequences.__module__)
    """
    ##
    # game_schema = json.load(open(game_schema_file_path, 'r'))
    with resources.open_text("monopoly_simulator", game_schema_file_path) as file:
        game_schema = json.load(file) 

    return initialize_game_elements.initialize_board(game_schema, player_decision_agents)


def inject_novelty(current_gameboard, novelty_schema=None):
    """
    Function for illustrating how we inject novelty
    ONLY FOR ILLUSTRATIVE PURPOSES
    :param current_gameboard: the current gameboard into which novelty will be injected. This gameboard will be modified
    :param novelty_schema: the novelty schema json, read in from file. It is more useful for running experiments at scale
    rather than in functions like these. For the most part, we advise writing your novelty generation routines, just like
    we do below, and for using the novelty schema for informational purposes (i.e. for making sense of the novelty_generator.py
    file and its functions.
    :return: None
    """

    ###Below are examples of Level 1, Level 2 and Level 3 Novelties
    ###Uncomment only the Level of novelty that needs to run (i.e, either Level1 or Level 2 or Level 3). Do not mix up novelties from different levels.

    '''
    #Level 1 Novelty

    numberDieNovelty = novelty_generator.NumberClassNovelty()
    numberDieNovelty.die_novelty(current_gameboard, 4, die_state_vector=[[1,2,3,4,5],[1,2,3,4],[5,6,7],[2,3,4]])
    
    classDieNovelty = novelty_generator.TypeClassNovelty()
    die_state_distribution_vector = ['uniform','uniform','biased','biased']
    die_type_vector = ['odd_only','even_only','consecutive','consecutive']
    classDieNovelty.die_novelty(current_gameboard, die_state_distribution_vector, die_type_vector)
    
    classCardNovelty = novelty_generator.TypeClassNovelty()
    novel_cc = dict()
    novel_cc["street_repairs"] = "alternate_contingency_function_1"
    novel_chance = dict()
    novel_chance["general_repairs"] = "alternate_contingency_function_1"
    classCardNovelty.card_novelty(current_gameboard, novel_cc, novel_chance)
    '''

    '''
    #Level 2 Novelty

    #The below combination reassigns property groups and individual properties to different colors.
    #On playing the game it is verified that the newly added property to the color group is taken into account for monopolizing a color group,
    # i,e the orchid color group now has Baltic Avenue besides St. Charles Place, States Avenue and Virginia Avenue. The player acquires a monopoly
    # only on the ownership of all the 4 properties in this case.
    
    inanimateNovelty = novelty_generator.InanimateAttributeNovelty()
    inanimateNovelty.map_property_set_to_color(current_gameboard, [current_gameboard['location_objects']['Park Place'], current_gameboard['location_objects']['Boardwalk']], 'Brown')
    inanimateNovelty.map_property_to_color(current_gameboard, current_gameboard['location_objects']['Baltic Avenue'], 'Orchid')

    #setting new rents for Indiana Avenue
    inanimateNovelty.rent_novelty(current_gameboard['location_objects']['Indiana Avenue'], {'rent': 50, 'rent_1_house': 150})
    '''

    '''
    #Level 3 Novelty

    granularityNovelty = novelty_generator.GranularityRepresentationNovelty()
    granularityNovelty.granularity_novelty(current_gameboard, current_gameboard['location_objects']['Baltic Avenue'], 6)
    granularityNovelty.granularity_novelty(current_gameboard, current_gameboard['location_objects']['States Avenue'], 20)
    granularityNovelty.granularity_novelty(current_gameboard, current_gameboard['location_objects']['Tennessee Avenue'], 27)

    spatialNovelty = novelty_generator.SpatialRepresentationNovelty()
    spatialNovelty.color_reordering(current_gameboard, ['Boardwalk', 'Park Place'], 'Blue')

    granularityNovelty.granularity_novelty(current_gameboard, current_gameboard['location_objects']['Park Place'], 52)
    '''



def play_game(inject_novelty_function=None):
    """
    Use this function if you want to test a single game instance and control lots of things. For experiments, we will directly
    call some of the functions in gameplay from test_harness.py.

    This is where everything begins. Assign decision agents to your players, set up the board and start simulating! You can
    control any number of players you like, and assign the rest to the simple agent. We plan to release a more sophisticated
    but still relatively simple agent soon.
    :return: String. the name of the player who won the game, if there was a winner, otherwise None.
    """

    # Create the log directory if it doesn't exist
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "single_tournament"))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print('Creating folder and logging gameplay.')
    else:
        print('Logging gameplay.')

    logger = log_file_create('../single_tournament/seed_6.log')
    player_decision_agents = dict()
    # for p in ['player_1','player_3']:
    #     player_decision_agents[p] = simple_decision_agent_1.decision_agent_methods

    agent = ServerAgent()
    f_name = 'play game without novelty'
    if not agent.start_tournament(f_name):
        print("Unable to start tournament")
        exit(0)
    else:
        pass

    # player_decision_agents['player_1'] = Agent(**background_agent_v3_1.decision_agent_methods)
    player_decision_agents['player_1'] = agent
    player_decision_agents['player_2'] = Agent(**background_agent_v3_1.decision_agent_methods)
    player_decision_agents['player_3'] = Agent(**background_agent_v3_1.decision_agent_methods)
    player_decision_agents['player_4'] = Agent(**background_agent_v3_1.decision_agent_methods)

    game_elements = set_up_board('./monopoly_game_schema_v1-2.json',
                                 player_decision_agents)

    # game_elements = set_up_board('monopoly_game_schema_v1-2.json',
    #                              player_decision_agents)

    #Comment out the above line and uncomment the piece of code to read the gameboard state from an existing json file so that
    #the game starts from a particular game state instead of initializing the gameboard with default start values.
    #Note that the novelties introduced in that particular game which was saved to file will be loaded into this game board as well.
    '''
    logger.debug("Loading gameboard from an existing game state that was saved to file.")
    infile = '../current_gameboard_state.json'
    game_elements = read_write_current_state.read_in_current_state_from_file(infile, player_decision_agents)
    '''
    if inject_novelty_function:
         inject_novelty_function(game_elements)

    if player_decision_agents['player_1'].startup(game_elements) == flag_config_dict['failure_code'] or \
            player_decision_agents['player_2'].startup(game_elements) == flag_config_dict['failure_code'] or \
            player_decision_agents['player_3'].startup(game_elements) == flag_config_dict['failure_code'] or \
            player_decision_agents['player_4'].startup(game_elements) == flag_config_dict['failure_code']:
        logger.error("Error in initializing agents. Cannot play the game.")
        return None
    else:
        logger.debug("Sucessfully initialized all player agents.")
        winner = simulate_game_instance(game_elements)
        if player_decision_agents['player_1'].shutdown() == flag_config_dict['failure_code'] or \
            player_decision_agents['player_2'].shutdown() == flag_config_dict['failure_code'] or \
            player_decision_agents['player_3'].shutdown() == flag_config_dict['failure_code'] or \
            player_decision_agents['player_4'].shutdown() == flag_config_dict['failure_code']:
            logger.error("Error in agent shutdown.")
            handlers_copy = logger.handlers[:]
            for handler in handlers_copy:
                logger.removeHandler(handler)
                handler.close()
                handler.flush()
            return None
        else:
            logger.debug("All player agents have been shutdown. ")
            logger.debug("GAME OVER")
            handlers_copy = logger.handlers[:]
            #for handler in handlers_copy:
             #   logger.removeHandler(handler)
              ## handler.flush()
            agent.end_tournament()
            cleanup_loggers()
            return winner


def play_game_in_tournament(game_seed, novelty_info=False, inject_novelty_function=None):
    logger.debug('seed used: ' + str(game_seed))
    player_decision_agents = dict()
    # for p in ['player_1','player_3']:
    #     player_decision_agents[p] = simple_decision_agent_1.decision_agent_methods
    player_decision_agents['player_1'] = Agent(**background_agent_v3_1.decision_agent_methods)
    player_decision_agents['player_2'] = Agent(**background_agent_v3_1.decision_agent_methods)
    player_decision_agents['player_3'] = Agent(**background_agent_v3_1.decision_agent_methods)
    player_decision_agents['player_4'] = Agent(**background_agent_v3_1.decision_agent_methods)

    game_elements = set_up_board('./monopoly_game_schema_v1-2.json',
                                 player_decision_agents)

    #Comment out the above line and uncomment the piece of code to read the gameboard state from an existing json file so that
    #the game starts from a particular game state instead of initializing the gameboard with default start values.
    #Note that the novelties introduced in that particular game which was saved to file will be loaded into this game board as well.
    '''
    logger.debug("Loading gameboard from an existing game state that was saved to file.")
    infile = '../current_gameboard_state.json'
    game_elements = read_write_current_state.read_in_current_state_from_file(infile, player_decision_agents)
    '''

    if inject_novelty_function:
        inject_novelty_function(game_elements)

    if not novelty_info:
        if player_decision_agents['player_1'].startup(game_elements) == flag_config_dict['failure_code'] or \
                player_decision_agents['player_2'].startup(game_elements) == flag_config_dict['failure_code'] or \
                player_decision_agents['player_3'].startup(game_elements) == flag_config_dict['failure_code'] or \
                player_decision_agents['player_4'].startup(game_elements) == flag_config_dict['failure_code']:
            logger.error("Error in initializing agents. Cannot play the game.")
            return None
        else:
            logger.debug("Sucessfully initialized all player agents.")
            winner = simulate_game_instance(game_elements, history_log_file=None, np_seed=game_seed)
            if player_decision_agents['player_1'].shutdown() == flag_config_dict['failure_code'] or \
                    player_decision_agents['player_2'].shutdown() == flag_config_dict['failure_code'] or \
                    player_decision_agents['player_3'].shutdown() == flag_config_dict['failure_code'] or \
                    player_decision_agents['player_4'].shutdown() == flag_config_dict['failure_code']:
                logger.error("Error in agent shutdown.")
                return None
            else:
                logger.debug("All player agents have been shutdown. ")
                logger.debug("GAME OVER")
                cleanup_loggers()
                return winner
    else:
        if inject_novelty_function:
            if player_decision_agents['player_1'].startup(game_elements, indicator=True) == flag_config_dict['failure_code'] or \
                    player_decision_agents['player_2'].startup(game_elements, indicator=True) == flag_config_dict['failure_code'] or \
                    player_decision_agents['player_3'].startup(game_elements, indicator=True) == flag_config_dict['failure_code'] or \
                    player_decision_agents['player_4'].startup(game_elements, indicator=True) == flag_config_dict['failure_code']:
                logger.error("Error in initializing agents. Cannot play the game.")
                return None
            else:
                logger.debug("Sucessfully initialized all player agents.")
                winner = simulate_game_instance(game_elements, history_log_file=None, np_seed=game_seed)
                if player_decision_agents['player_1'].shutdown() == flag_config_dict['failure_code'] or \
                        player_decision_agents['player_2'].shutdown() == flag_config_dict['failure_code'] or \
                        player_decision_agents['player_3'].shutdown() == flag_config_dict['failure_code'] or \
                        player_decision_agents['player_4'].shutdown() == flag_config_dict['failure_code']:
                    logger.error("Error in agent shutdown.")
                    return None
                else:
                    logger.debug("All player agents have been shutdown. ")
                    logger.debug("GAME OVER")
                    cleanup_loggers()
                    return winner
        else:
            if player_decision_agents['player_1'].startup(game_elements, indicator=False) == flag_config_dict['failure_code'] or \
                    player_decision_agents['player_2'].startup(game_elements, indicator=False) == flag_config_dict['failure_code'] or \
                    player_decision_agents['player_3'].startup(game_elements, indicator=False) == flag_config_dict['failure_code'] or \
                    player_decision_agents['player_4'].startup(game_elements, indicator=False) == flag_config_dict['failure_code']:
                logger.error("Error in initializing agents. Cannot play the game.")
                return None
            else:
                logger.debug("Sucessfully initialized all player agents.")
                winner = simulate_game_instance(game_elements, history_log_file=None, np_seed=game_seed)
                if player_decision_agents['player_1'].shutdown() == flag_config_dict['failure_code'] or \
                        player_decision_agents['player_2'].shutdown() == flag_config_dict['failure_code'] or \
                        player_decision_agents['player_3'].shutdown() == flag_config_dict['failure_code'] or \
                        player_decision_agents['player_4'].shutdown() == flag_config_dict['failure_code']:
                    logger.error("Error in agent shutdown.")
                    return None
                else:
                    logger.debug("All player agents have been shutdown. ")
                    logger.debug("GAME OVER")
                    cleanup_loggers()
                    return winner


def play_game_in_tournament_socket( game_seed, agent1, agent2, agent3, agent4, novelty_info = False, inject_novelty_function=False):
    """
    Use this function if you want to test a single game instance and control lots of things. For experiments, we will directly
    call some of the functions in gameplay from test_harness.py.

    This is where everything begins. Assign decision agents to your players, set up the board and start simulating! You can
    control any number of players you like, and assign the rest to the simple agent. We plan to release a more sophisticated
    but still relatively simple agent soon.
    :return: String. the name of the player who won the game, if there was a winner, otherwise None.
    """
    print("using seed: ", game_seed)
    try:
        os.makedirs('../single_tournament/')
        print('Creating folder and logging gameplay.')
    except:
        print('Logging gameplay.')

    logger = log_file_create('../single_tournament/seed_6.log')
    player_decision_agents = dict()

    if agent1 is not None:
        player_decision_agents['player_1'] = agent1
    else:
        player_decision_agents['player_1'] = Agent(**background_agent_v3_1.decision_agent_methods)
    player_decision_agents['player_2'] = Agent(**agent2.decision_agent_methods)
    player_decision_agents['player_3'] = Agent(**agent3.decision_agent_methods)
    player_decision_agents['player_4'] = Agent(**agent4.decision_agent_methods)

    #print(Player.process_move_consequences.__module__)
    #print(action_choices.buy_property.__module__)
    game_elements = set_up_board('./monopoly_game_schema_v1-2.json',
                                 player_decision_agents)

    #Comment out the above line and uncomment the piece of code to read the gameboard state from an existing json file so that
    #the game starts from a particular game state instead of initializing the gameboard with default start values.
    #Note that the novelties introduced in that particular game which was saved to file will be loaded into this game board as well.
    '''
    logger.debug("Loading gameboard from an existing game state that was saved to file.")
    infile = '../current_gameboard_state.json'
    game_elements = read_write_current_state.read_in_current_state_from_file(infile, player_decision_agents)
    '''
    class_name, func_name, arg_value, meta_seed_value = False, False, False, False
    if inject_novelty_function==True:
        pass
    elif inject_novelty_function:
        # -------
        game_elements['novelty_xxxx'] = True
        # -------
        class_name, func_name, arg_value, meta_seed_value = inject_novelty_function(game_elements, game_seed)
    #print(Player.process_move_consequences.__module__)
    #print(action_choices.buy_property.__module__)
    if not novelty_info:
        ###
        ###
        if player_decision_agents['player_1'].startup(game_elements) == flag_config_dict['failure_code'] or \
                player_decision_agents['player_2'].startup(game_elements) == flag_config_dict['failure_code'] or \
                player_decision_agents['player_3'].startup(game_elements) == flag_config_dict['failure_code'] or \
                player_decision_agents['player_4'].startup(game_elements) == flag_config_dict['failure_code']:
            logger.error("Error in initializing agents. Cannot play the game.")
            return None
        else:
            logger.debug("Sucessfully initialized all player agents.")
            winner = simulate_game_instance(game_elements, history_log_file=None, np_seed=game_seed)
            if player_decision_agents['player_1'].shutdown() == flag_config_dict['failure_code'] or \
                    player_decision_agents['player_2'].shutdown() == flag_config_dict['failure_code'] or \
                    player_decision_agents['player_3'].shutdown() == flag_config_dict['failure_code'] or \
                    player_decision_agents['player_4'].shutdown() == flag_config_dict['failure_code']:
                logger.error("Error in agent shutdown.")
                return None
            else:
                logger.debug("All player agents have been shutdown. ")
                logger.debug("GAME OVER")
                cleanup_loggers()
                return winner, class_name, func_name, arg_value, meta_seed_value
    else:
        if inject_novelty_function:
            if player_decision_agents['player_1'].startup(game_elements, indicator=True) == flag_config_dict['failure_code'] or \
                    player_decision_agents['player_2'].startup(game_elements, indicator=True) == flag_config_dict['failure_code'] or \
                    player_decision_agents['player_3'].startup(game_elements, indicator=True) == flag_config_dict['failure_code'] or \
                    player_decision_agents['player_4'].startup(game_elements, indicator=True) == flag_config_dict['failure_code']:
                logger.error("Error in initializing agents. Cannot play the game.")
                return None
            else:
                logger.debug("Sucessfully initialized all player agents.")
                winner = simulate_game_instance(game_elements, history_log_file=None, np_seed=game_seed)
                if player_decision_agents['player_1'].shutdown() == flag_config_dict['failure_code'] or \
                        player_decision_agents['player_2'].shutdown() == flag_config_dict['failure_code'] or \
                        player_decision_agents['player_3'].shutdown() == flag_config_dict['failure_code'] or \
                        player_decision_agents['player_4'].shutdown() == flag_config_dict['failure_code']:
                    logger.error("Error in agent shutdown.")
                    return None
                else:
                    logger.debug("All player agents have been shutdown. ")
                    logger.debug("GAME OVER")
                    cleanup_loggers()
                    return winner, class_name, func_name, arg_value, meta_seed_value
        else:
            if player_decision_agents['player_1'].startup(game_elements, indicator=False) == flag_config_dict['failure_code'] or \
                    player_decision_agents['player_2'].startup(game_elements, indicator=False) == flag_config_dict['failure_code'] or \
                    player_decision_agents['player_3'].startup(game_elements, indicator=False) == flag_config_dict['failure_code'] or \
                    player_decision_agents['player_4'].startup(game_elements, indicator=False) == flag_config_dict['failure_code']:
                logger.error("Error in initializing agents. Cannot play the game.")
                return None
            else:
                logger.debug("Sucessfully initialized all player agents.")
                winner = simulate_game_instance(game_elements, history_log_file=None, np_seed=game_seed)
                if player_decision_agents['player_1'].shutdown() == flag_config_dict['failure_code'] or \
                        player_decision_agents['player_2'].shutdown() == flag_config_dict['failure_code'] or \
                        player_decision_agents['player_3'].shutdown() == flag_config_dict['failure_code'] or \
                        player_decision_agents['player_4'].shutdown() == flag_config_dict['failure_code']:
                    logger.error("Error in agent shutdown.")
                    return None
                else:
                    logger.debug("All player agents have been shutdown. ")
                    logger.debug("GAME OVER")
                    cleanup_loggers()
                    return winner, class_name, func_name, arg_value, meta_seed_value

def play_game_in_tournament_socket_phase3( game_seed, agent1, agent2, agent3, agent4, novelty_info = False, inject_novelty_function = False, novelty_name = None,state_logger=None):
    """
    Use this function if you want to test a single game instance and control lots of things. For experiments, we will directly
    call some of the functions in gameplay from test_harness.py.

    This is where everything begins. Assign decision agents to your players, set up the board and start simulating! You can
    control any number of players you like, and assign the rest to the simple agent. We plan to release a more sophisticated
    but still relatively simple agent soon.
    :return: String. the name of the player who won the game, if there was a winner, otherwise None.
    """
    print("using seed: ", game_seed)
    try:
        os.makedirs('../single_tournament/')
        print('Creating folder and logging gameplay.')
    except:
        print('Logging gameplay.')

    logger = log_file_create('../single_tournament/seed_6.log')
    player_decision_agents = dict()

    if agent1 is not None:
        player_decision_agents['player_1'] = agent1
    else:
        player_decision_agents['player_1'] = Agent(**background_agent_v3_1.decision_agent_methods)
    player_decision_agents['player_2'] = Agent(**agent2.decision_agent_methods)
    player_decision_agents['player_3'] = Agent(**agent3.decision_agent_methods)
    player_decision_agents['player_4'] = Agent(**agent4.decision_agent_methods)

    #print(Player.process_move_consequences.__module__)
    #print(action_choices.buy_property.__module__)
    game_elements = set_up_board('monopoly_game_schema_v1-2.json',
                                 player_decision_agents)

    #Comment out the above line and uncomment the piece of code to read the gameboard state from an existing json file so that
    #the game starts from a particular game state instead of initializing the gameboard with default start values.
    #Note that the novelties introduced in that particular game which was saved to file will be loaded into this game board as well.
    '''
    logger.debug("Loading gameboard from an existing game state that was saved to file.")
    infile = '../current_gameboard_state.json'
    game_elements = read_write_current_state.read_in_current_state_from_file(infile, player_decision_agents)
    '''
    class_name, func_name, arg_value, meta_seed_value = False, False, False, False
    if inject_novelty_function==True:
        pass
    elif inject_novelty_function:
        # -------
        game_elements['novelty_xxxx'] = True
        # -------
        inject_novelty_function(game_elements, novelty_name)
    #print(Player.process_move_consequences.__module__)
    #print(action_choices.buy_property.__module__)
    if not novelty_info:
        ###
        ###
        if player_decision_agents['player_1'].startup(game_elements) == flag_config_dict['failure_code'] or \
                player_decision_agents['player_2'].startup(game_elements) == flag_config_dict['failure_code'] or \
                player_decision_agents['player_3'].startup(game_elements) == flag_config_dict['failure_code'] or \
                player_decision_agents['player_4'].startup(game_elements) == flag_config_dict['failure_code']:
            logger.error("Error in initializing agents. Cannot play the game.")
            return None
        else:
            logger.debug("Sucessfully initialized all player agents.")
            winner = simulate_game_instance(game_elements, history_log_file=None, np_seed=game_seed,state_logger=state_logger)
            if player_decision_agents['player_1'].shutdown() == flag_config_dict['failure_code'] or \
                    player_decision_agents['player_2'].shutdown() == flag_config_dict['failure_code'] or \
                    player_decision_agents['player_3'].shutdown() == flag_config_dict['failure_code'] or \
                    player_decision_agents['player_4'].shutdown() == flag_config_dict['failure_code']:
                logger.error("Error in agent shutdown.")
                return None
            else:
                logger.debug("All player agents have been shutdown. ")
                logger.debug("GAME OVER")
                cleanup_loggers()
                return winner
    else:
        if inject_novelty_function:
            if player_decision_agents['player_1'].startup(game_elements, indicator=True) == flag_config_dict['failure_code'] or \
                    player_decision_agents['player_2'].startup(game_elements, indicator=True) == flag_config_dict['failure_code'] or \
                    player_decision_agents['player_3'].startup(game_elements, indicator=True) == flag_config_dict['failure_code'] or \
                    player_decision_agents['player_4'].startup(game_elements, indicator=True) == flag_config_dict['failure_code']:
                logger.error("Error in initializing agents. Cannot play the game.")
                return None
            else:
                logger.debug("Sucessfully initialized all player agents.")
                winner = simulate_game_instance(game_elements, history_log_file=None, np_seed=game_seed,state_logger=state_logger)
                if player_decision_agents['player_1'].shutdown() == flag_config_dict['failure_code'] or \
                        player_decision_agents['player_2'].shutdown() == flag_config_dict['failure_code'] or \
                        player_decision_agents['player_3'].shutdown() == flag_config_dict['failure_code'] or \
                        player_decision_agents['player_4'].shutdown() == flag_config_dict['failure_code']:
                    logger.error("Error in agent shutdown.")
                    return None
                else:
                    logger.debug("All player agents have been shutdown. ")
                    logger.debug("GAME OVER")
                    cleanup_loggers()
                    return winner
        else:
            if player_decision_agents['player_1'].startup(game_elements, indicator=False) == flag_config_dict['failure_code'] or \
                    player_decision_agents['player_2'].startup(game_elements, indicator=False) == flag_config_dict['failure_code'] or \
                    player_decision_agents['player_3'].startup(game_elements, indicator=False) == flag_config_dict['failure_code'] or \
                    player_decision_agents['player_4'].startup(game_elements, indicator=False) == flag_config_dict['failure_code']:
                logger.error("Error in initializing agents. Cannot play the game.")
                return None
            else:
                logger.debug("Sucessfully initialized all player agents.")
                winner = simulate_game_instance(game_elements, history_log_file=None, np_seed=game_seed,state_logger=state_logger)
                if player_decision_agents['player_1'].shutdown() == flag_config_dict['failure_code'] or \
                        player_decision_agents['player_2'].shutdown() == flag_config_dict['failure_code'] or \
                        player_decision_agents['player_3'].shutdown() == flag_config_dict['failure_code'] or \
                        player_decision_agents['player_4'].shutdown() == flag_config_dict['failure_code']:
                    logger.error("Error in agent shutdown.")
                    return None
                else:
                    logger.debug("All player agents have been shutdown. ")
                    logger.debug("GAME OVER")
                    cleanup_loggers()
                    return winner
  
        
        
def play_game_in_tournament_1(game_seed, agent0, agent1, agent2, agent3, novelty_info=False, inject_novelty_function=None, history_log_file = None):

    logger.debug('seed used: ' + str(game_seed))

    agent = ServerAgent()
    f_name = 'play game without novelty'
    if not agent.start_tournament(f_name):
        print("Unable to start tournament")
        exit(0)
    else:
        pass

    # player_decision_agents['player_1'] = Agent(**background_agent_v3_1.decision_agent_methods)


    player_decision_agents = dict()
    # for p in ['player_1','player_3']:
    #     player_decision_agents[p] = simple_decision_agent_1.decision_agent_methods
    player_decision_agents['player_1'] = agent
    #player_decision_agents['player_1'] = Agent(**agent0.decision_agent_methods)

    player_decision_agents['player_2'] = Agent(**agent1.decision_agent_methods)


    player_decision_agents['player_3'] = Agent(**agent2.decision_agent_methods)
    player_decision_agents['player_4'] = Agent(**agent3.decision_agent_methods)


    game_elements = set_up_board('./monopoly_game_schema_v1-2.json',
                                 player_decision_agents)

    #Comment out the above line and uncomment the piece of code to read the gameboard state from an existing json file so that
    #the game starts from a particular game state instead of initializing the gameboard with default start values.
    #Note that the novelties introduced in that particular game which was saved to file will be loaded into this game board as well.
    '''
    logger.debug("Loading gameboard from an existing game state that was saved to file.")
    infile = '../current_gameboard_state.json'
    game_elements = read_write_current_state.read_in_current_state_from_file(infile, player_decision_agents)
    '''

    if inject_novelty_function:
        inject_novelty_function(game_elements)
        #print(sys.modules[__name__])
        #game_elements = novelty_distributions_v3.reinitialize_game_elements(game_elements)
        #game_elements = set_up_board('../monopoly_game_schema_v1-2.json', player_decision_agents)
        #print(game_elements['location_objects']['Chance'].perform_action)
        #import inspect # for debug
        #print(inspect.getsource(action_choices.improve_property))  # for debug
        #print(inspect.getsource(card_utility_actions.pick_card_from_chance)) # for debug
        #print(inspect.getsource(game_elements['location_objects']['Chance'].perform_action))
        #print(inspect.getsource(game_elements['location_objects']['Chance'].perform_action))
    if not novelty_info:
        if player_decision_agents['player_1'].startup(game_elements) == flag_config_dict['failure_code'] or \
                player_decision_agents['player_2'].startup(game_elements) == flag_config_dict['failure_code'] or \
                player_decision_agents['player_3'].startup(game_elements) == flag_config_dict['failure_code'] or \
                player_decision_agents['player_4'].startup(game_elements) == flag_config_dict['failure_code']:
            logger.error("Error in initializing agents. Cannot play the game.")
            #agent.end_tournament()
            return None
        else:
            logger.debug("Sucessfully initialized all player agents.")
            winner = simulate_game_instance(game_elements, history_log_file=history_log_file, np_seed=game_seed)
            if player_decision_agents['player_1'].shutdown() == flag_config_dict['failure_code'] or \
                    player_decision_agents['player_2'].shutdown() == flag_config_dict['failure_code'] or \
                    player_decision_agents['player_3'].shutdown() == flag_config_dict['failure_code'] or \
                    player_decision_agents['player_4'].shutdown() == flag_config_dict['failure_code']:
                logger.error("Error in agent shutdown.")
                #agent.end_tournament()
                return None
            else:
                logger.debug("All player agents have been shutdown. ")
                logger.debug("GAME OVER")
                cleanup_loggers()
                #agent.end_tournament()
                return winner
    else:
        if inject_novelty_function:
            if player_decision_agents['player_1'].startup(game_elements, indicator=True) == flag_config_dict['failure_code'] or \
                    player_decision_agents['player_2'].startup(game_elements, indicator=True) == flag_config_dict['failure_code'] or \
                    player_decision_agents['player_3'].startup(game_elements, indicator=True) == flag_config_dict['failure_code'] or \
                    player_decision_agents['player_4'].startup(game_elements, indicator=True) == flag_config_dict['failure_code']:
                logger.error("Error in initializing agents. Cannot play the game.")
                #agent.end_tournament()
                return None
            else:
                logger.debug("Sucessfully initialized all player agents.")
                winner = simulate_game_instance(game_elements, history_log_file=None, np_seed=game_seed)
                if player_decision_agents['player_1'].shutdown() == flag_config_dict['failure_code'] or \
                        player_decision_agents['player_2'].shutdown() == flag_config_dict['failure_code'] or \
                        player_decision_agents['player_3'].shutdown() == flag_config_dict['failure_code'] or \
                        player_decision_agents['player_4'].shutdown() == flag_config_dict['failure_code']:
                    logger.error("Error in agent shutdown.")
                    #agent.end_tournament()
                    return None
                else:
                    logger.debug("All player agents have been shutdown. ")
                    logger.debug("GAME OVER")
                    #agent.end_tournament()
                    cleanup_loggers()
                    return winner
        else:
            if player_decision_agents['player_1'].startup(game_elements, indicator=False) == flag_config_dict['failure_code'] or \
                    player_decision_agents['player_2'].startup(game_elements, indicator=False) == flag_config_dict['failure_code'] or \
                    player_decision_agents['player_3'].startup(game_elements, indicator=False) == flag_config_dict['failure_code'] or \
                    player_decision_agents['player_4'].startup(game_elements, indicator=False) == flag_config_dict['failure_code']:
                logger.error("Error in initializing agents. Cannot play the game.")
                #agent.end_tournament()
                return None
            else:
                logger.debug("Sucessfully initialized all player agents.")
                winner = simulate_game_instance(game_elements, history_log_file=None, np_seed=game_seed)
                if player_decision_agents['player_1'].shutdown() == flag_config_dict['failure_code'] or \
                        player_decision_agents['player_2'].shutdown() == flag_config_dict['failure_code'] or \
                        player_decision_agents['player_3'].shutdown() == flag_config_dict['failure_code'] or \
                        player_decision_agents['player_4'].shutdown() == flag_config_dict['failure_code']:
                    logger.error("Error in agent shutdown.")
                    #agent.end_tournament()
                    return None
                else:
                    logger.debug("All player agents have been shutdown. ")
                    logger.debug("GAME OVER")
                    #agent.end_tournament()
                    cleanup_loggers()
                    return winner


# play_game(inject_novelty_function = getattr(sys.modules[__name__], nov))
#play_game()
