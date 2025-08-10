from monopoly_simulator.flag_config import flag_config_dict
import logging
logger = logging.getLogger('monopoly_simulator.logging_info.agent')


class Agent(object):
    def __init__(self, **kwargs):
        """
        While you could always instantiate this class directly, sub-classing may be a better option since it will allow
        you to maintain your own state/variables.
        All of these are decision functions that need to be implemented in your agent. See simple decision agent v1 and
        background agent v1 for example implementations of these functions.
        :param handle_negative_cash_balance:
        :param make_pre_roll_move:
        :param make_out_of_turn_move:
        :param make_post_roll_move:
        :param make_buy_property_decision:
        :param make_bid:
        :param type: type of dictionary
        """

        self.handle_negative_cash_balance = kwargs.get('handle_negative_cash_balance')
        self.make_pre_roll_move = kwargs.get('make_pre_roll_move')
        self.make_out_of_turn_move = kwargs.get('make_out_of_turn_move')
        self.make_post_roll_move = kwargs.get('make_post_roll_move')
        self.make_buy_property_decision = kwargs.get('make_buy_property_decision')
        self.make_bid = kwargs.get('make_bid')
        self.type = kwargs.get('type')

        self._calculate_reward = kwargs.get('_calculate_reward', self._default_calculate_reward)
        self._is_episode_done = kwargs.get('_is_episode_done', self._default_is_episode_done)
        self.get_training_mode = kwargs.get('get_training_mode', self._default_get_training_mode)
        self.get_last_action_idx = kwargs.get('get_last_action_idx', self._default_get_last_action_idx)

        self.is_running = False   #a flag which says if the agent is active or shutdown
        self._agent_memory = dict()  # a scratchpad for the agent

    def _default_calculate_reward(self, player, current_gameboard):
        return 0

    def _default_is_episode_done(self, current_gameboard):
        return False

    def _default_get_training_mode(self):
        return False
        
    def _default_get_last_action_idx(self):
        return None


    def startup(self, current_gameboard, indicator=None):
        """
        This function is called before simulating the game instance to startup the player agents by setting their is_running
        flag to true. This is done only after making sure that all the agent functions have been initialized.
        :param current_gameboard: the initial state of current_gameboard right after setting up the board.
        :param indicator: a string that can be used to indicate the type of game startup (like normal startup, restart, etc)
        :return: returns successful action code if all function handlers are intialized and after agent is started up. Any error in doing so results
        in a return value of failure code.
        """
        if self.handle_negative_cash_balance == None:
            logger.error("Agent not initialized properly. Returning failure code.")
            return flag_config_dict['failure_code']
        if self.make_pre_roll_move == None:
            logger.error("Agent not initialized properly. Returning failure code.")
            return flag_config_dict['failure_code']
        if self.make_out_of_turn_move == None:
            logger.error("Agent not initialized properly. Returning failure code.")
            return flag_config_dict['failure_code']
        if self.make_post_roll_move == None:
            logger.error("Agent not initialized properly. Returning failure code.")
            return flag_config_dict['failure_code']
        if self.make_buy_property_decision == None:
            logger.error("Agent not initialized properly. Returning failure code.")
            return flag_config_dict['failure_code']
        if self.make_bid == None:
            logger.error("Agent not initialized properly. Returning failure code.")
            return flag_config_dict['failure_code']

        self.is_running = True
        return flag_config_dict['successful_action']


    def shutdown(self):
        """
        This function is called to shutdown a running agent after the game terminates.
        :return: function returns successful action code is the agent is successfully shut down else return failure code. (if trying to shutdown an
        already shutdown agent.)
        """
        if self.is_running == False:
            logger.error("Trying to shutdown an already shutdown agent. Returning failure code.")
            return flag_config_dict['failure_code']
        else:
            self.is_running = False
            return flag_config_dict['successful_action']
