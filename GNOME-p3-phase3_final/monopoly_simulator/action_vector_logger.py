import logging
import numpy as np
from monopoly_simulator.action_encoding import ActionEncoder
import os
from pathlib import Path

class ActionVectorLogger:
    """
    Handles logging of action vectors and masks during Monopoly gameplay.
    Integrates with the existing game logging system and action encoder.
    """
    def __init__(self, logger_name='monopoly_simulator.action_vectors'):
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)
        self.action_encoder = ActionEncoder()
        self.file_handler = None  # Ensure file handler is defined
        
    def setup_file_logging(self, log_file_path):
        try:
            # Create directory if it doesn't exist
            log_dir = os.path.dirname(log_file_path)
            if log_dir:
                Path(log_dir).mkdir(parents=True, exist_ok=True)
                
            # Close existing handler if any
            if self.file_handler:
                self.file_handler.close()
                self.logger.removeHandler(self.file_handler)
            
            # Create new handler
            formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')
            self.file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
            self.file_handler.setFormatter(formatter)
            self.logger.addHandler(self.file_handler)
            
            self.logger.info(f"Action vector logging initialized. Writing to: {log_file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to setup file logging at {log_file_path}: {str(e)}")
            raise

    def log_action_vector(self, player, game_elements, phase):
        """
        Log the action vector and mask for a given player and game phase.
        
        Args:
            player: The player object making the move
            game_elements: Current game state dictionary
            phase: String indicating game phase ("pre_roll", "post_roll", or "out_of_turn")
        """
        encoded_vector, mask = self.action_encoder.encode(player, game_elements, phase)
        
        # Log encoded vector to game history
        if 'history' in game_elements and 'action_encoding' in game_elements['history']:
            game_elements['history']['action_encoding'].append({
                'player': player.player_name,
                'phase': phase,
                'vector': encoded_vector.tolist(),
                'mask': mask.tolist(),
                'time_step': game_elements.get('time_step_indicator', 0)
            })
        
        # Log detailed vector information
        self.logger.debug(f"\nAction Vector Log - {phase.upper()}")
        self.logger.debug(f"Player: {player.player_name}")
        self.logger.debug(f"Vector: {encoded_vector}")
        self.logger.debug(f"Mask: {mask}")
        # self.logger.debug(f"Full vector:\n{np.array2string(encoded_vector, threshold=np.inf)}")
        # self.logger.debug(f"Full mask:\n{np.array2string(mask, threshold=np.inf)}")
        
        return encoded_vector, mask

    def log_pre_roll(self, player, game_elements):
        """Log pre-roll phase actions"""
        return self.log_action_vector(player, game_elements, "pre_roll")
    
    def log_post_roll(self, player, game_elements):
        """Log post-roll phase actions"""
        return self.log_action_vector(player, game_elements, "post_roll")
    
    def log_out_of_turn(self, player, game_elements):
        """Log out-of-turn phase actions"""
        return self.log_action_vector(player, game_elements, "out_of_turn")

def integrate_with_gameplay(game_elements, history_log_file=None):

    vector_logger = ActionVectorLogger()
    
    # Define default log folder
    log_folder = r"C:\GNOME-p3-phase3_final\monopoly_simulator"

    if history_log_file:
        log_file = history_log_file.replace('.xlsx', '_action_vectors.log')
    else:
        log_file = os.path.join(log_folder, "action_vectors.log")

    # Ensure the directory exists
    Path(os.path.dirname(log_file)).mkdir(parents=True, exist_ok=True)
    
    vector_logger.setup_file_logging(log_file)
        
    return vector_logger
