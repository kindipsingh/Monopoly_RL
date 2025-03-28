
"""
Monopoly Action Encoder
------------------------
This module implements the updated ActionEncoder for Phase 2: Action Space Implementation,
including comprehensive action definitions, trading rules enforcement, and mapping the encoded
actions to their required execution parameter lists.

Key Considerations:
    - Rolling doubles are treated as regular dice rolls.
    - Trading enforces that:
          * Only unimproved (no houses/hotels) and unmortgaged properties can be traded.
          * Trade offers can be made to multiple players simultaneously.
          * A receiver may accept or reject an offer.
          * Once a trade is accepted, all other simultaneous offers for the same property are terminated.
          * A player can only have one outstanding trade offer at a time.
    - The gameplay is divided into three phases:
          * Pre-roll: The current player takes actions.
          * Out-of-turn: Other players act in round-robin until completion.
          * Post-roll: The current player acts based on the dice outcome.
    - If a player ends post-roll with negative cash, they may attempt remediation; failing that initiates bankruptcy.

Action Definitions and Dimensionalities:
    1. make_trade_offer_exchange : 2240
    2. make_trade_offer_sell     : 252
    3. make_trade_offer_buy      : 252
    4. improve_property          : 44
    5. sell_house_hotel          : 44
    6. sell_property             : 28
    7. mortgage_property         : 28
    8. free_mortgage             : 28
    9. skip_turn                 : 1
    10. conclude_actions         : 1
    11. use_get_out_of_jail_card : 1
    12. pay_jail_fine            : 1
    13. accept_trade_offer       : 1
    14. buy_property             : 1

Total action space dimensionality is 2922.

Integration:
    The encoder maps decision outputs from the agent directly to the parameters needed by the
    in-game execution command (_execute_action). Action masking is used to enforce valid moves
    based on both the current game phase and game-specific rules (including trade constraints).

Additional Functionality:
    - This updated version imports the individual mapping builder functions.
    - It introduces methods to build the full action mapping list and decode a selected action index,
      returning the associated action parameters required for execution.

Assumptions:
    - Helper methods exist on the player/agent for computing allowable actions.
    - Mapping builder functions output lists of dictionaries with the structure:
         { "action": <action_name>, "parameters": { ... } }
    - The full action mapping is constructed in the same order as defined in action_definitions.
"""

import numpy as np
import logging
from typing import Dict, Tuple

# Import mapping builder functions from individual mapping files
from monopoly_simulator.make_trade_offer_exchange_mapping import build_make_trade_offer_exchange_list
from monopoly_simulator.make_trade_offer_sell_mapping import build_make_trade_offer_sell_list
from monopoly_simulator.make_trade_offer_buy_mapping import build_make_trade_offer_buy_list
from monopoly_simulator.improve_property_mapping import build_improve_property_list
from monopoly_simulator.sell_house_hotel_mapping import build_sell_house_hotel_list
from monopoly_simulator.sell_property_mapping import build_sell_property_list
from monopoly_simulator.mortgage_property_mapping import build_mortgage_property_list
from monopoly_simulator.free_mortgage_mapping import build_free_mortgage_list
from monopoly_simulator.other_actions_mapping import build_other_actions_mapping
# Assuming build_buy_property_list is provided via full_action_mapping module or similar
from monopoly_simulator.full_action_mapping import build_buy_property_list

# Configure logging
logger = logging.getLogger('monopoly_simulator.action_vectors')
logger.propagate = False

class ActionEncoder:
    def __init__(self):
        """Initialize the ActionEncoder with action definitions."""
        self.action_definitions = {
            "make_trade_offer_exchange": {"dim": 2240, "phases": {"pre_roll", "out_of_turn"}},
            "make_trade_offer_sell": {"dim": 252, "phases": {"pre_roll", "out_of_turn"}},
            "make_trade_offer_buy": {"dim": 252, "phases": {"pre_roll", "out_of_turn"}},
            "improve_property": {"dim": 44, "phases": {"pre_roll", "out_of_turn"}},
            "sell_house_hotel": {"dim": 44, "phases": {"pre_roll", "post_roll", "out_of_turn"}},
            "sell_property": {"dim": 28, "phases": {"pre_roll", "post_roll", "out_of_turn"}},
            "mortgage_property": {"dim": 28, "phases": {"pre_roll", "post_roll", "out_of_turn"}},
            "free_mortgage": {"dim": 28, "phases": {"pre_roll", "post_roll", "out_of_turn"}},
            "skip_turn": {"dim": 1, "phases": {"pre_roll", "post_roll", "out_of_turn"}},
            "conclude_actions": {"dim": 1, "phases": {"pre_roll", "post_roll", "out_of_turn"}},
            "use_get_out_of_jail_card": {"dim": 1, "phases": {"pre_roll"}},
            "pay_jail_fine": {"dim": 1, "phases": {"pre_roll"}},
            "accept_trade_offer": {"dim": 1, "phases": {"pre_roll", "out_of_turn"}},
            "buy_property": {"dim": 1, "phases": {"post_roll"}}
        }

    def encode(self, player, current_gameboard, game_phase: str) -> Tuple[np.ndarray, np.ndarray]:
        """Encode the current game state into an action vector and mask."""
        # Initialize vectors
        encoded_action_vector = np.zeros(2922)  # Total dimension
        action_mask = np.zeros(2922, dtype=bool)
        
        # Get computed allowable actions based on game phase
        if game_phase == "pre_roll":
            computed_allowable = player.compute_allowable_pre_roll_actions(current_gameboard)
            logger.debug(f"Pre-roll computed allowable actions for {player.player_name}: {computed_allowable}")
        elif game_phase == "post_roll":
            computed_allowable = player.compute_allowable_post_roll_actions(current_gameboard)
            logger.debug(f"Post-roll computed allowable actions for {player.player_name}: {computed_allowable}")
        else:  # out_of_turn
            computed_allowable = player.compute_allowable_out_of_turn_actions(current_gameboard)
            logger.debug(f"Out-of-turn computed allowable actions for {player.player_name}: {computed_allowable}")
        
        # Convert computed_allowable to list if it's a set
        if isinstance(computed_allowable, set):
            computed_allowable = sorted(list(computed_allowable))
            logger.debug(f"Converted allowable actions to list: {computed_allowable}")

        current_index = 0
        for action_name, definition in self.action_definitions.items():
            dim = definition["dim"]
            allowed_phases = definition["phases"]
            
            phase_valid = game_phase in allowed_phases
            computed_valid = action_name in computed_allowable
            
            if action_name.startswith("make_trade_offer"):
                computed_valid = computed_valid and self._is_trade_allowed(player, current_gameboard)
            
            is_valid = phase_valid and computed_valid
            
            if is_valid:
                logger.debug(f"Valid action for {player.player_name}: {action_name}")
            
            vector_segment = np.zeros(dim)
            mask_segment = np.zeros(dim, dtype=bool)
            
            if is_valid:
                vector_segment[0] = 1
                mask_segment[0] = True
                
                # Handle multi-dimensional actions if necessary
                if dim > 1:
                    self._handle_multi_dimensional_action(
                        vector_segment, 
                        mask_segment, 
                        action_name, 
                        player, 
                        current_gameboard
                    )
            
            encoded_action_vector[current_index:current_index + dim] = vector_segment
            action_mask[current_index:current_index + dim] = mask_segment
            current_index += dim
        
        # Ensure we have non-null vectors for debugging
        if np.sum(encoded_action_vector) == 0:
            logger.warning(f"Encoded action vector is all zeros for {player.player_name} in {game_phase} phase")
        
        if np.sum(action_mask) == 0:
            logger.warning(f"Action mask is all zeros for {player.player_name} in {game_phase} phase")
        
        logger.debug(f"Encoded action vector sum: {np.sum(encoded_action_vector)}")
        logger.debug(f"Action mask sum: {np.sum(action_mask)}")
        logger.debug(f"Number of allowed actions: {len(computed_allowable)}")
        
        return encoded_action_vector, action_mask

    def _handle_multi_dimensional_action(self, vector_segment: np.ndarray, 
                                           mask_segment: np.ndarray, 
                                           action_name: str, 
                                           player, 
                                           current_gameboard: Dict):
        """Handle multi-dimensional actions by setting appropriate vector and mask values."""
        try:
            if action_name.startswith("make_trade_offer"):
                self._handle_trade_offer(vector_segment, mask_segment, player, current_gameboard)
            elif action_name in ["improve_property", "sell_house_hotel"]:
                self._handle_property_improvement(vector_segment, mask_segment, player, action_name)
            elif action_name in ["mortgage_property", "free_mortgage", "sell_property"]:
                self._handle_property_action(vector_segment, mask_segment, player, action_name)
        except Exception as e:
            logger.error(f"Error handling multi-dimensional action {action_name}: {str(e)}")

    def _handle_trade_offer(self, vector_segment: np.ndarray, 
                              mask_segment: np.ndarray, 
                              player, 
                              current_gameboard: Dict):
        """Handle trade offer actions."""
        # Get valid properties for trading
        tradeable_properties = []
        if hasattr(player, 'properties'):
            tradeable_properties = [
                i for i, prop in enumerate(player.properties)
                if not getattr(prop, 'is_mortgaged', False) and 
                   not getattr(prop, 'is_improved', False)
            ]
        
        # Set mask for tradeable properties; index 0 is reserved for the action type
        for prop_idx in tradeable_properties:
            mask_segment[prop_idx + 1] = True

    def _handle_property_improvement(self, vector_segment: np.ndarray, 
                                     mask_segment: np.ndarray, 
                                     player, 
                                     action_name: str):
        """Handle property improvement actions."""
        if hasattr(player, 'properties'):
            for i, prop in enumerate(player.properties):
                can_improve = (action_name == "improve_property" and 
                               getattr(prop, 'can_be_improved', False))
                can_sell = (action_name == "sell_house_hotel" and 
                            getattr(prop, 'has_improvements', False))
                
                if can_improve or can_sell:
                    mask_segment[i + 1] = True  # +1 because index 0 is reserved for action type

    def _handle_property_action(self, vector_segment: np.ndarray, 
                                mask_segment: np.ndarray, 
                                player, 
                                action_name: str):
        """Handle property-related actions (mortgage, unmortgage, sell)."""
        if hasattr(player, 'properties'):
            for i, prop in enumerate(player.properties):
                if action_name == "mortgage_property":
                    if not getattr(prop, 'is_mortgaged', True) and not getattr(prop, 'is_improved', False):
                        mask_segment[i + 1] = True
                elif action_name == "free_mortgage":
                    if getattr(prop, 'is_mortgaged', False):
                        mask_segment[i + 1] = True
                elif action_name == "sell_property":
                    if not getattr(prop, 'is_mortgaged', True) and not getattr(prop, 'is_improved', False):
                        mask_segment[i + 1] = True

    def _is_trade_allowed(self, player, current_gameboard: Dict) -> bool:
        """Check if trading is allowed for the player."""
        if hasattr(player, 'outstanding_trade_offer') and player.outstanding_trade_offer:
            return False
            
        if hasattr(player, 'properties'):
            tradeable_properties = [
                prop for prop in player.properties 
                if not getattr(prop, 'is_mortgaged', True) and 
                   not getattr(prop, 'is_improved', False)
            ]
            return len(tradeable_properties) > 0
        return False

    def build_full_action_mapping(self, acting_player, current_gameboard, schema_filepath="monopoly_game_schema_v1-2.json") -> list:
        """
        Construct the full flat mapping list for all actions by concatenating the individual mappings.
        The sequence follows the order defined in action_definitions.
        Returns:
            A list of mapping dictionaries. Each dictionary contains:
              - "action": the action name.
              - "parameters": a dictionary of parameters required for execution.
        """
        full_mapping = []
        full_mapping.extend(build_make_trade_offer_exchange_list(acting_player,current_gameboard, schema_filepath))
        full_mapping.extend(build_make_trade_offer_sell_list(acting_player, schema_filepath))
        full_mapping.extend(build_make_trade_offer_buy_list(acting_player,current_gameboard,schema_filepath))
        full_mapping.extend(build_improve_property_list(schema_filepath))
        full_mapping.extend(build_sell_house_hotel_list(acting_player,current_gameboard,schema_filepath))
        full_mapping.extend(build_sell_property_list(acting_player,current_gameboard,schema_filepath))
        full_mapping.extend(build_mortgage_property_list(acting_player,current_gameboard,schema_filepath))
        full_mapping.extend(build_free_mortgage_list(acting_player,current_gameboard,schema_filepath))
        full_mapping.extend(build_other_actions_mapping(acting_player, current_gameboard))
        full_mapping.extend(build_buy_property_list())
        return full_mapping

    def decode_action(self, acting_player, current_gameboard, chosen_index: int, schema_filepath="monopoly_game_schema_v1-2.json") -> Dict:
        """
        Given a chosen action index (from 0 to 2921), decode it into the corresponding mapping entry
        which contains the action and the parameters required to execute it.
        
        Args:
            acting_player: The player whose action is being decoded.
            current_gameboard: The current game board dict.
            chosen_index: Flattened index of the chosen action in the full action space.
            schema_filepath: Schema file path for mapping builders.
            
        Returns:
            A dictionary with keys "action" and "parameters" corresponding to the selected action.
        """
        full_mapping = self.build_full_action_mapping(acting_player, current_gameboard, schema_filepath)
        if chosen_index < 0 or chosen_index >= len(full_mapping):
            logger.error(f"Invalid action index: {chosen_index}. It must be between 0 and {len(full_mapping)-1}")
            raise ValueError("Chosen action index is out of range.")
        selected_mapping = full_mapping[chosen_index]
        logger.debug(f"Decoded action at index {chosen_index}: {selected_mapping}")
        return selected_mapping

    def log_action_encoding(self, encoded_action_vector: np.ndarray, 
                            action_mask: np.ndarray, 
                            game_elements: Dict):
        """Log the action encoding and mask to game history."""
        try:
            if 'history' not in game_elements:
                game_elements['history'] = dict()
            if 'action_encoding' not in game_elements['history']:
                game_elements['history']['action_encoding'] = list()
            
            game_elements['history']['action_encoding'].append({
                'encoded_action_vector': encoded_action_vector.tolist(),
                'action_mask': action_mask.tolist(),
                'time_step': game_elements.get('time_step_indicator', 0)
            })
            
        except Exception as e:
            logger.error(f"Error logging action encoding: {str(e)}")

    def make_decision(self, player, current_gameboard, game_phase):
        """
        High-level method integrating encoded action vectors with specific decision outputs.
        Handles:
            - Property actions (e.g., buy decisions post-roll).
            - Trade proposals (subject to trading rules and phase).
            - Additional financial or developmental decisions such as mortgage or improvement.
        
        Returns:
            A dictionary containing:
            - 'action_vector': The encoded action vector.
            - 'action_mask': Boolean mask indicating valid actions.
            - Additional decision outputs (e.g., 'buy_property', 'trade_proposal').
        """
        encoded_vector, mask = self.encode(player, current_gameboard, game_phase)
        decision = {}

        # Property decision: Only applicable during the post_roll phase.
        if game_phase == "post_roll" and "buy_property" in player.compute_allowable_post_roll_actions(current_gameboard):
            decision['buy_property'] = player.agent.make_buy_property_decision(player, current_gameboard)

        # Trade proposals: Enforced if trading is allowed and applicable.
        allowable_pre = player.compute_allowable_pre_roll_actions(current_gameboard)
        allowable_out = player.compute_allowable_out_of_turn_actions(current_gameboard)
        if (( "make_trade_offer_exchange" in allowable_pre or 
              "make_trade_offer_sell" in allowable_pre or 
              "make_trade_offer_buy" in allowable_pre) or
             ("make_trade_offer_exchange" in allowable_out)) and self._is_trade_allowed(player, current_gameboard):
            decision['trade_proposal'] = player.agent.trade_proposal(player, current_gameboard)

        # Additional decisions (e.g., mortgage, improvement, bankruptcy) can be integrated similarly.
        decision['action_vector'] = encoded_vector
        decision['action_mask'] = mask

        return decision
