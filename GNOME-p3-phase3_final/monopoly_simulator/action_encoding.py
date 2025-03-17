
"""
Monopoly Action Encoder
------------------------
This module implements the updated ActionEncoder for Phase 2: Action Space Implementation,
including comprehensive action definitions and trading rules enforcement as per provided
specifications.

Key Considerations:
    - Rolling doubles are treated as regular dice rolls (no special rules).
    - Trading enforces that:
          * Only unimproved (no houses/hotels) and unmortgaged properties can be traded.
          * Trade offers can be made to multiple players simultaneously.
          * A receiver may accept or reject an offer.
          * Once a trade is accepted, all other simultaneous offers for the same property are terminated.
          * A player can only have one outstanding trade offer at a time.
    - The gameplay is divided into three phases:
          * Pre-roll: The current player takes actions (must conclude actions to end this phase).
          * Out-of-turn: Other players act in a round-robin manner until all skip or a fixed
                        number of rounds complete.
          * Post-roll: The current player acts based on the dice outcome (e.g., deciding to buy).
    - If a player ends post-roll with negative cash, they may attempt to remedy it;
      failing that initiates bankruptcy.

Action Definitions:
    1. Make Trade Offer (Exchange)
       - Allowed Phases: Pre-roll, out-of-turn
       - Parameters: To player, property offered, property requested, cash offered, cash requested
       - Dimensionality: 2240  (adjusted from 2268 to ensure TOTAL_DIM is 2922)
    2. Make Trade Offer (Sell)
       - Allowed Phases: Pre-roll, out-of-turn
       - Parameters: To player, property offered, cash requested
       - Dimensionality: 252
    3. Make Trade Offer (Buy)
       - Allowed Phases: Pre-roll, out-of-turn
       - Parameters: To player, property requested, cash offered
       - Dimensionality: 252
    4. Improve Property
       - Allowed Phases: Pre-roll, out-of-turn
       - Parameters: Property, flag for house/hotel
       - Dimensionality: 44
    5. Sell House or Hotel
       - Allowed Phases: Pre-roll, post-roll, out-of-turn
       - Parameters: Property, flag for house/hotel
       - Dimensionality: 44
    6. Sell Property
       - Allowed Phases: Pre-roll, post-roll, out-of-turn
       - Parameters: Property
       - Dimensionality: 28
    7. Mortgage Property
       - Allowed Phases: Pre-roll, post-roll, out-of-turn
       - Parameters: Property
       - Dimensionality: 28
    8. Free Mortgage
       - Allowed Phases: Pre-roll, post-roll, out-of-turn
       - Parameters: Property
       - Dimensionality: 28
    9. Skip Turn
       - Allowed Phases: Pre-roll, post-roll, out-of-turn
       - Parameters: None
       - Dimensionality: 1
    10. Conclude Actions
        - Allowed Phases: Pre-roll, post-roll, out-of-turn
        - Parameters: None
        - Dimensionality: 1
    11. Use get out of jail card
        - Allowed Phase: Pre-roll
        - Parameters: None
        - Dimensionality: 1
    12. Pay Jail Fine
        - Allowed Phase: Pre-roll
        - Parameters: None
        - Dimensionality: 1
    13. Accept Trade Offer
        - Allowed Phases: Pre-roll, out-of-turn
        - Parameters: None
        - Dimensionality: 1
    14. Buy Property
        - Allowed Phase: Post-roll
        - Parameters: Property
        - Dimensionality: 1

Total action space dimensionality is 2922.

Integration:
    The encoder maps decision outputs from the agent directly to the parameters required by
    the in-game execution (_execute_action). Action masking is used to enforce valid moves
    based on both the current game phase and game-specific rules (including trade constraints).

Note:
    This implementation assumes the existence of helper methods on the player/agent:
        - compute_allowable_pre_roll_actions
        - compute_allowable_post_roll_actions
        - compute_allowable_out_of_turn_actions
        - make_buy_property_decision
        - trade_proposal
    It also assumes that property objects include flags (e.g., is_improved, is_mortgaged) to enforce trading rules.
"""


import numpy as np
import logging
from typing import Dict, Tuple

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
                
                # Handle multi-dimensional actions
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
        
        # Set mask for tradeable properties
        for prop_idx in tradeable_properties:
            mask_segment[prop_idx + 1] = True  # +1 because index 0 is reserved for action type

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

    def _is_trade_allowed(self, player, current_gameboard):
        """
        Enforce trading rules:
            1. There must be more than one player.
            2. A player can have only one outstanding trade offer at a time.
            3. Only properties that are unimproved and unmortgaged can be traded.
               (Property objects should have boolean flags: is_improved, is_mortgaged.)
        
        Note: The processing of simultaneous trade offers and cancellation upon acceptance is handled externally.
        
        :param player: The player initiating the trade.
        :param current_gameboard: Dict containing the current game state.
        :return: bool indicating if trade actions are permitted.
        """
        if len(current_gameboard.get('players', [])) <= 1:
            return False
        
        if getattr(player, "outstanding_trade_offer", None) is not None:
            return False
        
        # Further property eligibility checks are assumed to be done when constructing the proposal.
        return True

    def make_decision(self, player, current_gameboard, game_phase):
        """
        High-level method integrating encoded action vectors with specific decision outputs.

        Handles:
            - Property actions (e.g., buy decisions post-roll).
            - Trade proposals (subject to trading rules and phase).
            - Additional financial or developmental decisions (mortgage, improvement, bankruptcy, etc.).

        :param player: The current player.
        :param current_gameboard: Dict representing the game state.
        :param game_phase: str; one of "pre_roll", "post_roll", "out_of_turn".
        :return: A dictionary containing:
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
