
import sys
import os
import json

# Import mapping builder functions from individual mapping files.
from monopoly_simulator.make_trade_offer_exchange_mapping import build_make_trade_offer_exchange_list
from monopoly_simulator.make_trade_offer_sell_mapping import build_make_trade_offer_sell_list
from monopoly_simulator.make_trade_offer_buy_mapping import build_make_trade_offer_buy_list
from monopoly_simulator.improve_property_mapping import build_improve_property_list
from monopoly_simulator.sell_house_hotel_mapping import build_sell_house_hotel_list
from monopoly_simulator.sell_property_mapping import build_sell_property_list
from monopoly_simulator.mortgage_property_mapping import build_mortgage_property_list
from monopoly_simulator.free_mortgage_mapping import build_free_mortgage_list
from monopoly_simulator.other_actions_mapping import build_other_actions_mapping
from monopoly_simulator.player import Player

# Updated buy_property mapping: now includes a parameter 'asset' determined from the player's current_position.
def build_buy_property_list(player, current_gameboard, schema_filepath):
    """
    Build a flat list for the "buy_property" action. In addition to the standard parameters,
    the mapping now includes 'asset', which is determined by reading the schema file (assumed
    to be in JSON format) and indexing into the 'location_sequence' using player's current_position.
    
    Args:
      player: The acting player instance.
      current_gameboard: The current game board dictionary.
      schema_filepath: Path to the game schema file (e.g., "monopoly_game_schema2.json").
    
    Returns:
      A list containing one dictionary with the buy_property action and its parameters.
    """
    # Load the game schema.
    with open(schema_filepath, "r") as f:
        schema = json.load(f)
    
    # Retrieve the location sequence from the schema.
    location_sequence = schema.get("location_sequence", [])
    if not location_sequence:
        raise ValueError("Schema does not contain a valid 'location_sequence'.")
    
    # Determine the asset based on the player's current_position.
    if player.current_position is None or player.current_position >= len(location_sequence):
        raise ValueError("Player's current_position is invalid or out of bounds.")
    asset = location_sequence[player.current_position]
    if asset is None:
        raise ValueError("The location at player's current_position does not have a 'name' field.")
    
    return [{
        "action": "buy_property", 
        "parameters": {
            "player": player, 
            "asset": asset,
            "current_gameboard": current_gameboard
        }
    }]

def build_full_action_mapping(acting_player, current_gameboard, schema_filepath="monopoly_game_schema_v1-2.json"):
    """
    Build the full flat mapping list for all actions by concatenating the individual mappings.
    
    The order is:
      1. make_trade_offer_exchange  (dimension: ideally 2240 or your configured value)
      2. make_trade_offer_sell      (252 entries)
      3. make_trade_offer_buy       (252 entries)
      4. improve_property           (44 entries)
      5. sell_house_hotel           (44 entries)
      6. sell_property              (28 entries)
      7. mortgage_property          (28 entries)
      8. free_mortgage              (28 entries)
      9. other actions (5 entries):
             - pay_jail_fine          : {"player": acting_player, "current_gameboard": current_gameboard}
             - use_get_out_of_jail_card : {"player": acting_player, "current_gameboard": current_gameboard}
             - accept_trade_offer       : {"player": acting_player, "current_gameboard": current_gameboard}
             - skip_turn                : {}
             - concluded_actions        : {}
     10. buy_property              (1 entry)
     
    Args:
      acting_player: The acting player instance.
      current_gameboard: The current game board dict.
      schema_filepath: Path to the game schema file.
     
    Returns:
        The combined flat mapping list.
    """
    full_mapping = []
    full_mapping.extend(build_make_trade_offer_exchange_list(acting_player, schema_filepath))
    full_mapping.extend(build_make_trade_offer_sell_list(acting_player, schema_filepath))
    full_mapping.extend(build_make_trade_offer_buy_list(acting_player, schema_filepath))
    full_mapping.extend(build_improve_property_list(schema_filepath))
    full_mapping.extend(build_sell_house_hotel_list(schema_filepath))
    full_mapping.extend(build_sell_property_list(schema_filepath))
    full_mapping.extend(build_mortgage_property_list(schema_filepath))
    full_mapping.extend(build_free_mortgage_list(acting_player, current_gameboard, schema_filepath))
    # Pass acting_player and current_gameboard to other actions mapping.
    full_mapping.extend(build_other_actions_mapping(acting_player, current_gameboard))
    full_mapping.extend(build_buy_property_list(acting_player, current_gameboard, schema_filepath))
    return full_mapping
