import sys
import os

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

# For buy_property mapping, we create a simple dummy mapping (1 entry) 
def build_buy_property_list():
    return [{"action": "buy_property", "parameters": {}}]

# For buy_property mapping, we create a simple dummy mapping (1 entry) 
def build_buy_property_list():
    return [{"action": "buy_property", "parameters": {}}]

def build_full_action_mapping(acting_player, schema_filepath="monopoly_game_schema_v1-2.json"):
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
      9. other actions (skip_turn, conclude_actions, use_get_out_of_jail_card, pay_jail_fine, accept_trade_offer)
         (5 entries)
     10. buy_property              (1 entry)
     
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
    full_mapping.extend(build_free_mortgage_list(schema_filepath))
    full_mapping.extend(build_other_actions_mapping())
    full_mapping.extend(build_buy_property_list())
    return full_mapping

if __name__ == "__main__":
    # Create an acting player using the Player class from our codebase.
    # Provide minimal default values for demonstration.
    acting_player = Player(
        current_position=0,
        status="waiting_for_move",
        has_get_out_of_jail_community_chest_card=False,
        has_get_out_of_jail_chance_card=False,
        current_cash=1500,
        num_railroads_possessed=0,
        player_name="player_1",
        assets=set(),
        full_color_sets_possessed=set(),
        currently_in_jail=False,
        num_utilities_possessed=0,
        agent=None  # Replace with an actual agent if needed.
    )
    
    full_mapping = build_full_action_mapping(acting_player)
    print("Total number of mapping entries:", len(full_mapping))
    print("First 5 mapping entries:")
    for entry in full_mapping[:5]:
        print(entry)
