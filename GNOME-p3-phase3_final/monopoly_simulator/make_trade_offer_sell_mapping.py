import os
import json
from monopoly_simulator.player import Player  # Import the Player class from the code base

def load_property_objects_from_schema(schema_filepath="monopoly_game_schema_v1-2.json"):
    """
    Load the list of property objects from the JSON schema file.

    This version handles a schema in which property objects are not at the top level.
    It first attempts to read the list from "properties" and, if not found, then looks in the 
    "locations" object under "location_states" and filters for those with loc_class in {"real_estate", "railroad", "utility"}.

    Returns:
        A list of 28 property objects. Each property object must include at least:
          - "name": the property's name.
          - "loc_class": typically one of "real_estate", "railroad", or "utility".
          
    Raises:
        KeyError, TypeError, or ValueError if the property list cannot be found or does not have 28 entries.
    """
    if not os.path.isabs(schema_filepath):
        schema_filepath = os.path.join(os.path.dirname(__file__), schema_filepath)
    
    try:
        with open(schema_filepath, "r") as f:
            schema = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Error reading {schema_filepath}: {e}")
    
    if "properties" in schema:
        property_list = schema["properties"]
    elif "locations" in schema and "location_states" in schema["locations"]:
        property_list = schema["locations"]["location_states"]
        property_list = [loc for loc in property_list if loc.get("loc_class") in {"real_estate", "railroad", "utility"}]
    else:
        raise KeyError(f"Could not find a valid property list in {schema_filepath}.")
    
    if not isinstance(property_list, list):
        raise TypeError(f"Property list is not a list in {schema_filepath}")
    
    if len(property_list) != 28:
        raise ValueError(f"Expected 28 property objects, but got {len(property_list)} from {schema_filepath}")
    
    for prop in property_list:
        if not isinstance(prop, dict):
            raise TypeError("Each property must be a dictionary with 'name' and 'loc_class'")
        if "name" not in prop or "loc_class" not in prop:
            raise ValueError("Each property object must contain both 'name' and 'loc_class'")
    
    return property_list

def build_make_trade_offer_sell_list(acting_player, schema_filepath="monopoly_game_schema_v1-2.json"):
    """
    Build a flat list for the "make_trade_offer_sell" action.

    This action is represented with a 252-dimensional space based on:
      - To player: 3 possible choices (the three other players chosen cyclically).
      - Property offered: 28 options (from the property list read from the schema).
      - Cash requested: 3 discretized values corresponding to:
           "below_market" (0.75 x purchase price), "at_market" (1 x purchase price), and 
           "above_market" (1.25 x purchase price).

    For an acting player "player_N", the allowed target players are computed cyclically as:
      "player_{((N-1+1)%4)+1}", "player_{((N-1+2)%4)+1}", "player_{((N-1+3)%4)+1}"
    
    Returns:
        A list of 252 dictionaries. Each dictionary has:
            - "action": "make_trade_offer_sell"
            - "parameters": a flat dictionary with keys:
                "to_player": target player's name
                "property_offered": offered property name
                "cash_requested": one of ["below_market", "at_market", "above_market"]
                
    Total combinations: 3 x 28 x 3 = 252.
    """
    if not isinstance(acting_player, Player):
        raise TypeError("acting_player must be an instance of Player from the code base.")
    
    try:
        acting_index = int(acting_player.player_name.split('_')[-1])
    except (ValueError, AttributeError) as e:
        raise ValueError("player_name must be in the format 'player_N' where N is an integer") from e
    
    allowed_targets = []
    for offset in range(1, 4):
        target_index = ((acting_index - 1 + offset) % 4) + 1
        allowed_targets.append(f"player_{target_index}")
    
    properties = load_property_objects_from_schema(schema_filepath)
    
    cash_categories = ["below_market", "at_market", "above_market"]
    
    flat_mapping = []
    for target in allowed_targets:
        for offered_property in properties:
            for cash in cash_categories:
                mapping_entry = {
                    "action": "make_trade_offer_sell",
                    "parameters": {
                        "to_player": target,
                        "property_offered": offered_property["name"],
                        "cash_requested": cash
                    }
                }
                flat_mapping.append(mapping_entry)
    
    if len(flat_mapping) != 252:
        raise ValueError(f"Expected 252 entries but got {len(flat_mapping)}")
    
    return flat_mapping

# Example usage:
if __name__ == "__main__":
    # Creating a dummy subclass of Player for demonstration.
    class DummyPlayer(Player):
        def __init__(self, name):
            super().__init__(current_position=0, status="active", has_get_out_of_jail_community_chest_card=False,
                             has_get_out_of_jail_chance_card=False, current_cash=1500, num_railroads_possessed=0,
                             player_name=name, assets=set(), full_color_sets_possessed=set(), currently_in_jail=False,
                             num_utilities_possessed=0, agent=None)
    
    acting_player = DummyPlayer("player_1")
    mapping_list = build_make_trade_offer_sell_list(acting_player)
    print("Total number of entries:", len(mapping_list))
    print("Sample entries:")
    for entry in mapping_list[:5]:
        print(entry)
    
    # Uncomment the next line to use the mapping list in your application:
    # mapping_list