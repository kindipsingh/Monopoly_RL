import os
import json
from monopoly_simulator.player import Player  # Import the Player class from the code base

def load_property_objects_from_schema(schema_filepath="monopoly_game_schema_v1-2.json"):
    """
    Load the list of property objects from the JSON schema file.

    This version handles a schema in which property objects are not stored at the top level.
    Instead, it looks in the "locations" object under the key "location_states" and filters out
    entries whose "loc_class" is one of {"real_estate", "railroad", "utility"}.

    Returns:
        A list of 28 property objects. Each property object must contain at least:
           - "name": the property's name.
           - "loc_class": typically "real_estate", "railroad", or "utility".
    
    Raises:
        KeyError, TypeError, or ValueError if the expected structure is not found or if the
        number of property objects does not equal 28.
    """
    # Build an absolute path relative to this module, if the provided path is not absolute.
    if not os.path.isabs(schema_filepath):
        schema_filepath = os.path.join(os.path.dirname(__file__), schema_filepath)
    
    try:
        with open(schema_filepath, "r") as f:
            schema = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Error reading {schema_filepath}: {e}")
    
    # First, try to get the property list directly if available.
    if "properties" in schema:
        property_list = schema["properties"]
    # Otherwise, assume a structure with a "locations" key.
    elif "locations" in schema and "location_states" in schema["locations"]:
        property_list = schema["locations"]["location_states"]
        # Filter out only the property objects that have loc_class of interest.
        property_list = [loc for loc in property_list if loc.get("loc_class") in {"real_estate", "railroad", "utility"}]
    else:
        raise KeyError(f"Could not find a valid property list in {schema_filepath}.")
    
    if not isinstance(property_list, list):
        raise TypeError(f"Expected the property list to be a list in {schema_filepath}")
    
    if len(property_list) != 28:
        raise ValueError(f"Expected 28 property objects, but got {len(property_list)} from {schema_filepath}")
    
    for prop in property_list:
        if not isinstance(prop, dict):
            raise TypeError("Each property must be a dictionary containing at least 'name' and 'loc_class'")
        if "name" not in prop or "loc_class" not in prop:
            raise ValueError("Each property object must contain both 'name' and 'loc_class' keys")
    
    return property_list

def build_make_trade_offer_exchange_list(acting_player, schema_filepath="monopoly_game_schema_v1-2.json"):
    """
    Build a flat list for the "make_trade_offer_exchange" action.

    Reads the list of property objects from the provided JSON schema file.

    Parameters:
        acting_player: An instance of Player with an attribute `player_name` formatted as "player_N", where N is an integer 1-4.
        schema_filepath: Path to the JSON schema file containing property objects.
    
    Returns:
        A list of 2268 dictionaries. Each dictionary represents one allowed combination with:
            - "action": The string "make_trade_offer_exchange"
            - "parameters": A flat dictionary containing:
                   {"to_player": <target_player>, "property_offered": <offered_property_name>,
                    "property_requested": <requested_property_name>}
    
    The allowed target players for an acting player "player_N" are:
         "player_{((N-1+1)%4)+1}", "player_{((N-1+2)%4)+1}", "player_{((N-1+3)%4)+1}"
         
    For each offered property (from the list of 28 properties), the requested property is any of the remaining 27.
    
    Total combinations: 3 * 28 * 27 = 2268.
    """
    if not isinstance(acting_player, Player):
        raise TypeError("acting_player must be an instance of Player from the code base.")
        
    # Extract acting player's number.
    try:
        acting_index = int(acting_player.player_name.split('_')[-1])
    except (ValueError, AttributeError) as e:
        raise ValueError("player_name must be in the format 'player_N' where N is an integer") from e

    # Compute allowed targets cyclically.
    allowed_targets = []
    for offset in range(1, 4):
        target_index = ((acting_index - 1 + offset) % 4) + 1
        allowed_targets.append(f"player_{target_index}")
    
    # Load property objects from JSON schema.
    properties = load_property_objects_from_schema(schema_filepath)
    
    flat_mapping = []
    # For each allowed target, each offered property and for each requested property (ensuring they are different)
    for target in allowed_targets:
        for offered_property in properties:
            for requested_property in (p for p in properties if p["name"] != offered_property["name"]):
                mapping_entry = {
                    "action": "make_trade_offer_exchange",
                    "parameters": {
                        "to_player": target,
                        "property_offered": offered_property["name"],
                        "property_requested": requested_property["name"]
                    }
                }
                flat_mapping.append(mapping_entry)
    
    if len(flat_mapping) != 2268:
        raise ValueError(f"Expected 2268 entries but got {len(flat_mapping)}")
    
    return flat_mapping

# Example usage:
if __name__ == "__main__":
    # For demonstration, we create a dummy subclass of Player that requires only player_name.
    # In the real codebase, Player is fully implemented in monopoly_simulator/player.py.
    class DummyPlayer(Player):
        def __init__(self, name):
            # Call the original initializer with dummy values for parameters not needed here.
            # These dummy values are only used to allow accessing the player_name property.
            # Adjust as appropriate for your testing environment.
            super().__init__(current_position=0, status="active", has_get_out_of_jail_community_chest_card=False,
                             has_get_out_of_jail_chance_card=False, current_cash=1500, num_railroads_possessed=0,
                             player_name=name, assets=set(), full_color_sets_possessed=set(), currently_in_jail=False,
                             num_utilities_possessed=0, agent=None)
    
    acting_player = DummyPlayer("player_1")
    mapping_list = build_make_trade_offer_exchange_list(acting_player)
    print("Total number of entries:", len(mapping_list))
    print("Sample entries)")
    print(len(mapping_list))
    for entry in mapping_list[:5]:
        print(entry)
