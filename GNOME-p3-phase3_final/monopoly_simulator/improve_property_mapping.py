import os
import json
from monopoly_simulator.player import Player  # Import the Player class from the code base

def load_property_objects_from_schema(schema_filepath="monopoly_game_schema_v1-2.json"):
    """
    
    Load the list of property objects from the JSON schema file.

    This function attempts to read the property objects from a schema. It first
    checks for a "properties" key. If not found, it looks in the "locations" object
    under "location_states" and filters for location objects with "loc_class" in
    {"real_estate", "railroad", "utility"}.

    Returns:
        A list of property objects. In our context, we expect a total of 28 property objects,
        where 22 of these are "real_estate", 4 "railroad" and 2 "utility".
    
    Raises:
        KeyError, TypeError, or ValueError if the expected property list is not found
        or if its length is not 28.
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
        raise TypeError(f"Expected the property list to be a list in {schema_filepath}")
    
    if len(property_list) != 28:
        raise ValueError(f"Expected 28 property objects, but got {len(property_list)} from {schema_filepath}")
    
    for prop in property_list:
        if not isinstance(prop, dict):
            raise TypeError("Each property must be a dictionary with 'name' and 'loc_class'")
        if "name" not in prop or "loc_class" not in prop:
            raise ValueError("Each property object must contain both 'name' and 'loc_class'")
    
    return property_list

def build_improve_property_list(player, current_gameboard, schema_filepath="monopoly_game_schema_v1-2.json"):
    """
    Build a flat list for the "improve_property" action mapping.

    Each mapping entry contains:
      - "action": "improve_property"
      - "parameters": a dictionary with keys:
            "player": the player object,
            "asset": the property name (string),  # Will later be converted if needed.
            "current_gameboard": the current gameboard dictionary.
    
    Returns:
        A list of mapping dictionaries for improve_property.
    """
    properties = load_property_objects_from_schema(schema_filepath)
    if len(properties) != 28:
        raise ValueError(f"Expected 28 property objects but got {len(properties)}")
    
    mapping_list = []
    for prop in properties:
        mapping_entry = {
            "action": "improve_property",
            "parameters": {
                "player": player,
                "asset": prop["name"],
                "current_gameboard": current_gameboard
            }
        }
        mapping_list.append(mapping_entry)
    
    return mapping_list

# Example usage:
if __name__ == "__main__":
    # Create dummy player and gameboard for testing
    dummy_player = Player(1000, "P1", "red")
    dummy_gameboard = {"dummy": "gameboard"}
    
    mapping_list = build_improve_property_list(dummy_player, dummy_gameboard)
    print("Total number of entries:", len(mapping_list))
    print("Sample entries:")
    for entry in mapping_list[:5]:
        print(entry)
