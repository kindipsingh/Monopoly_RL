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
        A list of property objects. We expect a total of 28 property objects, where
        22 of these are "real_estate", 4 "railroad" and 2 "utility".
    
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

def build_sell_house_hotel_list(player,current_gameboard,schema_filepath="monopoly_game_schema_v1-2.json"):
    """
    Build a flat list for the "sell_house_hotel" action.

    For this action, we want a 44-dimensional space based on 22 viable "real_estate" properties
    and two options for the sale:
      - Selling a house (sell_house=True, sell_hotel=False)
      - Selling a hotel (sell_house=False, sell_hotel=True)

    Returns:
        A list of 44 dictionaries. Each dictionary has:
            - "action": "sell_house_hotel"
            - "parameters": a flat dictionary with keys:
                  "asset": name of the real estate property,
                  "sell_house": boolean indicating if selling a house,
                  "sell_hotel": boolean indicating if selling a hotel
                  
    Total combinations: 22 (properties) x 2 (sale options) = 44.
    """
    properties = load_property_objects_from_schema(schema_filepath)
    # Filter only "real_estate" properties.
    real_estate_properties = [prop for prop in properties if prop.get("loc_class") == "real_estate"]
    
    if len(real_estate_properties) != 22:
        raise ValueError(f"Expected 22 real estate properties, but got {len(real_estate_properties)}")
    
    flat_mapping = []
    for prop in real_estate_properties:
        # Mapping entry for selling a house.
        mapping_house = {
            "action": "sell_house_hotel",
            "parameters": {
                "player":player,
                "asset": prop["name"],
                "current_gameboard":current_gameboard,
                "sell_house": True,
                "sell_hotel": False
            }
        }
        # Mapping entry for selling a hotel.
        mapping_hotel = {
            "action": "sell_house_hotel",
            "parameters": {
                "player":player,
                "asset": prop["name"],
                "current_gameboard":current_gameboard,
                "sell_house": False,
                "sell_hotel": True
            }
        }
        flat_mapping.append(mapping_house)
        flat_mapping.append(mapping_hotel)
    
    if len(flat_mapping) != 44:
        raise ValueError(f"Expected 44 entries but got {len(flat_mapping)}")
    
    return flat_mapping
