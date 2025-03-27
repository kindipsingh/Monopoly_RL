import os
import json
from monopoly_simulator.player import Player  # Import the Player class from the code base

def load_property_objects_from_schema(schema_filepath="monopoly_game_schema_v1-2.json"):
    """
    Load the list of property objects from the JSON schema file.

    This function attempts to read the property objects from the schema.
    It first checks for a "properties" key. If not found, it looks inside the
    "locations" object under "location_states" and filters for locations with
    loc_class in {"real_estate", "railroad", "utility"}.

    Returns:
        A list of 28 property objects. Each property object must include:
          - "name": the property's name.
          - "loc_class": typically one of "real_estate", "railroad", or "utility".
    
    Raises:
        KeyError, TypeError, or ValueError if the expected property list is not found,
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

def build_sell_property_list(schema_filepath="monopoly_game_schema_v1-2.json"):
    """
    Build a flat list for the "sell_property" action.

    This action requires a mapping for each property.
    The resulting list consists of 28 entries where each entry is a flat dictionary with:
      - "action": "sell_property"
      - "parameters": { "property": <property_name> }

    Returns:
        A list of 28 dictionaries.
    """
    properties = load_property_objects_from_schema(schema_filepath)
    
    if len(properties) != 28:
        raise ValueError(f"Expected 28 property objects but got {len(properties)}")
    
    flat_mapping = []
    for prop in properties:
        mapping_entry = {
            "action": "sell_property",
            "parameters": {
                "property": prop["name"]
            }
        }
        flat_mapping.append(mapping_entry)
    
    if len(flat_mapping) != 28:
        raise ValueError(f"Expected 28 entries but got {len(flat_mapping)}")
    
    return flat_mapping

# Example usage:
if __name__ == "__main__":
    mapping_list = build_sell_property_list()
    print("Total number of entries:", len(mapping_list))
    print("Sample entries:")
    for entry in mapping_list[:5]:
        print(entry)