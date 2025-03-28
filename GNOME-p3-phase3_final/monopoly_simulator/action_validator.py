# action_validator.py

def convert_cash_values(offer, rl_logger):
    """
    Convert cash_offered and cash_wanted in the offer dictionary if they are provided as strings.
    """
    cash_value_mapping = {
        "below_market": 5,
        "at_market": 10,
        "above_market": 15
    }
    # Convert cash_offered
    if "cash_offered" in offer and isinstance(offer["cash_offered"], str):
        offer_key = offer["cash_offered"].lower()
        if offer_key in cash_value_mapping:
            offer["cash_offered"] = cash_value_mapping[offer_key]
        else:
            rl_logger.error(f"Invalid cash_offered value: {offer['cash_offered']}")
            raise ValueError(f"Invalid cash_offered value: {offer['cash_offered']}")
    # Convert cash_wanted
    if "cash_wanted" in offer and isinstance(offer["cash_wanted"], str):
        offer_key = offer["cash_wanted"].lower()
        if offer_key in cash_value_mapping:
            offer["cash_wanted"] = cash_value_mapping[offer_key]
        else:
            rl_logger.error(f"Invalid cash_wanted value: {offer['cash_wanted']}")
            raise ValueError(f"Invalid cash_wanted value: {offer['cash_wanted']}")
    return offer

def convert_offer_properties(offer, current_gameboard, rl_logger):
    """
    Convert property names in property_set_offered and property_set_wanted to property objects
    using current_gameboard's available locations. Performs case-insensitive matching.
    """
    for key in ["property_set_offered", "property_set_wanted"]:
        if key in offer:
            converted_set = set()
            # Get locations list from current_gameboard.
            locations_field = current_gameboard.get("locations", {})
            if isinstance(locations_field, dict) and "location_states" in locations_field:
                locations_list = locations_field.get("location_states", [])
            elif isinstance(locations_field, list):
                locations_list = locations_field
            else:
                locations_list = []
            # Fallback to location_sequence if necessary.
            if not locations_list:
                locations_list = current_gameboard.get("location_sequence", [])
            
            available_names = [ (loc.get("name", "").lower() if isinstance(loc, dict) else getattr(loc, "name", "").lower()) for loc in locations_list ]
            for item in offer[key]:
                if isinstance(item, str):
                    prop_obj = next((loc for loc in locations_list 
                                     if ((loc.get("name", "") if isinstance(loc, dict) else getattr(loc, "name", "")).lower() == item.lower())), None)
                    if prop_obj is None:
                        rl_logger.error(f"Property '{item}' not found in current_gameboard locations. Available names: {available_names}")
                        raise ValueError(f"Property '{item}' not found in current_gameboard.")
                    converted_set.add(prop_obj)
                else:
                    converted_set.add(item)
            offer[key] = converted_set
    return offer

def validate_trade_offer_parameters(parameters, player, current_gameboard, rl_logger):
    """
    Performs conversion of cash values and conversion of offer property names.
    Also validates that the offered properties are actually owned by the player (using player.assets).
    """
    offer = parameters.get("offer", {})
    # Convert cash values.
    offer = convert_cash_values(offer, rl_logger)
    parameters["offer"] = offer

    # Ensure from_player is set.
    if "from_player" not in parameters:
        parameters["from_player"] = player

    # Convert the offer properties using current_gameboard locations.
    offer = convert_offer_properties(offer, current_gameboard, rl_logger)
    parameters["offer"] = offer

    # Validate that the offered properties are owned by the player.
    if hasattr(player, "assets"):
        owned_property_names = { (prop.get("name", "") if isinstance(prop, dict) else getattr(prop, "name", "")) for prop in player.assets }
    else:
        rl_logger.error(f"Player {player.player_name} does not have assets attribute.")
        raise ValueError("Player does not have an assets attribute.")

    valid_offer = set()
    for prop in offer.get("property_set_offered", set()):
        if isinstance(prop, dict):
            prop_name = prop.get("name", "")
        else:
            prop_name = getattr(prop, "name", "") if hasattr(prop, "name") else str(prop)
        if prop_name.lower() in {name.lower() for name in owned_property_names}:
            valid_offer.add(prop)
        else:
            rl_logger.debug(f"{player.player_name} does not own {prop_name}. Removing it from the offer.")

    if not valid_offer:
        rl_logger.error(f"After validation, {player.player_name} does not own any properties in the offer. Rejecting action.")
        return None  # Or alternatively, return a failure code indicator.
    
    offer["property_set_offered"] = valid_offer
    parameters["offer"] = offer
    return parameters

def convert_sell_offer_asset(parameters, current_gameboard, rl_logger):
    """
    For the 'make_sell_property_offer' action, convert the asset parameter from a string 
    (property name) to the property object.
    """
    if "asset" in parameters and isinstance(parameters["asset"], str):
        asset_name = parameters["asset"]
        locations_field = current_gameboard.get("locations", {})
        if isinstance(locations_field, dict) and "location_states" in locations_field:
            locations_list = locations_field.get("location_states", [])
        elif isinstance(locations_field, list):
            locations_list = locations_field
        else:
            locations_list = []
        if not locations_list:
            locations_list = current_gameboard.get("location_sequence", [])
        asset_obj = next((loc for loc in locations_list 
                          if ((loc.get("name", "") if isinstance(loc, dict) else getattr(loc, "name", "")).lower() == asset_name.lower())), None)
        if asset_obj is None:
            rl_logger.error(f"Asset '{asset_name}' not found in current_gameboard locations.")
            raise ValueError(f"Asset '{asset_name}' not found in current_gameboard.")
        parameters["asset"] = asset_obj
    return parameters