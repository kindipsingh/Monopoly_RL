import logging

def convert_cash_values(offer, current_gameboard, rl_logger):
    """
    Convert cash_offered and cash_wanted in the offer dictionary if cash_offered is provided 
    as a string. This new conversion computes cash_offered using the property price contained in the 
    first property of property_set_wanted:
    
      - "below_market": 0.75 * property price
      - "at_market": 1.0 * property price
      - "above_market": 1.25 * property price

    The cash_wanted is forced to be 0.
    
    Args:
        offer (dict): The trade offer dictionary.
        current_gameboard (dict): The current gameboard containing available property locations.
        rl_logger: Logger instance for logging errors and debug messages.
    
    Returns:
        dict: The updated offer with numeric values for cash_offered and cash_wanted.
        
    Raises:
        ValueError: If conversion fails due to missing property or invalid string value.
    """
    multiplier_map = {
        "below_market": 0.75,
        "at_market": 1.0,
        "above_market": 1.25
    }
    # Convert cash_offered
    if "cash_offered" in offer and isinstance(offer["cash_offered"], str):
        key = offer["cash_offered"].lower()
        if key in multiplier_map:
            # We require at least one property in property_set_wanted to determine price.
            if "property_set_wanted" not in offer or not offer["property_set_wanted"]:
                rl_logger.error("No property in property_set_wanted to determine price for cash_offered conversion.")
                raise ValueError("Invalid offer: missing property_set_wanted for cash conversion.")
            # Take the first property from property_set_wanted.
            prop_item = next(iter(offer["property_set_wanted"]))
            property_price = None
            # If the property is a string, look it up in current_gameboard.
            if isinstance(prop_item, str):
                locations_field = current_gameboard.get("locations", {})
                if isinstance(locations_field, dict) and "location_states" in locations_field:
                    locations_list = locations_field.get("location_states", [])
                elif isinstance(locations_field, list):
                    locations_list = locations_field
                else:
                    locations_list = []
                if not locations_list:
                    locations_list = current_gameboard.get("location_sequence", [])
                prop_obj = next((loc for loc in locations_list 
                                 if ((loc.get("name", "") if isinstance(loc, dict) else getattr(loc, "name", "")).lower() 
                                     == prop_item.lower())), None)
                if prop_obj is None:
                    rl_logger.error(f"Property '{prop_item}' not found in current_gameboard locations during cash conversion.")
                    raise ValueError(f"Property '{prop_item}' not found in current_gameboard.")
                property_price = prop_obj.get("price") if isinstance(prop_obj, dict) else getattr(prop_obj, "price", None)
            else:
                # prop_item is already an object.
                property_price = prop_item.get("price") if isinstance(prop_item, dict) else getattr(prop_item, "price", None)
            if property_price is None:
                rl_logger.error("Property price not found for conversion.")
                raise ValueError("Cannot determine property price for cash conversion.")
            offer["cash_offered"] = int(multiplier_map[key] * property_price)
        else:
            rl_logger.error(f"Invalid cash_offered value: {offer['cash_offered']}")
            raise ValueError(f"Invalid cash_offered value: {offer['cash_offered']}")
    elif "cash_offered" in offer and not isinstance(offer["cash_offered"], (int, float)):
        try:
            offer["cash_offered"] = int(float(offer["cash_offered"]))
        except (ValueError, TypeError) as e:
            rl_logger.error(f"Trade offer conversion failed: invalid numeric value for 'cash_offered': {offer.get('cash_offered')}. Error: {e}")
            raise ValueError(f"Invalid value for cash_offered: {offer.get('cash_offered')}")
    
    # Force cash_wanted to be 0.
    offer["cash_wanted"] = 0
    return offer

def validate_make_sell_property_offer(parameters, current_gameboard, rl_logger):
    """
    Validates parameters for a make_sell_property_offer action.

    Expected parameters:
      - asset (str or property object): The property name (if string) representing the offered property.
      - to_player (object): The player object to whom the offer is being made. Must have a 'player_name' attribute.
      - price (numeric): The asking price for the asset.

    This function converts the asset from a string to the corresponding property object from current_gameboard.
    It also validates that the price is numeric and non-negative, and that to_player appears valid.

    Args:
        parameters (dict): The offer parameters.
        current_gameboard (dict): The current gameboard containing available property locations.
        rl_logger: Logger instance for logging errors and debug messages.

    Returns:
        dict: Updated parameters with converted 'asset' and numeric 'price'.

    Raises:
        ValueError: If a required parameter is missing or invalid.
    """
    # Check for required keys.
    for key in ["asset", "to_player", "price"]:
        if key not in parameters:
            rl_logger.error(f"validate_make_sell_property_offer failed: missing required key '{key}'")
            raise ValueError(f"Missing required parameter: {key}")

    # Validate and convert asset.
    if isinstance(parameters["asset"], str):
        asset_name = parameters["asset"]
        rl_logger.debug(f"validate_make_sell_property_offer: Converting asset '{asset_name}'")
        # Retrieve locations from current_gameboard.
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
        asset_obj = next(
            (
                loc for loc in locations_list
                if ((loc.get("name", "") if isinstance(loc, dict) else getattr(loc, "name", "")).lower() == asset_name.lower())
            ),
            None
        )
        if asset_obj is None:
            rl_logger.error(f"validate_make_sell_property_offer: Asset '{asset_name}' not found in current gameboard.")
            raise ValueError(f"Asset '{asset_name}' not found in current gameboard.")
        parameters["asset"] = asset_obj
        rl_logger.debug(f"validate_make_sell_property_offer: Asset converted to {asset_obj}.")
    # Otherwise, asset is assumed to already be a property object.

    # Validate price: ensure numeric value and non-negative.
    price = parameters["price"]
    try:
        numeric_price = int(float(price))
        if numeric_price < 0:
            rl_logger.error("validate_make_sell_property_offer: Price must be non-negative.")
            raise ValueError("Price must be non-negative.")
        parameters["price"] = numeric_price
    except (ValueError, TypeError) as e:
        rl_logger.error(f"validate_make_sell_property_offer: Invalid price value '{price}'. Error: {e}")
        raise ValueError(f"Invalid price value: {price}")

    # Validate the to_player parameter.
    to_player = parameters["to_player"]
    if not hasattr(to_player, "player_name"):
        rl_logger.error("validate_make_sell_property_offer: 'to_player' parameter is invalid; missing player_name attribute.")
        raise ValueError("Invalid to_player parameter; missing player_name attribute.")

    return parameters

def convert_offer_properties(offer, current_gameboard, rl_logger):
    """
    Convert property names in property_set_offered and property_set_wanted to property objects
    using current_gameboard's available locations. Performs case-insensitive matching.
    Changed to use lists rather than sets to avoid unhashable dict errors.
    """
    for key in ["property_set_offered", "property_set_wanted"]:
        if key in offer:
            converted_list = []
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
            
            available_names = [(loc.get("name", "").lower() if isinstance(loc, dict) 
                                 else getattr(loc, "name", "").lower()) for loc in locations_list]
            for item in offer[key]:
                if isinstance(item, str):
                    prop_obj = next((loc for loc in locations_list 
                                     if ((loc.get("name", "") if isinstance(loc, dict) 
                                          else getattr(loc, "name", "")).lower() == item.lower())), None)
                    if prop_obj is None:
                        rl_logger.error(f"Property '{item}' not found in current_gameboard locations. Available names: {available_names}")
                        raise ValueError(f"Property '{item}' not found in current_gameboard.")
                    converted_list.append(prop_obj)
                else:
                    converted_list.append(item)
            offer[key] = converted_list
    return offer

def validate_trade_offer_parameters(parameters, player, current_gameboard, rl_logger):
    """
    Performs conversion of cash values and conversion of offer property names.
    Also validates that the offered properties are actually owned by the player (using player.assets).
    Updated to use lists rather than sets for property collections.
    """
    offer = parameters.get("offer", {})
    # Convert cash values.
    offer = convert_cash_values(offer, current_gameboard, rl_logger)
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

    valid_offer = []
    for prop in offer.get("property_set_offered", []):
        if isinstance(prop, dict):
            prop_name = prop.get("name", "")
        else:
            prop_name = getattr(prop, "name", "") if hasattr(prop, "name") else str(prop)
        if prop_name.lower() in {name.lower() for name in owned_property_names}:
            valid_offer.append(prop)
        else:
            rl_logger.debug(f"{player.player_name} does not own {prop_name}. Removing it from the offer.")

    # If there were offered properties, but none of them are valid, reject the offer.
    if offer.get("property_set_offered", []) and not valid_offer:
        rl_logger.error(f"After validation, {player.player_name} does not own any properties in the offer. Rejecting action.")
        return None
    
    offer["property_set_offered"] = valid_offer
    parameters["offer"] = offer
    return parameters

def validate_sell_property(parameters, current_gameboard, rl_logger):
    """
    Validates and converts the 'asset' parameter in a sell property action.
    If the asset is provided as a string, this function attempts to convert it to the corresponding
    property object from current_gameboard. The conversion looks for matching property names in
    'locations' (using 'location_states' if available) and falls back to 'location_sequence'. It also logs
    the asset parameter before and after conversion.

    Args:
        parameters (dict): The dictionary containing the sell property parameters where 'asset' is expected.
        current_gameboard (dict): The gameboard containing available property locations.
        rl_logger: Logger instance for logging errors and debugging information.
    
    Returns:
        dict: The updated parameters with the 'asset' converted to a property object, if conversion is successful.
    
    Raises:
        ValueError: If no matching property is found for the given asset name.
    """
    if "asset" in parameters and isinstance(parameters["asset"], str):
        asset_name = parameters["asset"]
        rl_logger.debug(f"sell_property: Asset parameter before conversion: {asset_name}")
        # Retrieve locations from current_gameboard.
        locations_field = current_gameboard.get("locations", {})
        if isinstance(locations_field, dict) and "location_states" in locations_field:
            locations_list = locations_field.get("location_states", [])
        elif isinstance(locations_field, list):
            locations_list = locations_field
        else:
            locations_list = []
        # Fallback to location_sequence if no locations in 'locations'
        if not locations_list:
            locations_list = current_gameboard.get("location_sequence", [])
            
        # Attempt to find the property object matching the asset name.
        asset_obj = next(
            (
                loc for loc in locations_list
                if ((loc.get("name", "") if isinstance(loc, dict) else getattr(loc, "name", "")).lower() == asset_name.lower())
            ),
            None
        )
        if asset_obj is None:
            rl_logger.error(f"sell_property: Asset '{asset_name}' not found in current gameboard locations.")
            raise ValueError(f"Asset '{asset_name}' not found in current_gameboard.")
        parameters["asset"] = asset_obj
        rl_logger.debug(f"sell_property: Asset parameter after conversion: {parameters.get('asset')}")
    return parameters

def convert_sell_offer_asset(parameters, current_gameboard, rl_logger):
    """
    For the 'make_sell_property_offer' action, convert the asset parameter from a string 
    (property name) to the property object.
    
    This function is maintained for backward compatibility.
    New code should use validate_sell_property instead.
    """
    return validate_sell_property(parameters, current_gameboard, rl_logger)

def validate_free_mortgage(parameters, current_gameboard, rl_logger):
    """
    Validates and converts the 'asset' parameter in a free_mortgage action.
    If the asset is provided as a string, this function attempts to convert it
    to the corresponding property object from current_gameboard. It searches in
    'locations' (using 'location_states' if available) and falls back on
    'location_sequence' if necessary. The function logs the asset before and
    after conversion.

    Args:
        parameters (dict): The dictionary containing the free_mortgage action parameters where 'asset' is expected.
        current_gameboard (dict): The gameboard containing available property locations.
        rl_logger: Logger instance for logging errors and debugging information.

    Returns:
        dict: Updated parameters with the 'asset' converted to a property object.

    Raises:
        ValueError: If no matching property is found for the given asset name.
    """
    if "asset" in parameters and isinstance(parameters["asset"], str):
        asset_name = parameters["asset"]
        rl_logger.debug(f"free_mortgage: Asset parameter before conversion: {asset_name}")
        # Retrieve locations from current_gameboard.
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
            
        # Find the property object matching the asset name.
        asset_obj = next(
            (
                loc for loc in locations_list
                if ((loc.get("name", "") if isinstance(loc, dict) else getattr(loc, "name", "")).lower() == asset_name.lower())
            ),
            None
        )
        if asset_obj is None:
            rl_logger.error(f"free_mortgage: Asset '{asset_name}' not found in current gameboard locations.")
            raise ValueError(f"Asset '{asset_name}' not found in current_gameboard.")
        parameters["asset"] = asset_obj
        rl_logger.debug(f"free_mortgage: Asset parameter after conversion: {parameters.get('asset')}")
    return parameters

def validate_sell_house_hotel_asset(parameters, current_gameboard, rl_logger):
    """
    Validates and converts the 'asset' parameter in a sell house/hotel action.
    If the asset is provided as a string, the function attempts to map it to the corresponding
    property object from the current_gameboard. It searches in 'locations' (using 'location_states'
    if present) and falls back on 'location_sequence' if no locations are found.
    
    Args:
        parameters (dict): The dictionary containing the sell house/hotel action parameters where 'asset' is expected.
        current_gameboard (dict): The gameboard containing available property locations.
        rl_logger: Logger instance for logging errors and debugging information.
        
    Returns:
        dict: The updated parameters with the 'asset' converted to a property object, if conversion is successful.
        
    Raises:
        ValueError: If no matching property is found for the given asset name.
    """
    if "asset" in parameters and isinstance(parameters["asset"], str):
        asset_name = parameters["asset"]
        rl_logger.debug(f"sell_house_hotel: Asset parameter before conversion: {asset_name}")
        # Retrieve the locations list from the current_gameboard.
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
            
        # Find the property object matching the asset name.
        asset_obj = next(
            (
                loc for loc in locations_list
                if ((loc.get("name", "") if isinstance(loc, dict) else getattr(loc, "name", "")).lower() == asset_name.lower())
            ),
            None
        )
        if asset_obj is None:
            rl_logger.error(f"sell_house_hotel: Asset '{asset_name}' not found in the current gameboard locations.")
            raise ValueError(f"Asset '{asset_name}' not found in current_gameboard.")
        parameters["asset"] = asset_obj
        rl_logger.debug(f"sell_house_hotel: Asset parameter after conversion: {parameters.get('asset')}")
    return parameters

def validate_improve_property(parameters, current_gameboard, rl_logger):
    """
    Validates and converts the 'asset' parameter in an improve_property action.
    If the asset is provided as a string, this function attempts to convert it to the corresponding
    property object from current_gameboard. It searches in 'locations' (using 'location_states'
    if available) and falls back to 'location_sequence' if necessary.
    
    Args:
        parameters (dict): The dictionary containing the improve_property action parameters where 'asset' is expected.
        current_gameboard (dict): The gameboard containing available property locations.
        rl_logger: Logger instance for logging errors and debugging information.
    
    Returns:
        dict: The updated parameters with the 'asset' converted to a property object, if conversion is successful.
    
    Raises:
        ValueError: If no matching property is found for the given asset name.
    """
    if "asset" in parameters and isinstance(parameters["asset"], str):
        asset_name = parameters["asset"]
        rl_logger.debug(f"improve_property: Asset parameter before conversion: {asset_name}")
        # Retrieve locations from current_gameboard.
        locations_field = current_gameboard.get("locations", {})
        if isinstance(locations_field, dict) and "location_states" in locations_field:
            locations_list = locations_field.get("location_states", [])
        elif isinstance(locations_field, list):
            locations_list = locations_field
        else:
            locations_list = []
        # Fallback to location_sequence if needed.
        if not locations_list:
            locations_list = current_gameboard.get("location_sequence", [])
            
        # Locate the asset object matching the asset name (case-insensitive).
        asset_obj = next(
            (loc for loc in locations_list
             if ((loc.get("name", "") if isinstance(loc, dict) else getattr(loc, "name", "")).lower() 
                 == asset_name.lower())),
            None
        )
        if asset_obj is None:
            rl_logger.error(f"improve_property: Asset '{asset_name}' not found in current_gameboard locations.")
            raise ValueError(f"Asset '{asset_name}' not found in current_gameboard.")
        parameters["asset"] = asset_obj
        rl_logger.debug(f"improve_property: Asset parameter after conversion: {parameters.get('asset')}")
    return parameters

def validate_make_trade_offer(parameters, player, current_gameboard, rl_logger):
    """
    Validate parameters for a make_trade_offer action.

    This function ensures that the parameters have an 'offer' dictionary containing the keys
    'cash_offered' and 'cash_wanted'. It attempts to convert these values if they are strings
    (like 'below_market', 'at_market', 'above_market') and validates that they are non-negative.
    If any check fails, the function returns None indicating validation failure.
    """
    if "offer" not in parameters or not isinstance(parameters["offer"], dict):
        rl_logger.error("Trade offer validation failed: missing 'offer' dictionary in parameters.")
        return None

    offer = parameters["offer"]
    required_keys = ["cash_offered", "cash_wanted"]

    # First convert any string cash values using the updated function.
    try:
        offer = convert_cash_values(offer, current_gameboard, rl_logger)
    except ValueError:
        return None

    # Check that required keys exist and values are valid
    for key in required_keys:
        if key not in offer:
            rl_logger.error(f"Trade offer validation failed: missing required key '{key}' in offer.")
            return None
        
        # Ensure the values are numeric
        if not isinstance(offer[key], (int, float)):
            try:
                offer[key] = float(offer[key])
                rl_logger.debug(f"Converted '{key}' value to {offer[key]}.")
            except (ValueError, TypeError) as e:
                rl_logger.error(f"Trade offer validation failed: invalid value for '{key}': {offer.get(key)}. Error: {e}")
                return None

        # Additional check: ensure that the cash values are not negative
        if offer[key] < 0:
            rl_logger.error(f"Trade offer validation failed: '{key}' must be non-negative, got {offer[key]}.")
            return None

    # Ensure from_player is set
    if "from_player" not in parameters:
        parameters["from_player"] = player

    # Convert and validate property sets if present
    try:
        offer = convert_offer_properties(offer, current_gameboard, rl_logger)
    except ValueError:
        return None

    # Validate that offered properties are owned by the player
    if "property_set_offered" in offer and offer["property_set_offered"]:
        if not hasattr(player, "assets"):
            rl_logger.error(f"Player {player.player_name} does not have assets attribute.")
            return None
            
        owned_property_names = {(prop.get("name", "") if isinstance(prop, dict) else getattr(prop, "name", "")).lower() 
                                for prop in player.assets}
        
        valid_offer = []
        for prop in offer["property_set_offered"]:
            if isinstance(prop, dict):
                prop_name = prop.get("name", "")
            else:
                prop_name = getattr(prop, "name", "") if hasattr(prop, "name") else str(prop)
            if prop_name.lower() in owned_property_names:
                valid_offer.append(prop)
            else:
                rl_logger.debug(f"{player.player_name} does not own {prop_name}. Removing it from the offer.")
        
        if offer["property_set_offered"] and not valid_offer:
            rl_logger.error(f"After validation, {player.player_name} does not own any properties in the offer. Rejecting action.")
            return None
            
        offer["property_set_offered"] = valid_offer

    parameters["offer"] = offer
    return parameters

# ------------------------------
# Main function for testing
# ------------------------------
if __name__ == "__main__":
    # Set up a basic logger for testing
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s:%(name)s:%(levelname)s:%(message)s")
    rl_logger = logging.getLogger("action_validator_test")

    # Create a dummy current gameboard.
    # We add a 'locations' key as a list containing a property for "Electric Company"
    current_gameboard = {
        "locations": [
            {"name": "Electric Company", "price": 150, "owned_by": "Player1"},
            {"name": "Water Works", "price": 150, "owned_by": "Player2"}
        ]
    }

    # Define the input offer object to be tested.
    test_offer = {
        "property_set_offered": [],
        "property_set_wanted": ["Electric Company"],
        "cash_offered": "below_market",
        "cash_wanted": 0
    }

    rl_logger.debug("Original offer: %s", test_offer)
    try:
        # Call the updated cash conversion function.
        converted_offer = convert_cash_values(test_offer.copy(), current_gameboard, rl_logger)
        # Optionally, also convert the offer properties.
        converted_offer = convert_offer_properties(converted_offer, current_gameboard, rl_logger)
        rl_logger.info("Converted offer: %s", converted_offer)
    except Exception as e:
        rl_logger.error("Conversion failed with error: %s", e)
        
    # Test the validate_sell_property function
    rl_logger.debug("\nTesting validate_sell_property function:")
    test_params = {
        "asset": "Electric Company"
    }
    try:
        updated_params = validate_sell_property(test_params.copy(), current_gameboard, rl_logger)
        rl_logger.info("Sell property parameters validated and converted successfully.")
        rl_logger.info(f"Converted asset: {updated_params['asset']}")
    except Exception as e:
        rl_logger.error("Validation failed with error: %s", e)
        
    # Test the validate_sell_house_hotel_asset function
    rl_logger.debug("\nTesting validate_sell_house_hotel_asset function:")
    test_params = {
        "asset": "Electric Company"
    }
    try:
        updated_params = validate_sell_house_hotel_asset(test_params.copy(), current_gameboard, rl_logger)
        rl_logger.info("Sell house/hotel parameters validated and converted successfully.")
        rl_logger.info(f"Converted asset: {updated_params['asset']}")
    except Exception as e:
        rl_logger.error("Validation failed with error: %s", e)
        
    # Test the validate_free_mortgage function
    rl_logger.debug("\nTesting validate_free_mortgage function:")
    test_params = {
        "asset": "Electric Company"
    }
    try:
        updated_params = validate_free_mortgage(test_params.copy(), current_gameboard, rl_logger)
        rl_logger.info("Free mortgage parameters validated and converted successfully.")
        rl_logger.info(f"Converted asset: {updated_params['asset']}")
    except Exception as e:
        rl_logger.error("Validation failed with error: %s", e)
        
    # Test the validate_improve_property function
    rl_logger.debug("\nTesting validate_improve_property function:")
    test_params = {
        "asset": "Electric Company"
    }
    try:
        updated_params = validate_improve_property(test_params.copy(), current_gameboard, rl_logger)
        rl_logger.info("Improve property parameters validated and converted successfully.")
        rl_logger.info(f"Converted asset: {updated_params['asset']}")
    except Exception as e:
        rl_logger.error("Validation failed with error: %s", e)
