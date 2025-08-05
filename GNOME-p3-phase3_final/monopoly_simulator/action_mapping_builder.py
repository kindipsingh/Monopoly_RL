import logging
import sys
import json
from typing import Dict, List, Any, Tuple
import numpy as np
from monopoly_simulator.action_encoding import ActionEncoder

# Configure logging
logger = logging.getLogger('monopoly_simulator.action_mapping_builder')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Get the RL agent logger
rl_logger = logging.getLogger('rl_agent_logs')

def build_complete_action_mapping(player, current_gameboard, game_phase: str) -> List[Dict[str, Any]]:
    """
    Builds a complete action mapping for a player in the given game phase.
    
    This function uses the ActionEncoder to build the full action mapping,
    which contains all possible actions and their required parameters.
    
    Args:
        player: The player object for whom to build the action mapping
        current_gameboard: The current state of the game board
        game_phase: The current phase of the game ("pre_roll", "post_roll", or "out_of_turn")
    
    Returns:
        List of dictionaries, each containing:
        - "action": the action name or function
        - "parameters": a dictionary of parameters required for execution
    """
    logger.debug(f"Building complete action mapping for player {player.player_name} in {game_phase} phase")
    
    # Initialize the action encoder
    action_encoder = ActionEncoder()
    
    # Build the full action mapping
    full_mapping = action_encoder.build_full_action_mapping(player, current_gameboard)
    
    logger.info(f"Built complete action mapping with {len(full_mapping)} actions for {player.player_name} in {game_phase} phase")
    
    return full_mapping
def check_sell_property_offer_validity(mapping_entry: Dict, current_gameboard: Dict, game_phase: str) -> bool:
    """
    Checks if a sell property offer action would be valid without actually executing it.
    
    This function examines the parameters of a make_sell_property_offer action to determine
    if it would return a failure code based on the rules for selling properties.
    It also considers the current game phase to ensure the action is allowed in that phase.
    
    Args:
        mapping_entry: The action mapping entry containing the action and parameters
        current_gameboard: The current state of the game board
        game_phase: The current phase of the game ("pre_roll", "post_roll", or "out_of_turn")
        
    Returns:
        bool: True if the sell property offer would be valid, False otherwise
    """
    # First check if sell property offers are allowed in the current game phase
    if game_phase not in ["pre_roll", "out_of_turn"]:
        logger.debug(f"Sell property offers are not allowed in the {game_phase} phase")
        return False
    
    # Extract parameters from the mapping entry
    params = mapping_entry["parameters"]
    to_player_name = params["to_player"]
    asset_name = params["asset"]
    price_category = params["price"]
    
    # Find the from_player (current player making the offer)
    from_player = None
    for player in current_gameboard.get("players", []):
        if hasattr(player, "status") and player.status != "lost" and player.status != "spectating":
            # Assuming the current player is the one making the offer
            if player == current_gameboard.get("current_player"):
                from_player = player
                break
    
    if from_player is None:
        logger.debug("Could not determine the player making the offer")
        return False
    
    # Find the to_player object from the name
    to_player = None
    for player in current_gameboard.get("players", []):
        if hasattr(player, "player_name") and player.player_name == to_player_name:
            to_player = player
            break
    
    if to_player is None:
        logger.debug(f"Player {to_player_name} not found in gameboard")
        return False
    
    # Check if the to_player has lost the game
    if hasattr(to_player, "status") and to_player.status == 'lost':
        logger.debug(f"Sell offer is being made to player {to_player_name} who has lost the game")
        return False
    
    # Check if the to_player already has an outstanding trade offer
    if hasattr(to_player, 'is_trade_offer_outstanding') and to_player.is_trade_offer_outstanding:
        logger.debug(f"Player {to_player_name} already has a trade offer outstanding")
        return False
    
    # Find the property object from its name
    asset = None
    for location in current_gameboard.get('location_sequence', []):
        if hasattr(location, 'name') and location.name == asset_name:
            asset = location
            break
    
    if asset is None:
        logger.debug(f"Property {asset_name} not found in gameboard")
        return False
    
    # Check if from_player owns the property
    if not hasattr(asset, 'owned_by') or asset.owned_by != from_player:
        logger.debug(f"Player {from_player.player_name} does not own {asset_name}")
        return False
    
    # Check if the property has improvements
    if asset.loc_class == 'real_estate' and hasattr(asset, 'num_houses') and hasattr(asset, 'num_hotels') and (asset.num_houses > 0 or asset.num_hotels > 0):
        logger.debug(f"Property {asset_name} has improvements")
        return False
    
    # Check if the property is mortgaged
    if (asset.loc_class in ['real_estate', 'railroad', 'utility']) and hasattr(asset, 'is_mortgaged') and asset.is_mortgaged:
        logger.debug(f"Property {asset_name} is mortgaged")
        return False
    
    # Check if the property is part of a monopoly with improvements
    if asset.loc_class == 'real_estate' and hasattr(asset, 'color'):
        color = asset.color
        if color in current_gameboard.get('color_assets', {}):
            for prop in current_gameboard['color_assets'][color]:
                if (hasattr(prop, 'owned_by') and prop.owned_by == from_player and 
                    hasattr(prop, 'num_houses') and hasattr(prop, 'num_hotels') and 
                    (prop.num_houses > 0 or prop.num_hotels > 0)):
                    logger.debug(f"Cannot sell {asset_name} because another property in the same color set has improvements")
                    return False
    
    # Calculate the actual price based on the price category
    if hasattr(asset, 'price'):
        base_price = asset.price
        if price_category == "below_market":
            actual_price = base_price * 0.75
        elif price_category == "at_market":
            actual_price = base_price
        elif price_category == "above_market":
            actual_price = base_price * 1.25
        else:
            logger.debug(f"Invalid price category: {price_category}")
            return False
        
        # Check if to_player has enough cash for the purchase
        if to_player.current_cash < actual_price:
            logger.debug(f"Player {to_player_name} does not have enough cash ({to_player.current_cash}) to buy {asset_name} at {actual_price}")
            return False
    else:
        logger.debug(f"Property {asset_name} does not have a price attribute")
        return False
    
    # If all checks pass, the sell property offer is valid
    return True

def check_buy_property_offer_validity(mapping_entry: Dict, current_gameboard: Dict, game_phase: str) -> bool:
    """
    Checks if a buy property offer action would be valid without actually executing it.
    
    This function examines the parameters of a make_trade_offer action where the player is offering cash
    to buy a property from another player. It determines if it would return a failure code based on
    the rules for buying properties through trade offers.
    
    Args:
        mapping_entry: The action mapping entry containing the action and parameters
        current_gameboard: The current state of the game board
        game_phase: The current phase of the game ("pre_roll", "post_roll", or "out_of_turn")
        
    Returns:
        bool: True if the buy property offer would be valid, False otherwise
    """
    # First check if trade offers are allowed in the current game phase
    if game_phase not in ["pre_roll", "out_of_turn"]:
        logger.debug(f"Buy property offers are not allowed in the {game_phase} phase")
        return False
    
    # Extract parameters from the mapping entry
    params = mapping_entry["parameters"]
    from_player = params["from_player"]
    to_player_name = params["to_player"]
    offer = params["offer"]
    
    # Verify this is a buy offer (cash offered, no cash wanted, no properties offered, one property wanted)
    if (offer['cash_wanted'] != 0 or 
        len(offer['property_set_offered']) != 0 or 
        len(offer['property_set_wanted']) != 1):
        logger.debug("Not a valid buy property offer structure")
        return False
    
    cash_offered = offer['cash_offered']
    property_wanted = next(iter(offer['property_set_wanted']))  # Get the single property name
    
    # Find the to_player object from the name
    to_player = None
    for player in current_gameboard.get("players", []):
        if hasattr(player, "player_name") and player.player_name == to_player_name:
            to_player = player
            break
    
    if to_player is None:
        logger.debug(f"Player {to_player_name} not found in gameboard")
        return False
    
    # Check if the to_player has lost the game
    if hasattr(to_player, "status") and to_player.status == 'lost':
        logger.debug(f"Buy offer is being made to player {to_player_name} who has lost the game")
        return False
    
    # Check if the to_player already has an outstanding trade offer
    if hasattr(to_player, 'is_trade_offer_outstanding') and to_player.is_trade_offer_outstanding:
        logger.debug(f"Player {to_player_name} already has a trade offer outstanding")
        return False
    
    # Check if cash offered is negative
    if cash_offered < 0:
        logger.debug("Cash offered cannot be negative")
        return False
    
    # Check if from_player has enough cash for the offer
    if from_player.current_cash < cash_offered:
        logger.debug(f"Player {from_player.player_name} does not have enough cash ({from_player.current_cash}) for this offer of {cash_offered}")
        return False
    
    # Find the property object from its name
    wanted_property = None
    for location in current_gameboard.get('location_sequence', []):
        if hasattr(location, 'name') and location.name == property_wanted:
            wanted_property = location
            break
    
    if wanted_property is None:
        logger.debug(f"Property {property_wanted} not found in gameboard")
        return False
    
    # Check if to_player owns the property
    if not hasattr(wanted_property, 'owned_by') or wanted_property.owned_by != to_player:
        logger.debug(f"Player {to_player_name} does not own {property_wanted}")
        return False
    
    # Check if the property has improvements
    if wanted_property.loc_class == 'real_estate' and hasattr(wanted_property, 'num_houses') and hasattr(wanted_property, 'num_hotels') and (wanted_property.num_houses > 0 or wanted_property.num_hotels > 0):
        logger.debug(f"Property {property_wanted} has improvements")
        return False
    
    # Check if the property is mortgaged
    if (wanted_property.loc_class in ['real_estate', 'railroad', 'utility']) and hasattr(wanted_property, 'is_mortgaged') and wanted_property.is_mortgaged:
        logger.debug(f"Property {property_wanted} is mortgaged")
        return False
    
    # Check if the property is part of a monopoly with improvements
    if wanted_property.loc_class == 'real_estate' and hasattr(wanted_property, 'color'):
        color = wanted_property.color
        if color in current_gameboard.get('color_assets', {}):
            for prop in current_gameboard['color_assets'][color]:
                if (hasattr(prop, 'owned_by') and prop.owned_by == to_player and 
                    hasattr(prop, 'num_houses') and hasattr(prop, 'num_hotels') and 
                    (prop.num_houses > 0 or prop.num_hotels > 0)):
                    logger.debug(f"Cannot buy {property_wanted} because another property in the same color set has improvements")
                    return False
    
    # Check if the offer is reasonable (optional)
    # A reasonable offer might be within a certain range of the property's value
    if hasattr(wanted_property, 'price'):
        base_price = wanted_property.price
        # Check if offer is too low (e.g., less than 50% of base price)
        if cash_offered < base_price * 0.5:
            logger.debug(f"Offer of {cash_offered} for {property_wanted} seems too low compared to base price {base_price}")
            # Note: We're not returning False here as this is just a warning, not a hard rule
        
        # Check if offer is suspiciously high (e.g., more than 3x base price)
        if cash_offered > base_price * 3:
            logger.debug(f"Offer of {cash_offered} for {property_wanted} seems suspiciously high compared to base price {base_price}")
            # Note: We're not returning False here as this is just a warning, not a hard rule
    
    # If all checks pass, the buy property offer is valid
    return True


def check_improve_property_validity(mapping_entry: Dict, current_gameboard: Dict, game_phase: str) -> bool:
    """
    Checks if an improve property action would be valid without actually executing it.
    
    This function examines the parameters of an improve_property action to determine
    if it would return a failure code based on the rules for property improvements.
    It also considers the current game phase to ensure the action is allowed in that phase.
    
    Args:
        mapping_entry: The action mapping entry containing the action and parameters
        current_gameboard: The current state of the game board
        game_phase: The current phase of the game ("pre_roll", "post_roll", or "out_of_turn")
        
    Returns:
        bool: True if the improve property action would be valid, False otherwise
    """
    # First check if property improvements are allowed in the current game phase
    if game_phase not in ["pre_roll", "out_of_turn"]:
        logger.debug(f"Property improvements are not allowed in the {game_phase} phase")
        return False
    
    # Extract parameters from the mapping entry
    params = mapping_entry["parameters"]
    player = params["player"]
    asset_name = params["asset"]
    
    # Find the property object from its name
    property_obj = None
    for location in current_gameboard.get('location_sequence', []):
        if hasattr(location, 'name') and location.name == asset_name:
            property_obj = location
            break
    
    if property_obj is None:
        logger.debug(f"Property {asset_name} not found in gameboard")
        return False
    
    # Check if the property is a real estate (only real estate can be improved)
    if not hasattr(property_obj, 'loc_class') or property_obj.loc_class != 'real_estate':
        logger.debug(f"Property {asset_name} is not a real estate and cannot be improved")
        return False
    
    # Check if player owns the property
    if not hasattr(property_obj, 'owned_by') or property_obj.owned_by != player:
        logger.debug(f"Player {player.player_name} does not own {asset_name}")
        return False
    
    # Check if the property is mortgaged
    if hasattr(property_obj, 'is_mortgaged') and property_obj.is_mortgaged:
        logger.debug(f"Property {asset_name} is mortgaged and cannot be improved")
        return False
    
    # Check if player owns all properties in the color group (monopoly)
    if not hasattr(property_obj, 'color'):
        logger.debug(f"Property {asset_name} does not have a color attribute")
        return False
    
    color = property_obj.color
    if color not in current_gameboard.get('color_assets', {}):
        logger.debug(f"Color {color} not found in color_assets")
        return False
    
    # Check if player has a monopoly on this color
    has_monopoly = True
    for prop in current_gameboard['color_assets'][color]:
        if not hasattr(prop, 'owned_by') or prop.owned_by != player:
            has_monopoly = False
            break
    
    if not has_monopoly:
        logger.debug(f"Player {player.player_name} does not have a monopoly on {color}")
        return False
    
    # Check if all properties in the color group are unmortgaged
    all_unmortgaged = True
    for prop in current_gameboard['color_assets'][color]:
        if hasattr(prop, 'is_mortgaged') and prop.is_mortgaged:
            all_unmortgaged = False
            break
    
    if not all_unmortgaged:
        logger.debug(f"Not all properties in {color} color group are unmortgaged")
        return False
    
    # Check if the property has reached maximum improvements
    if hasattr(property_obj, 'num_houses') and hasattr(property_obj, 'num_hotels'):
        if property_obj.num_hotels == 1:
            logger.debug(f"Property {asset_name} already has a hotel and cannot be improved further")
            return False
        
        if property_obj.num_houses == 4 and property_obj.num_hotels == 0:
            # Check if there are hotels available in the bank
            if current_gameboard.get('bank').total_hotels <= 0:
                logger.debug("No hotels available in the bank")
                return False
        else:
            # Check if there are houses available in the bank
            if current_gameboard.get('bank').total_houses <= 0:
                logger.debug("No houses available in the bank")
                return False
    
    # Check if improvements are balanced across the color group
    if hasattr(property_obj, 'num_houses') and hasattr(property_obj, 'num_hotels'):
        current_improvement_level = property_obj.num_houses
        if property_obj.num_hotels == 1:
            current_improvement_level = 5  # A hotel counts as 5 improvement levels
        
        # Check if this improvement would make the property more than 1 level ahead of others
        for prop in current_gameboard['color_assets'][color]:
            if prop == property_obj:
                continue  # Skip the property we're trying to improve
            
            other_improvement_level = 0
            if hasattr(prop, 'num_houses'):
                other_improvement_level = prop.num_houses
            if hasattr(prop, 'num_hotels') and prop.num_hotels == 1:
                other_improvement_level = 5
            
            if current_improvement_level + 1 > other_improvement_level + 1:
                logger.debug(f"Improving {asset_name} would make it more than 1 level ahead of other properties in the color group")
                return False
    
    # Check if player has enough cash for the improvement
    if hasattr(property_obj, 'house_price'):
        improvement_cost = property_obj.house_price
        if player.current_cash < improvement_cost:
            logger.debug(f"Player {player.player_name} does not have enough cash ({player.current_cash}) for improvement cost {improvement_cost}")
            return False
    else:
        logger.debug(f"Property {asset_name} does not have a house_price attribute")
        return False
    
    # If all checks pass, the improve property action is valid
    return True

def check_sell_house_hotel_validity(mapping_entry: Dict, current_gameboard: Dict, game_phase: str) -> bool:
    """
    Checks if a sell_house_hotel action would be valid without actually executing it.
    
    This function examines the parameters of a sell_house_hotel action to determine
    if it would return a failure code based on the rules for selling houses or hotels.
    It also considers the current game phase to ensure the action is allowed in that phase.
    
    Args:
        mapping_entry: The action mapping entry containing the action and parameters
        current_gameboard: The current state of the game board
        game_phase: The current phase of the game ("pre_roll", "post_roll", or "out_of_turn")
        
    Returns:
        bool: True if the sell_house_hotel action would be valid, False otherwise
    """
    # Sell house/hotel is allowed in all phases
    # Extract parameters from the mapping entry
    params = mapping_entry["parameters"]
    player = params["player"]
    asset_name = params["asset"]
    sell_house = params.get("sell_house", False)
    sell_hotel = params.get("sell_hotel", False)
    
    # Validate that exactly one of sell_house or sell_hotel is True
    if not (sell_house ^ sell_hotel):  # XOR operation - exactly one must be True
        logger.debug(f"Invalid sell_house_hotel parameters: sell_house={sell_house}, sell_hotel={sell_hotel}")
        return False
    
    # Find the property object from its name
    property_obj = None
    for location in current_gameboard.get('location_sequence', []):
        if hasattr(location, 'name') and location.name == asset_name:
            property_obj = location
            break
    
    if property_obj is None:
        logger.debug(f"Property {asset_name} not found in gameboard")
        return False
    
    # Check if the property is a real estate (only real estate can have houses/hotels)
    if not hasattr(property_obj, 'loc_class') or property_obj.loc_class != 'real_estate':
        logger.debug(f"Property {asset_name} is not a real estate and cannot have houses/hotels")
        return False
    
    # Check if player owns the property
    if not hasattr(property_obj, 'owned_by') or property_obj.owned_by != player:
        logger.debug(f"Player {player.player_name} does not own {asset_name}")
        return False
    
    # Check if the property has houses/hotels to sell
    if not hasattr(property_obj, 'num_houses') or not hasattr(property_obj, 'num_hotels'):
        logger.debug(f"Property {asset_name} does not have num_houses or num_hotels attributes")
        return False
    
    # Check if selling a house
    if sell_house:
        if property_obj.num_houses <= 0:
            logger.debug(f"Property {asset_name} has no houses to sell")
            return False
        
        # Check if improvements are balanced across the color group
        color = property_obj.color
        if color in current_gameboard.get('color_assets', {}):
            current_improvement_level = property_obj.num_houses
            
            # Check if removing this house would make the property more than 1 level behind others
            for prop in current_gameboard['color_assets'][color]:
                if prop == property_obj:
                    continue  # Skip the property we're trying to sell from
                
                other_improvement_level = 0
                if hasattr(prop, 'num_houses'):
                    other_improvement_level = prop.num_houses
                if hasattr(prop, 'num_hotels') and prop.num_hotels == 1:
                    other_improvement_level = 5
                
                if current_improvement_level - 1 < other_improvement_level - 1:
                    logger.debug(f"Selling a house from {asset_name} would make it more than 1 level behind other properties in the color group")
                    return False
    
    # Check if selling a hotel
    if sell_hotel:
        if property_obj.num_hotels <= 0:
            logger.debug(f"Property {asset_name} has no hotels to sell")
            return False
        
        # Check if there are enough houses in the bank to replace the hotel
        # A hotel is replaced with 4 houses when sold
        if current_gameboard.get('bank').total_houses < 4:
            logger.debug(f"Not enough houses in the bank to replace the hotel on {asset_name}")
            return False
    
    # Calculate the sale price (half of the house/hotel price)
    if hasattr(property_obj, 'house_price'):
        sale_price = property_obj.house_price / 2
        logger.debug(f"Selling a {'hotel' if sell_hotel else 'house'} from {asset_name} would yield ${sale_price}")
    else:
        logger.debug(f"Property {asset_name} does not have a house_price attribute")
        return False
    
    # If all checks pass, the sell_house_hotel action is valid
    return True


def check_sell_property_validity(mapping_entry: Dict, current_gameboard: Dict, game_phase: str) -> bool:
    """
    Checks if a sell_property action would be valid without actually executing it.
    
    This function examines the parameters of a sell_property action to determine
    if it would return a failure code based on the rules for selling properties.
    It also considers the current game phase to ensure the action is allowed in that phase.
    
    Args:
        mapping_entry: The action mapping entry containing the action and parameters
        current_gameboard: The current state of the game board
        game_phase: The current phase of the game ("pre_roll", "post_roll", or "out_of_turn")
        
    Returns:
        bool: True if the sell_property action would be valid, False otherwise
    """
    # Sell property is allowed in all phases (pre_roll, post_roll, out_of_turn)
    
    # Extract parameters from the mapping entry
    params = mapping_entry["parameters"]
    player = params["player"]
    asset_name = params["asset"]
    
    # Find the property object from its name
    property_obj = None
    for location in current_gameboard.get('location_sequence', []):
        if hasattr(location, 'name') and location.name == asset_name:
            property_obj = location
            break
    
    if property_obj is None:
        logger.debug(f"Property {asset_name} not found in gameboard")
        return False
    
    # Check if player owns the property
    if not hasattr(property_obj, 'owned_by') or property_obj.owned_by != player:
        logger.debug(f"Player {player.player_name} does not own {asset_name}")
        return False
    
    # Check if the property is mortgaged
    if hasattr(property_obj, 'is_mortgaged') and property_obj.is_mortgaged:
        logger.debug(f"Property {asset_name} is mortgaged and cannot be sold")
        return False
    
    # Check if the property has improvements
    if property_obj.loc_class == 'real_estate' and hasattr(property_obj, 'num_houses') and hasattr(property_obj, 'num_hotels'):
        if property_obj.num_houses > 0 or property_obj.num_hotels > 0:
            logger.debug(f"Property {asset_name} has improvements and cannot be sold directly")
            return False
    
    # Check if the property is part of a monopoly with improvements
    if property_obj.loc_class == 'real_estate' and hasattr(property_obj, 'color'):
        color = property_obj.color
        if color in current_gameboard.get('color_assets', {}):
            for prop in current_gameboard['color_assets'][color]:
                if (prop != property_obj and 
                    hasattr(prop, 'owned_by') and prop.owned_by == player and 
                    hasattr(prop, 'num_houses') and hasattr(prop, 'num_hotels') and 
                    (prop.num_houses > 0 or prop.num_hotels > 0)):
                    logger.debug(f"Cannot sell {asset_name} because another property in the same color set has improvements")
                    return False
    
    # Calculate the sale price (half of the property price)
    if hasattr(property_obj, 'price'):
        sale_price = property_obj.price / 2
        logger.debug(f"Selling {asset_name} would yield ${sale_price}")
    else:
        logger.debug(f"Property {asset_name} does not have a price attribute")
        return False
    
    # If all checks pass, the sell_property action is valid
    return True

def check_mortgage_property_validity(mapping_entry: Dict, current_gameboard: Dict, game_phase: str) -> bool:
    """
    Checks if a mortgage_property action would be valid without actually executing it.
    
    This function examines the parameters of a mortgage_property action to determine
    if it would return a failure code based on the rules for mortgaging properties.
    It also considers the current game phase to ensure the action is allowed in that phase.
    
    Args:
        mapping_entry: The action mapping entry containing the action and parameters
        current_gameboard: The current state of the game board
        game_phase: The current phase of the game ("pre_roll", "post_roll", or "out_of_turn")
        
    Returns:
        bool: True if the mortgage_property action would be valid, False otherwise
    """
    # Mortgage property is allowed in all phases (pre_roll, post_roll, out_of_turn)
    
    # Extract parameters from the mapping entry
    params = mapping_entry["parameters"]
    player = params["player"]
    asset_name = params["asset"]
    
    # Find the property object from its name
    property_obj = None
    for location in current_gameboard.get('location_sequence', []):
        if hasattr(location, 'name') and location.name == asset_name:
            property_obj = location
            break
    
    if property_obj is None:
        logger.debug(f"Property {asset_name} not found in gameboard")
        return False
    
    # Check if the property is a real estate, railroad, or utility (only these can be mortgaged)
    if not hasattr(property_obj, 'loc_class') or property_obj.loc_class not in ['real_estate', 'railroad', 'utility']:
        logger.debug(f"Property {asset_name} is not a real estate, railroad, or utility and cannot be mortgaged")
        return False
    
    # Check if player owns the property
    if not hasattr(property_obj, 'owned_by') or property_obj.owned_by != player:
        logger.debug(f"Player {player.player_name} does not own {asset_name}")
        return False
    
    # Check if the property is already mortgaged
    if hasattr(property_obj, 'is_mortgaged') and property_obj.is_mortgaged:
        logger.debug(f"Property {asset_name} is already mortgaged")
        return False
    
    # Check if the property has improvements (for real estate)
    if property_obj.loc_class == 'real_estate' and hasattr(property_obj, 'num_houses') and hasattr(property_obj, 'num_hotels'):
        if property_obj.num_houses > 0 or property_obj.num_hotels > 0:
            logger.debug(f"Property {asset_name} has improvements and cannot be mortgaged")
            return False
    
    # Check if any property in the same color group has improvements (for real estate)
    if property_obj.loc_class == 'real_estate' and hasattr(property_obj, 'color'):
        color = property_obj.color
        if color in current_gameboard.get('color_assets', {}):
            for prop in current_gameboard['color_assets'][color]:
                if (prop != property_obj and 
                    hasattr(prop, 'owned_by') and prop.owned_by == player and 
                    hasattr(prop, 'num_houses') and hasattr(prop, 'num_hotels') and 
                    (prop.num_houses > 0 or prop.num_hotels > 0)):
                    logger.debug(f"Cannot mortgage {asset_name} because another property in the same color set has improvements")
                    return False
    
    # Calculate the mortgage value
    if hasattr(property_obj, 'price'):
        mortgage_value = property_obj.price / 2
        logger.debug(f"Mortgaging {asset_name} would yield ${mortgage_value}")
    else:
        logger.debug(f"Property {asset_name} does not have a price attribute")
        return False
    
    # If all checks pass, the mortgage_property action is valid
    return True

def check_free_mortgage_validity(mapping_entry: Dict, current_gameboard: Dict, game_phase: str) -> bool:
    """
    Checks if a free_mortgage action would be valid without actually executing it.
    
    This function examines the parameters of a free_mortgage action to determine
    if it would return a failure code based on the rules for unmortgaging properties.
    It also considers the current game phase to ensure the action is allowed in that phase.
    
    Args:
        mapping_entry: The action mapping entry containing the action and parameters
        current_gameboard: The current state of the game board
        game_phase: The current phase of the game ("pre_roll", "post_roll", or "out_of_turn")
        
    Returns:
        bool: True if the free_mortgage action would be valid, False otherwise
    """
    # Free mortgage (unmortgage) is allowed in all phases (pre_roll, post_roll, out_of_turn)
    
    # Extract parameters from the mapping entry
    params = mapping_entry["parameters"]
    player = params["player"]
    asset_name = params["asset"]
    
    # Find the property object from its name
    property_obj = None
    for location in current_gameboard.get('location_sequence', []):
        if hasattr(location, 'name') and location.name == asset_name:
            property_obj = location
            break
    
    if property_obj is None:
        logger.debug(f"Property {asset_name} not found in gameboard")
        return False
    
    # Check if the property is a real estate, railroad, or utility (only these can be mortgaged/unmortgaged)
    if not hasattr(property_obj, 'loc_class') or property_obj.loc_class not in ['real_estate', 'railroad', 'utility']:
        logger.debug(f"Property {asset_name} is not a real estate, railroad, or utility and cannot be unmortgaged")
        return False
    
    # Check if player owns the property
    if not hasattr(property_obj, 'owned_by') or property_obj.owned_by != player:
        logger.debug(f"Player {player.player_name} does not own {asset_name}")
        return False
    
    # Check if the property is actually mortgaged
    if not hasattr(property_obj, 'is_mortgaged') or not property_obj.is_mortgaged:
        logger.debug(f"Property {asset_name} is not mortgaged")
        return False
    
    # Calculate the unmortgage cost (mortgage value + 10% interest)
    if hasattr(property_obj, 'price'):
        mortgage_value = property_obj.price / 2
        unmortgage_cost = mortgage_value * 1.1  # 10% interest
        logger.debug(f"Unmortgaging {asset_name} would cost ${unmortgage_cost}")
        
        # Check if player has enough cash to unmortgage
        if player.current_cash < unmortgage_cost:
            logger.debug(f"Player {player.player_name} does not have enough cash ({player.current_cash}) to unmortgage {asset_name} (cost: {unmortgage_cost})")
            return False
    else:
        logger.debug(f"Property {asset_name} does not have a price attribute")
        return False
    
    # If all checks pass, the free_mortgage action is valid
    return True

def check_pay_jail_fine_validity(mapping_entry: Dict, current_gameboard: Dict, game_phase: str) -> bool:
    """
    Checks if a pay_jail_fine action would be valid without actually executing it.
    
    This function examines the parameters of a pay_jail_fine action to determine
    if it would return a failure code based on the rules for paying jail fines.
    It also considers the current game phase to ensure the action is allowed in that phase.
    
    Args:
        mapping_entry: The action mapping entry containing the action and parameters
        current_gameboard: The current state of the game board
        game_phase: The current phase of the game ("pre_roll", "post_roll", or "out_of_turn")
        
    Returns:
        bool: True if the pay_jail_fine action would be valid, False otherwise
    """
    # Pay jail fine is only allowed in the pre_roll phase
    if game_phase != "pre_roll":
        logger.debug(f"Pay jail fine is not allowed in the {game_phase} phase")
        return False
    
    # Extract parameters from the mapping entry
    params = mapping_entry["parameters"]
    player = params["player"]
    
    # Check if player is actually in jail
    if not hasattr(player, 'currently_in_jail') or not player.currently_in_jail:
        logger.debug(f"Player {player.player_name} is not in jail")
        return False
    
    # Check if player has enough cash to pay the fine (usually $50)
    jail_fine = 50  # Standard jail fine in Monopoly
    if player.current_cash < jail_fine:
        logger.debug(f"Player {player.player_name} does not have enough cash ({player.current_cash}) to pay the jail fine ({jail_fine})")
        return False
    
    # If all checks pass, the pay_jail_fine action is valid
    return True

def check_use_get_out_of_jail_card_validity(mapping_entry: Dict, current_gameboard: Dict, game_phase: str) -> bool:
    """
    Checks if a use_get_out_of_jail_card action would be valid without actually executing it.
    
    This function examines the parameters of a use_get_out_of_jail_card action to determine
    if it would return a failure code based on the rules for using get out of jail cards.
    It also considers the current game phase to ensure the action is allowed in that phase.
    
    Args:
        mapping_entry: The action mapping entry containing the action and parameters
        current_gameboard: The current state of the game board
        game_phase: The current phase of the game ("pre_roll", "post_roll", or "out_of_turn")
        
    Returns:
        bool: True if the use_get_out_of_jail_card action would be valid, False otherwise
    """
    # Use get out of jail card is only allowed in the pre_roll phase
    if game_phase != "pre_roll":
        logger.debug(f"Use get out of jail card is not allowed in the {game_phase} phase")
        return False
    
    # Extract parameters from the mapping entry
    params = mapping_entry["parameters"]
    player = params["player"]
    
    # Check if player is actually in jail
    if not hasattr(player, 'currently_in_jail') or not player.currently_in_jail:
        logger.debug(f"Player {player.player_name} is not in jail")
        return False
    
    # Check if player has a get out of jail card
    has_chance_card = hasattr(player, 'has_get_out_of_jail_chance_card') and player.has_get_out_of_jail_chance_card
    has_cc_card = hasattr(player, 'has_get_out_of_jail_community_chest_card') and player.has_get_out_of_jail_community_chest_card
    
    if not (has_chance_card or has_cc_card):
        logger.debug(f"Player {player.player_name} does not have any get out of jail cards")
        return False
    
    # If all checks pass, the use_get_out_of_jail_card action is valid
    return True

def check_accept_trade_offer_validity(mapping_entry: Dict, current_gameboard: Dict, game_phase: str) -> bool:
    """
    Checks if an accept_trade_offer action would be valid without actually executing it.
    
    This function examines the parameters of an accept_trade_offer action to determine
    if it would return a failure code based on the rules for accepting trade offers.
    It also considers the current game phase to ensure the action is allowed in that phase.
    
    Args:
        mapping_entry: The action mapping entry containing the action and parameters
        current_gameboard: The current state of the game board
        game_phase: The current phase of the game ("pre_roll", "post_roll", or "out_of_turn")
        
    Returns:
        bool: True if the accept_trade_offer action would be valid, False otherwise
    """
    # Accept trade offer is allowed in pre_roll and out_of_turn phases
    if game_phase not in ["pre_roll", "out_of_turn"]:
        logger.debug(f"Accept trade offer is not allowed in the {game_phase} phase")
        return False
    
    # Extract parameters from the mapping entry
    params = mapping_entry["parameters"]
    player = params["player"]
    
    # Check if player has an outstanding trade offer
    if not hasattr(player, 'is_trade_offer_outstanding') or not player.is_trade_offer_outstanding:
        logger.debug(f"Player {player.player_name} does not have any outstanding trade offers")
        return False
    
    # Check if the trade offer is valid
    if not hasattr(player, 'outstanding_trade_offer'):
        logger.debug(f"Player {player.player_name} has is_trade_offer_outstanding=True but no outstanding_trade_offer object")
        return False
    
    trade_offer = player.outstanding_trade_offer
    
    # Check if the from_player still exists and is active
    if not hasattr(trade_offer, 'from_player') or trade_offer.from_player.status == 'lost':
        logger.debug(f"The player who made the trade offer no longer exists or has lost the game")
        return False
    
    # Check if player has enough cash for the cash_wanted
    if hasattr(trade_offer, 'cash_wanted') and player.current_cash < trade_offer.cash_wanted:
        logger.debug(f"Player {player.player_name} does not have enough cash ({player.current_cash}) for the trade (requires {trade_offer.cash_wanted})")
        return False
    
    # Check if from_player has enough cash for the cash_offered
    if hasattr(trade_offer, 'cash_offered') and hasattr(trade_offer, 'from_player') and trade_offer.from_player.current_cash < trade_offer.cash_offered:
        logger.debug(f"From player {trade_offer.from_player.player_name} does not have enough cash ({trade_offer.from_player.current_cash}) for the trade (requires {trade_offer.cash_offered})")
        return False
    
    # Check if all properties in the offer still exist and are owned by the correct players
    if hasattr(trade_offer, 'property_set_offered'):
        for prop in trade_offer.property_set_offered:
            if not hasattr(prop, 'owned_by') or prop.owned_by != trade_offer.from_player:
                logger.debug(f"Property {prop.name if hasattr(prop, 'name') else 'unknown'} is no longer owned by the offering player")
                return False
            
            # Check if the property has been improved or mortgaged since the offer was made
            if prop.loc_class == 'real_estate' and hasattr(prop, 'num_houses') and hasattr(prop, 'num_hotels') and (prop.num_houses > 0 or prop.num_hotels > 0):
                logger.debug(f"Property {prop.name} has been improved since the offer was made")
                return False
            
            if hasattr(prop, 'is_mortgaged') and prop.is_mortgaged:
                logger.debug(f"Property {prop.name} has been mortgaged since the offer was made")
                return False
    
    if hasattr(trade_offer, 'property_set_wanted'):
        for prop in trade_offer.property_set_wanted:
            if not hasattr(prop, 'owned_by') or prop.owned_by != player:
                logger.debug(f"Property {prop.name if hasattr(prop, 'name') else 'unknown'} is no longer owned by the receiving player")
                return False
            
            # Check if the property has been improved or mortgaged since the offer was made
            if prop.loc_class == 'real_estate' and hasattr(prop, 'num_houses') and hasattr(prop, 'num_hotels') and (prop.num_houses > 0 or prop.num_hotels > 0):
                logger.debug(f"Property {prop.name} has been improved since the offer was made")
                return False
            
            if hasattr(prop, 'is_mortgaged') and prop.is_mortgaged:
                logger.debug(f"Property {prop.name} has been mortgaged since the offer was made")
                return False
    
    # If all checks pass, the accept_trade_offer action is valid
    return True

def check_skip_turn_validity(mapping_entry: Dict, current_gameboard: Dict, game_phase: str) -> bool:
    """
    Checks if a skip_turn action would be valid without actually executing it.
    
    This function examines if skipping a turn is allowed in the current game phase.
    
    Args:
        mapping_entry: The action mapping entry containing the action and parameters
        current_gameboard: The current state of the game board
        game_phase: The current phase of the game ("pre_roll", "post_roll", or "out_of_turn")
        
    Returns:
        bool: True if the skip_turn action would be valid, False otherwise
    """
    # Skip turn is allowed in all phases
    # No specific conditions to check for skip_turn
    return True

def check_concluded_actions_validity(mapping_entry: Dict, current_gameboard: Dict, game_phase: str) -> bool:
    """
    Checks if a concluded_actions action would be valid without actually executing it.
    
    This function examines if concluding actions is allowed in the current game phase.
    
    Args:
        mapping_entry: The action mapping entry containing the action and parameters
        current_gameboard: The current state of the game board
        game_phase: The current phase of the game ("pre_roll", "post_roll", or "out_of_turn")
        
    Returns:
        bool: True if the concluded_actions action would be valid, False otherwise
    """
    # Concluded actions is allowed in all phases
    # No specific conditions to check for concluded_actions
    return True

def check_buy_property_validity(mapping_entry: Dict, current_gameboard: Dict, game_phase: str) -> bool:
    """
    Checks if a buy_property action would be valid without actually executing it.
    
    This function examines if the player is on an unowned property that can be purchased.
    It also considers the current game phase to ensure the action is allowed in that phase.
    
    Args:
        mapping_entry: The action mapping entry containing the action and parameters
        current_gameboard: The current state of the game board
        game_phase: The current phase of the game ("pre_roll", "post_roll", or "out_of_turn")
        
    Returns:
        bool: True if the buy_property action would be valid, False otherwise
    """
    # Buy property is only allowed in the post_roll phase
    if game_phase != "post_roll":
        logger.debug(f"Buy property is not allowed in the {game_phase} phase")
        return False
    
    # Extract parameters from the mapping entry
    params = mapping_entry.get("parameters", {})
    player = params.get("player")
    
    if not player:
        logger.debug("No player found in parameters")
        return False
    
    # Check if the player has the option to buy property
    if hasattr(player, '_option_to_buy') and player._option_to_buy:
        # Find the property at the player's current position
        current_position = player.current_position
        
        # Find the property at the player's current position
        property_obj = None
        for location in current_gameboard.get('location_sequence', []):
            if hasattr(location, 'start_position') and location.start_position == current_position:
                property_obj = location
                break
        
        if property_obj is None:
            logger.debug(f"No property found at position {current_position}")
            return False
        
        # Check if the property is purchasable
        if not hasattr(property_obj, 'loc_class') or property_obj.loc_class not in ['real_estate', 'railroad', 'utility']:
            logger.debug(f"Property at position {current_position} is not purchasable")
            return False
        
        # Check if the property is already owned by someone other than the bank
        if hasattr(property_obj, 'owned_by') and property_obj.owned_by is not None and property_obj.owned_by != current_gameboard.get('bank'):
            logger.debug(f"Property at position {current_position} is already owned")
            return False
        
        # Check if player has enough cash to buy the property
        if hasattr(property_obj, 'price'):
            if player.current_cash < property_obj.price:
                logger.debug(f"Player {player.player_name} does not have enough cash (${player.current_cash}) to buy property at position {current_position} (price: ${property_obj.price})")
                return False
            
            logger.debug(f"Player {player.player_name} CAN buy property {property_obj.name} at position {current_position} for ${property_obj.price}")
            return True
        else:
            logger.debug(f"Property at position {current_position} does not have a price attribute")
            return False
    
    logger.debug(f"Player {player.player_name} does not have the option to buy property")
    return False

def convert_cash_value(value):
    """
    Converts a cash amount value into a numeric type.
    If the value is numeric, it is simply returned as a float.
    If the value is a recognized string (e.g., 'below_market', 'at_market', 'above_market'),
    then it is mapped to a numeric value. Adjust these mappings as needed.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        mapping = {
            "below_market": 0.75,  # For example, offers below market are treated as 0.0
            "at_market": 1.0,      # At market could be interpreted as a full market offer (modify as necessary)
            "above_market": 1.25    # Above market might imply a premium (modify as necessary)
        }
        try:
            return float(mapping[value])
        except KeyError as e:
            raise ValueError(f"Unrecognized cash value string: {value}") from e
    raise TypeError(f"Invalid type for cash value: {type(value)}")


def check_trade_offer_validity(mapping_entry: dict, current_gameboard: dict, game_phase: str) -> bool:
    """
    Checks if a trade offer action would be valid without actually executing it.
    
    This function examines the parameters of a make_trade_offer action to determine
    if it would return a failure code based on the rules in the make_trade_offer function.
    It also considers the current game phase to ensure the action is allowed in that phase.
    
    Args:
        mapping_entry: The action mapping entry containing the action and parameters.
        current_gameboard: The current state of the game board.
        game_phase: The current phase of the game ("pre_roll", "post_roll", or "out_of_turn").
            
    Returns:
        bool: True if the trade offer would be valid, False otherwise.
    """
    try:
        # First check if trade offers are allowed in the current game phase
        if game_phase not in ["pre_roll", "out_of_turn"]:
            logger.debug(f"Trade offers are not allowed in the {game_phase} phase")
            return False
        
        # Extract parameters from the mapping entry
        params = mapping_entry.get("parameters", {})
        if not params:
            logger.debug("No parameters found in mapping entry")
            return False
            
        from_player = params.get("from_player")
        if not from_player:
            logger.debug("No from_player found in parameters")
            return False
            
        to_player_name = params.get("to_player")
        if not to_player_name:
            logger.debug("No to_player found in parameters")
            return False
            
        offer = params.get("offer", {})
        if not offer:
            logger.debug("No offer found in parameters")
            return False
        
        # Find the to_player object from the name
        to_player = None
        for player in current_gameboard.get("players", []):
            if hasattr(player, "player_name") and player.player_name == to_player_name:
                to_player = player
                break
        
        if to_player is None:
            logger.debug(f"Player {to_player_name} not found in gameboard")
            return False
        
        # Check if the to_player has lost the game
        if hasattr(to_player, "status") and to_player.status == 'lost':
            logger.debug(f"Trade offer is being made to player {to_player_name} who has lost the game")
            return False
        
        # Check if the to_player already has an outstanding trade offer
        if hasattr(to_player, 'is_trade_offer_outstanding') and to_player.is_trade_offer_outstanding:
            logger.debug(f"Player {to_player_name} already has a trade offer outstanding")
            return False
        
        # Convert cash_offered and cash_wanted to numeric types.
        cash_offered_raw = offer.get('cash_offered', 0)
        cash_wanted_raw = offer.get('cash_wanted', 0)
        try:
            cash_offered = int(convert_cash_value(cash_offered_raw))
            cash_wanted = int(convert_cash_value(cash_wanted_raw))
        except (ValueError, TypeError) as e:
            logger.debug(f"Error converting cash amounts: cash_offered='{cash_offered_raw}', cash_wanted='{cash_wanted_raw}'. Error: {e}")
            return False

        # Check if cash offered or wanted is negative
        if cash_offered < 0 or cash_wanted < 0:
            logger.debug("Cash offered or cash wanted amounts cannot be negative")
            return False
        
        # Check if from_player has enough cash for the offer
        if hasattr(from_player, 'current_cash') and from_player.current_cash < cash_offered:
            logger.debug(f"Player {from_player.player_name} does not have enough cash for this offer")
            return False
        
        # Check if to_player has enough cash for what's wanted
        if hasattr(to_player, 'current_cash') and to_player.current_cash < cash_wanted:
            logger.debug(f"Player {to_player_name} does not have enough cash for what's wanted")
            return False
        
        # Check property ownership and conditions for offered properties
        property_set_offered = offer.get('property_set_offered', set())
        if not isinstance(property_set_offered, (set, list, tuple)):
            logger.debug("property_set_offered is not a collection type")
            return False
            
        for prop_name in property_set_offered:
            # Find the property object from its name
            offered_property = None
            for location in current_gameboard.get('location_sequence', []):
                if hasattr(location, 'name') and location.name == prop_name:
                    offered_property = location
                    break
            
            if offered_property is None:
                logger.debug(f"Property {prop_name} not found in gameboard")
                return False
            
            # Check if from_player owns the property
            if not hasattr(offered_property, 'owned_by') or offered_property.owned_by != from_player:
                logger.debug(f"Player {from_player.player_name} does not own {prop_name}")
                return False
            
            # Check if the property has improvements
            if (offered_property.loc_class == 'real_estate' and 
                hasattr(offered_property, 'num_houses') and 
                hasattr(offered_property, 'num_hotels') and 
                (offered_property.num_houses > 0 or offered_property.num_hotels > 0)):
                logger.debug(f"Property {prop_name} has improvements")
                return False
            
            # Check if the property is mortgaged
            if ((offered_property.loc_class in ['real_estate', 'railroad', 'utility']) and 
                hasattr(offered_property, 'is_mortgaged') and 
                offered_property.is_mortgaged):
                logger.debug(f"Property {prop_name} is mortgaged")
                return False
        
        # Check property ownership and conditions for wanted properties
        property_set_wanted = offer.get('property_set_wanted', set())
        if not isinstance(property_set_wanted, (set, list, tuple)):
            logger.debug("property_set_wanted is not a collection type")
            return False
            
        for prop_name in property_set_wanted:
            # Find the property object from its name
            wanted_property = None
            for location in current_gameboard.get('location_sequence', []):
                if hasattr(location, 'name') and location.name == prop_name:
                    wanted_property = location
                    break
            
            if wanted_property is None:
                logger.debug(f"Property {prop_name} not found in gameboard")
                return False
            
            # Check if to_player owns the property
            if not hasattr(wanted_property, 'owned_by') or wanted_property.owned_by != to_player:
                logger.debug(f"Player {to_player_name} does not own {prop_name}")
                return False
            
            # Check if the property has improvements
            if (wanted_property.loc_class == 'real_estate' and 
                hasattr(wanted_property, 'num_houses') and 
                hasattr(wanted_property, 'num_hotels') and 
                (wanted_property.num_houses > 0 or wanted_property.num_hotels > 0)):
                logger.debug(f"Property {prop_name} has improvements")
                return False
            
            # Check if the property is mortgaged
            if ((wanted_property.loc_class in ['real_estate', 'railroad', 'utility']) and 
                hasattr(wanted_property, 'is_mortgaged') and 
                wanted_property.is_mortgaged):
                logger.debug(f"Property {prop_name} is mortgaged")
                return False
        
        # If all checks pass, the trade offer is valid
        return True
        
    except Exception as e:
        logger.error(f"Error in check_trade_offer_validity: {str(e)}")
        return False


def create_action_mask(player, current_gameboard, game_phase: str) -> np.ndarray:
    """
    Creates a boolean mask for the entire action space based on the current game state.
    
    This function builds the complete action mapping and then checks each action's validity
    using the appropriate validity check function. The result is a boolean mask where
    True (1) indicates a valid action and False (0) indicates an invalid action.
    
    Args:
        player: The player object for whom to create the action mask
        current_gameboard: The current state of the game board
        game_phase: The current phase of the game ("pre_roll", "post_roll", or "out_of_turn")
    
    Returns:
        A numpy boolean array of size 2922 where True (1) indicates a valid action
        and False (0) indicates an invalid action.
    """
    rl_logger.info(f"Creating action mask for player {player.player_name} in {game_phase} phase")
    
    # Build the full action mapping
    action_encoder = ActionEncoder()
    full_mapping = action_encoder.build_full_action_mapping(player, current_gameboard)
    
    # Initialize the mask with all False values
    action_mask = np.zeros(len(full_mapping), dtype=bool)
    
    # Dictionary mapping action names to their validity check functions
    validity_check_functions = {
        "make_trade_offer": check_trade_offer_validity,
        "make_sell_property_offer": check_sell_property_offer_validity,
        "improve_property": check_improve_property_validity,
        "sell_house_hotel": check_sell_house_hotel_validity,
        "sell_property": check_sell_property_validity,
        "mortgage_property": check_mortgage_property_validity,
        "free_mortgage": check_free_mortgage_validity,
        "pay_jail_fine": check_pay_jail_fine_validity,
        "use_get_out_of_jail_card": check_use_get_out_of_jail_card_validity,
        "accept_trade_offer": check_accept_trade_offer_validity,
        "skip_turn": check_skip_turn_validity,
        "concluded_actions": check_concluded_actions_validity,
        "buy_property": check_buy_property_validity
    }
    
    # Count valid actions by type
    valid_action_counts = {}
    
    # Iterate through the full mapping and check each action's validity
    valid_actions_count = 0
    for idx, mapping_entry in enumerate(full_mapping):
        action_name = mapping_entry["action"]
        
        # Get the appropriate validity check function
        check_function = validity_check_functions.get(action_name)
        
        if check_function:
            # Check if the action is valid
            is_valid = check_function(mapping_entry, current_gameboard, game_phase)
            action_mask[idx] = is_valid
            
            if is_valid:
                valid_actions_count += 1
                # Count valid actions by type
                valid_action_counts[action_name] = valid_action_counts.get(action_name, 0) + 1
                
                # Log detailed information for valid actions
                if action_name not in ["skip_turn", "concluded_actions"]:
                    params = mapping_entry.get("parameters", {})
                    param_str = ""
                    if "asset" in params:
                        param_str += f", asset: {params['asset']}"
                    if "to_player" in params:
                        param_str += f", to_player: {params['to_player']}"
                    rl_logger.info(f"Valid action at index {idx}: {action_name}{param_str}")
        else:
            rl_logger.warning(f"No validity check function found for action: {action_name}")
    
    # Log statistics about the mask
    rl_logger.info(f"Created action mask with {valid_actions_count} valid actions out of {len(action_mask)} total actions")
    rl_logger.info(f"Valid action counts by type: {valid_action_counts}")
    
    # Log player's properties and cash
    rl_logger.info(f"Player {player.player_name} has ${player.current_cash} cash")
    if hasattr(player, 'assets'):
        rl_logger.info(f"Player {player.player_name} has {len(player.assets)} assets")
        for asset in player.assets:
            mortgage_status = "mortgaged" if hasattr(asset, 'is_mortgaged') and asset.is_mortgaged else "not mortgaged"
            improvements = ""
            if hasattr(asset, 'num_houses') and hasattr(asset, 'num_hotels'):
                improvements = f", houses: {asset.num_houses}, hotels: {asset.num_hotels}"
            rl_logger.info(f"  - {asset.name} ({mortgage_status}{improvements})")
    
    # Log valid action indices
    valid_indices = np.where(action_mask)[0]
    rl_logger.info(f"Valid action indices: {valid_indices}")
    
    # Log buy property validity specifically for debugging
    if game_phase == "post_roll" and hasattr(player, '_option_to_buy'):
        rl_logger.info(f"Player {player.player_name} has _option_to_buy: {player._option_to_buy}")
        if player._option_to_buy:
            current_position = player.current_position
            for location in current_gameboard.get('location_sequence', []):
                if hasattr(location, 'start_position') and location.start_position == current_position:
                    property_obj = location
                    rl_logger.info(f"Property at position {current_position}: {property_obj.name if hasattr(property_obj, 'name') else 'unknown'}")
                    rl_logger.info(f"Property owned by: {property_obj.owned_by.player_name if hasattr(property_obj, 'owned_by') and hasattr(property_obj.owned_by, 'player_name') else 'bank/none'}")
                    rl_logger.info(f"Player cash: ${player.current_cash}, Property price: ${property_obj.price if hasattr(property_obj, 'price') else 'unknown'}")
                    break
    
    # Log a sample of valid actions (up to 10)
    sample_size = min(10, len(valid_indices))
    if sample_size > 0:
        sample_indices = np.random.choice(valid_indices, sample_size, replace=False)
        rl_logger.info("Sample of valid actions:")
        for idx in sample_indices:
            action_name = full_mapping[idx]["action"]
            params = full_mapping[idx].get("parameters", {})
            param_str = ""
            if "asset" in params:
                param_str += f", asset: {params['asset']}"
            if "to_player" in params:
                param_str += f", to_player: {params['to_player']}"
            rl_logger.info(f"  - Index {idx}: {action_name}{param_str}")
    
    return action_mask