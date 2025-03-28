# other_actions_mapping.py
# This file creates a flat mapping list for actions with their required parameters.
# The actions include: pay_jail_fine, use_get_out_of_jail_card, accept_trade_offer, skip_turn, concluded_actions.
# Each entry in the returned list is a dictionary with the structure:
#    {
#      "action": <action_name>,
#      "parameters": { expected parameters }
#    }
# Total number of entries: 5.

def build_other_actions_mapping(player, current_gameboard):
    """
    Build a flat mapping list for actions with their required parameters.
    
    The actions and their runtime parameters:
      - "pay_jail_fine": {"player": acting player, "current_gameboard": current game board}
      - "use_get_out_of_jail_card": {"player": acting player, "current_gameboard": current game board}
      - "accept_trade_offer": {"player": acting player, "current_gameboard": current game board}
      - "skip_turn": {}
      - "concluded_actions": {}
      
    Args:
      player: The acting player instance.
      current_gameboard: The current game board dict.
      
    Returns:
         A list of 5 mapping dictionaries.
    """
    actions = [
        {
            "action": "pay_jail_fine",
            "parameters": {"player": player, "current_gameboard": current_gameboard}
        },
        {
            "action": "use_get_out_of_jail_card",
            "parameters": {"player": player, "current_gameboard": current_gameboard}
        },
        {
            "action": "accept_trade_offer",
            "parameters": {"player": player, "current_gameboard": current_gameboard}
        },
        {
            "action": "skip_turn",
            "parameters": {}
        },
        {
            "action": "concluded_actions",
            "parameters": {}
        }
    ]
    
    return actions