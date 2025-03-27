# other_actions_mapping.py
# This file creates a flat mapping list for actions that have a dimension of 1 and take no parameters.
# The actions include: skip_turn, conclude_actions, use_get_out_of_jail_card, pay_jail_fine, and accept_trade_offer.

def build_other_actions_mapping():
    """
    Build a flat mapping list for actions with no parameters.
    
    The actions included are:
      - "skip_turn"
      - "conclude_actions"
      - "use_get_out_of_jail_card"
      - "pay_jail_fine"
      - "accept_trade_offer"
      
    Each entry in the returned list is a dictionary with the structure:
       {
         "action": <action_name>,
         "parameters": {}
       }
    Total number of entries: 5.
    
    Returns:
         A list of 5 mapping dictionaries.
    """
    actions = [
        "skip_turn",
        "conclude_actions",
        "use_get_out_of_jail_card",
        "pay_jail_fine",
        "accept_trade_offer"
    ]
    
    mapping_list = [{
        "action": action,
        "parameters": {}
    } for action in actions]
    
    return mapping_list

# Example usage:
if __name__ == "__main__":
    mapping_list = build_other_actions_mapping()
    print("Total number of entries:", len(mapping_list))
    print("Mapping entries:")
    for entry in mapping_list:
        print(entry)