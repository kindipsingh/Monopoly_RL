import numpy as np
import torch
import json
import os
import sys
import numpy as np
import torch
from tabulate import tabulate

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monopoly_simulator.location import Location
from monopoly_simulator.player import Player
from monopoly_simulator.bank import Bank

class MonopolyStateVisualizer:
    def __init__(self, encoder):
        self.encoder = encoder
        
    def visualize_state(self, encoded_state):
        """Visualize the encoded state in a readable matrix format"""
        # Convert to numpy for easier manipulation
        if isinstance(encoded_state, torch.Tensor):
            state = encoded_state.squeeze().numpy()
        else:
            state = encoded_state.squeeze()
            
        # Player state visualization (16 dimensions: 4 players × 4 features)
        player_features = ['Position', 'Cash', 'In Jail', 'Has Card']
        player_matrix = state[:16].reshape(4, 4)
        
        # Property state visualization (224 dimensions: 28 properties × 8 features)
        property_features = ['Owner1', 'Owner2', 'Owner3', 'Owner4', 
                           'Mortgaged', 'Monopoly', 'Houses', 'Hotels']
        property_matrix = state[16:].reshape(28, 8)
        
        # Format player states
        print("\n=== Player States ===")
        player_table = tabulate(
            player_matrix,
            headers=player_features,
            showindex=[f"Player {i+1}" for i in range(4)],
            floatfmt=".3f",
            tablefmt="grid"
        )
        print(player_table)
        
        # Format property states
        print("\n=== Property States ===")
        property_table = tabulate(
            property_matrix,
            headers=property_features,
            showindex=[f"Property {i+1}" for i in range(28)],
            floatfmt=".3f",
            tablefmt="grid"
        )
        print(property_table)
        
    def decode_state(self, encoded_state):
        """Decode and print human-readable state information"""
        print("\n=== Decoded State Information ===")
        
        # Decode player information
        print("\nPlayer Information:")
        for i in range(4):
            position = self.encoder.decode_position(encoded_state, i)
            cash = int(self.encoder.decode_cash(encoded_state, i))
            print(f"\nPlayer {i+1}:")
            print(f"  Position: {position}")
            print(f"  Cash: ${cash:,}")
            print(f"  In Jail: {bool(encoded_state[0, i*4 + 2].item())}")
            print(f"  Has Get Out of Jail Card: {bool(encoded_state[0, i*4 + 3].item())}")
        
        # Decode property information
        print("\nProperty Information:")
        for i in range(28):
            owner = self.encoder.get_property_owner(encoded_state, i)
            base_idx = 16 + (i * 8)
            print(f"\nProperty {i+1}:")
            print(f"  Owner: {'Bank' if owner is None else f'Player {owner + 1}'}")
            print(f"  Mortgaged: {bool(encoded_state[0, base_idx + 4].item())}")
            print(f"  Monopoly: {bool(encoded_state[0, base_idx + 5].item())}")
            print(f"  Houses: {int(encoded_state[0, base_idx + 6].item() * 4)}")
            print(f"  Hotels: {int(encoded_state[0, base_idx + 7].item() * self.encoder.max_hotels)}")

def demo_visualization():
    """Demo the visualizer with a sample state"""
    
    # Create encoder and initial state
    encoder = MonopolyStateEncoder()
    initial_state = create_initial_state()
    
    # Create some sample data
    sample_state = initial_state.clone()
    
    # Set some sample values
    # Player 1: Position 5, $1500, not in jail, has card
    sample_state[0, 0:4] = torch.tensor([5/40, np.log(1500)/10, 0, 1])
    
    # Property 1: Owned by player 2, not mortgaged, monopoly, 2 houses
    sample_state[0, 16:24] = torch.tensor([0, 1, 0, 0, 0, 1, 2/4, 0])
    
    # Create and use visualizer
    visualizer = MonopolyStateVisualizer(encoder)
    visualizer.visualize_state(sample_state)
    visualizer.decode_state(sample_state)

class MonopolyStateEncoder:
    def __init__(self):
        self.state_size = 240  # Total state dimensions
        self.num_players = 4
        self.num_properties = 28
        
        # Dimensions breakdown
        self.player_dims = 16    # 4 players × 4 features
        self.property_dims = 224 # 28 properties × 8 features
        
        # Game constants from Bank class
        self.max_houses = 32   # From Bank.total_houses
        self.max_hotels = 12   # From Bank.total_hotels
        
        # Load game schema
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.schema_path = os.path.join(current_dir, "monopoly_game_schema_v1-2.json")
        
        try:
            with open(self.schema_path, 'r') as f:
                self.game_schema = json.load(f)
                
        except FileNotFoundError:
            print(f"Schema file not found at: {self.schema_path}")
            print("Current directory:", os.getcwd())
            raise
        except json.JSONDecodeError:
            print("Error decoding JSON file")
            with open(self.schema_path, 'r') as f:
                print("File contents:", f.read())
            raise
        
        # Initialize color sets from schema
        self.color_sets = self._initialize_color_sets()
        
    def _initialize_color_sets(self):
        """Initialize color sets from game schema"""
        color_sets = {}
        
        try:
            # Print schema structure for debugging
            print("Schema keys:", self.game_schema.keys())
            
            # Hardcoded color sets as fallback if schema loading fails
            default_color_sets = {
                'Brown': ['Mediterranean Avenue', 'Baltic Avenue'],
                'Light Blue': ['Oriental Avenue', 'Vermont Avenue', 'Connecticut Avenue'],
                'Pink': ['St. Charles Place', 'States Avenue', 'Virginia Avenue'],
                'Orange': ['St. James Place', 'Tennessee Avenue', 'New York Avenue'],
                'Red': ['Kentucky Avenue', 'Indiana Avenue', 'Illinois Avenue'],
                'Yellow': ['Atlantic Avenue', 'Ventnor Avenue', 'Marvin Gardens'],
                'Green': ['Pacific Avenue', 'North Carolina Avenue', 'Pennsylvania Avenue'],
                'Dark Blue': ['Park Place', 'Boardwalk']
            }
            
            # Try to load from schema first
            if 'locations' in self.game_schema:
                location_states = self.game_schema['locations']['location_states']
                for location in location_states:
                    if location['loc_class'] == 'real_estate':
                        color = location['color']
                        if color not in color_sets:
                            color_sets[color] = []
                        color_sets[color].append(location['name'])
                return color_sets
            else:
                print("Warning: Using default color sets")
                return default_color_sets
                
        except KeyError as e:
            print(f"KeyError: {e}")
            print("Warning: Using default color sets")
            return default_color_sets

    def encode_state(self, current_gameboard):
        """Encode the complete Monopoly game state into a 240-dimensional vector"""
        state = np.zeros(self.state_size)
        
        # 1. Encode Player States (first 16 dimensions)
        self._encode_players(state, current_gameboard['players'])
        
        # 2. Encode Property States (remaining 224 dimensions)
        self._encode_properties(state, current_gameboard['location_sequence'], 
                              current_gameboard['players'])
        
        return torch.FloatTensor(state).unsqueeze(0)

    def _encode_players(self, state, players):
        """Encode player states into first 16 dimensions"""
        for p_idx, player in enumerate(players):
            base_idx = p_idx * 4  # 4 features per player
            
            # 1. Current location (normalized by board size)
            state[base_idx] = player.current_position / 40
            
            # 2. Cash amount (log-normalized)
            state[base_idx + 1] = np.log(max(1, player.current_cash)) / 10
            
            # 3. Jail status
            state[base_idx + 2] = float(player.currently_in_jail)
            
            # 4. Get out of jail free card (either card type)
            has_card = (player.has_get_out_of_jail_community_chest_card or 
                       player.has_get_out_of_jail_chance_card)
            state[base_idx + 3] = float(has_card)

    def _encode_properties(self, state, location_sequence, players):
        """Encode property states into remaining 224 dimensions"""
        property_idx = 0
        player_base = 16  # Start after player encodings
        
        for location in location_sequence:
            if location.loc_class in ['real_estate', 'railroad', 'utility']:
                base_idx = player_base + (property_idx * 8)
                
                # 1. Owner representation (4-dim one-hot)
                if location.owned_by is not None:
                    for p_idx, player in enumerate(players):
                        if location in player.assets:
                            state[base_idx + p_idx] = 1
                            break
                
                # 2. Mortgaged flag
                state[base_idx + 4] = float(location.is_mortgaged)
                
                # 3. Monopoly flag
                if location.loc_class == 'real_estate':
                    state[base_idx + 5] = float(self._is_monopoly(location, players))
                
                # 4. Houses and Hotels fractions
                if location.loc_class == 'real_estate':
                    if location.num_houses < 5:  # Regular houses
                        state[base_idx + 6] = location.num_houses / 4  # Max 4 houses
                        state[base_idx + 7] = 0  # No hotels
                    else:
                        state[base_idx + 6] = 0  # No houses when hotel present
                        state[base_idx + 7] = location.num_hotels / self.max_hotels
                else:
                    # For railroads and utilities
                    state[base_idx + 6] = 0
                    state[base_idx + 7] = 0
                
                property_idx += 1

    def _is_monopoly(self, property, players):
        """Check if property is part of a monopoly set"""
        if property.owned_by is None:
            return False
            
        # Find property owner
        owner = None
        for player in players:
            if property in player.assets:
                owner = player
                break
                
        if owner is None:
            return False
            
        # Check if color is in player's full_color_sets_possessed
        return property.color in owner.full_color_sets_possessed

    def decode_position(self, encoded_state, player_idx):
        """Decode player position from encoded state"""
        pos_idx = player_idx * 4
        return int(encoded_state[0, pos_idx].item() * 40)

    def decode_cash(self, encoded_state, player_idx):
        """Decode player cash from encoded state"""
        cash_idx = player_idx * 4 + 1
        return np.exp(encoded_state[0, cash_idx].item() * 10)

    def get_property_owner(self, encoded_state, property_idx):
        """Get property owner from encoded state"""
        base_idx = 16 + (property_idx * 8)
        owner_encoding = encoded_state[0, base_idx:base_idx + 4]
        if torch.sum(owner_encoding) == 0:
            return None  # Bank owns property
        return torch.argmax(owner_encoding).item()

def create_initial_state():
    """Create initial state with all zeros"""
    encoder = MonopolyStateEncoder()
    return torch.zeros((1, encoder.state_size))

def main():
   # Initialize encoder and create a state
    encoder = MonopolyStateEncoder()
    state = create_initial_state()
    
    # Create visualizer
    visualizer = MonopolyStateVisualizer(encoder)
    
    # Visualize the state
    visualizer.visualize_state(state)
    visualizer.decode_state(state)

if __name__ == "__main__":
    demo_visualization()