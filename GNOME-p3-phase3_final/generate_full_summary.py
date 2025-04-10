#!/usr/bin/env python
"""
Script to generate a full summary of a replay buffer PKL file.
"""
import os
import sys
import pickle
import json
import numpy as np

# Add the parent directory to the path so we can import the replay buffer module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from monopoly_simulator.replay_buffer_module import ReplayBuffer

def numpy_to_serializable(obj):
    """Convert numpy arrays and data types to serializable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    else:
        return obj

def generate_full_summary(pkl_path):
    """Generate a full summary JSON file from a replay buffer PKL file."""
    # Load the pickle file
    with open(pkl_path, 'rb') as f:
        try:
            data = pickle.load(f)
        except Exception as e:
            print(f"Error loading pickle file: {e}")
            # Try as a plain buffer
            f.seek(0)
            buffer = pickle.load(f)
            data = {
                'buffer': buffer,
                'capacity': 100000,  # Default capacity
                'episode_rewards': []
            }
    
    # Create a summary dictionary
    summary = {
        'size': len(data['buffer']) if 'buffer' in data else len(data),
        'capacity': data.get('capacity', 100000),
        'utilization': len(data['buffer']) / data['capacity'] if 'buffer' in data and 'capacity' in data and data['capacity'] > 0 else 0,
        'episode_rewards': data.get('episode_rewards', []),
        'avg_reward': 0
    }
    
    # Get the buffer
    buffer = data['buffer'] if 'buffer' in data else data
    
    # Calculate average reward
    if len(buffer) > 0:
        rewards = [exp[2] for exp in buffer]
        summary['avg_reward'] = sum(rewards) / len(rewards)
    
    # Include ALL entries with COMPLETE state and next_state arrays
    all_entries = []
    for i, (s, a, r, ns, d) in enumerate(buffer):
        entry = {
            'index': i,
            'state': numpy_to_serializable(s),  # Full state array
            'action': numpy_to_serializable(a),
            'reward': numpy_to_serializable(r),
            'next_state': numpy_to_serializable(ns),  # Full next_state array
            'done': numpy_to_serializable(d)
        }
        all_entries.append(entry)
    
    summary['all_entries'] = all_entries
    
    # Save the summary to a JSON file
    summary_path = pkl_path + '.full_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Full summary saved to {summary_path}")
    return summary_path

def generate_compact_summary(pkl_path):
    """Generate a compact summary JSON file from a replay buffer PKL file.
    This version doesn't include the full state arrays to save space."""
    # Load the pickle file
    with open(pkl_path, 'rb') as f:
        try:
            data = pickle.load(f)
        except Exception as e:
            print(f"Error loading pickle file: {e}")
            # Try as a plain buffer
            f.seek(0)
            buffer = pickle.load(f)
            data = {
                'buffer': buffer,
                'capacity': 100000,  # Default capacity
                'episode_rewards': []
            }
    
    # Create a summary dictionary
    summary = {
        'size': len(data['buffer']) if 'buffer' in data else len(data),
        'capacity': data.get('capacity', 100000),
        'utilization': len(data['buffer']) / data['capacity'] if 'buffer' in data and 'capacity' in data and data['capacity'] > 0 else 0,
        'episode_rewards': data.get('episode_rewards', []),
        'avg_reward': 0
    }
    
    # Get the buffer
    buffer = data['buffer'] if 'buffer' in data else data
    
    # Calculate average reward
    if len(buffer) > 0:
        rewards = [exp[2] for exp in buffer]
        summary['avg_reward'] = sum(rewards) / len(rewards)
    
    # Include ALL entries but with summarized state and next_state
    all_entries = []
    for i, (s, a, r, ns, d) in enumerate(buffer):
        entry = {
            'index': i,
            'state_summary': f"Array of shape {s.shape if hasattr(s, 'shape') else 'unknown'}, mean: {np.mean(s) if hasattr(s, 'mean') else 'unknown'}",
            'action': numpy_to_serializable(a),
            'reward': numpy_to_serializable(r),
            'next_state_summary': f"Array of shape {ns.shape if hasattr(ns, 'shape') else 'unknown'}, mean: {np.mean(ns) if hasattr(ns, 'mean') else 'unknown'}",
            'done': numpy_to_serializable(d)
        }
        all_entries.append(entry)
    
    summary['all_entries'] = all_entries
    
    # Save the summary to a JSON file
    summary_path = pkl_path + '.summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Compact summary saved to {summary_path}")
    return summary_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_full_summary.py <path_to_pkl_file> [--compact]")
        print("  --compact: Generate a compact summary without full state arrays")
        sys.exit(1)
    
    pkl_path = sys.argv[1]
    if not os.path.exists(pkl_path):
        print(f"Error: File {pkl_path} does not exist")
        sys.exit(1)
    
    # Check if compact flag is provided
    compact = "--compact" in sys.argv
    
    if compact:
        summary_path = generate_compact_summary(pkl_path)
        print(f"Generated compact summary at: {summary_path}")
    else:
        summary_path = generate_full_summary(pkl_path)
        print(f"Generated full summary at: {summary_path}")
        print("Warning: The full summary may be very large due to including complete state arrays.")
