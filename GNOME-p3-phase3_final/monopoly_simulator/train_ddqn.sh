#!/bin/bash

# Set the Python path to include the project directory
export PYTHONPATH=${PWD}/../:${PYTHONPATH}

# Create necessary directories for logs, models and single_tournament buffers
mkdir -p ../logs
mkdir -p ../models

# Default configuration
MODE="train"                    # Default mode: train; alternatives: evaluate, analyze
NUM_GAMES=2                   # Number of games for training/evaluation
SAVE_INTERVAL=1                # Save model every SAVE_INTERVAL games
MODEL_PATH="../models/ddqn_model_final.pth"  # Model file to load/save
LEARNING_RATE=0.00001           # Learning rate
GAMMA=0.99                      # Discount factor gamma
BATCH_SIZE=64                   # Training batch size
REPLAY_CAPACITY=10000           # Maximum replay buffer capacity (unused when combining files)
TARGET_UPDATE_FREQ=5            # Update target network every TARGET_UPDATE_FREQ games
EPSILON_START=1.0               # Starting epsilon for exploration
EPSILON_END=0.1                 # Minimum epsilon
EPSILON_DECAY=0.995             # Epsilon decay rate

# Parse command line arguments to override defaults if provided
while [[ $# -gt 0 ]]; do
  case $1 in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --num_games)
      NUM_GAMES="$2"
      shift 2
      ;;
    --save_interval)
      SAVE_INTERVAL="$2"
      shift 2
      ;;
    --model_path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --learning_rate)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --gamma)
      GAMMA="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --replay_capacity)
      REPLAY_CAPACITY="$2"
      shift 2
      ;;
    --target_update_freq)
      TARGET_UPDATE_FREQ="$2"
      shift 2
      ;;
    --epsilon_start)
      EPSILON_START="$2"
      shift 2
      ;;
    --epsilon_end)
      EPSILON_END="$2"
      shift 2
      ;;
    --epsilon_decay)
      EPSILON_DECAY="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Print configuration
echo "=== DDQN Training Configuration ==="
echo "Mode: $MODE"
echo "Number of games: $NUM_GAMES"
echo "Save interval: $SAVE_INTERVAL"
echo "Model path: $MODEL_PATH"
echo "Learning rate: $LEARNING_RATE"
echo "Gamma: $GAMMA"
echo "Batch size: $BATCH_SIZE"
echo "Replay capacity: $REPLAY_CAPACITY"
echo "Target update frequency: $TARGET_UPDATE_FREQ"
echo "Epsilon start: $EPSILON_START"
echo "Epsilon end: $EPSILON_END"
echo "Epsilon decay: $EPSILON_DECAY"
echo "=================================="

# Run the appropriate mode
if [ "$MODE" = "train" ]; then
    echo "Starting DDQN agent training..."
    python train_ddqn_agent.py \
        --mode train \
        --num_games $NUM_GAMES \
        --save_interval $SAVE_INTERVAL \
        --learning_rate $LEARNING_RATE \
        --gamma $GAMMA \
        --batch_size $BATCH_SIZE \
        --replay_capacity $REPLAY_CAPACITY \
        --target_update_freq $TARGET_UPDATE_FREQ \
        --epsilon_start $EPSILON_START \
        --epsilon_end $EPSILON_END \
        --epsilon_decay $EPSILON_DECAY
elif [ "$MODE" = "evaluate" ]; then
    echo "Evaluating trained DDQN agent..."
    python train_ddqn_agent.py \
        --mode evaluate \
        --num_games $NUM_GAMES \
        --model_path $MODEL_PATH
elif [ "$MODE" = "analyze" ]; then
    echo "Analyzing replay buffer..."
    # In this analysis mode, adjust the buffer_path if needed.
    python train_ddqn_agent.py \
        --mode analyze \
        --buffer_path ../monopoly_simulator/replay_buffer.pkl
else
    echo "Unknown mode: $MODE"
    exit 1
fi

echo "Process complete!"