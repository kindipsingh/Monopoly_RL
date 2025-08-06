# Monopoly RL Agent

This project aims to develop a reinforcement learning agent capable of playing the board game Monopoly. The agent is trained using a Double Deep Q-Network (DDQN) to learn optimal strategies for winning the game.

## Project Goal

The primary objective of this project is to create an autonomous agent that can play Monopoly at a competitive level. This involves:
- Designing a suitable state representation for the game of Monopoly.
- Implementing a DDQN model to learn the value of different actions in different states.
- Training the agent in a simulated Monopoly environment.
- Evaluating the agent's performance against other baseline agents.

## Project Structure

The project is organized as follows:

- `monopoly_simulator/`: This directory contains the core components of the Monopoly game simulator and the RL agent.
  - `player.py`: Defines the Player class and its interactions with the game.
  - `ddqn_decision_agent.py`: Implements the decision-making logic for the DDQN agent.
  - `ddqnn.py`: Contains the implementation of the Double Deep Q-Network model.
  - `training_ddqnn.py` / `train_ddqn_agent.py`: Scripts for training the DDQN agent.
  - `replay_buffer_module.py`: Implements the replay buffer for storing experiences.
  - `test_harness_phase3.py`: A test harness for evaluating the agent's performance.
  - `gameplay_socket_phase3.py`: Handles the socket-based communication for gameplay.

## How to Run

### Training the Agent

To train the DDQN agent, you can use the provided shell script:

```bash
./GNOME-p3-phase3_final/monopoly_simulator/train_ddqn.sh
```

### Running the Agent

To execute a game with the trained agent, use the following script:

```bash
./GNOME-p3-phase3_final/monopoly_simulator/execute_phase3.sh
```
