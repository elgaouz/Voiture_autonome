# Self-driving Car AI Simulation

A PyTorch-based implementation of a self-driving car using Deep Q-Learning. The car learns to navigate through a custom environment, avoiding obstacles while trying to reach its target.

## Features
- Deep Q-Learning implementation for autonomous navigation
- Interactive environment with drawable obstacles
- Real-time visualization of car sensors and decision making
- Save/Load functionality for trained models
- Performance tracking and visualization

## Requirements
- Python 3.x
- PyTorch
- Pygame
- Numpy
- Matplotlib

## How to Use
1. Run `map.py` to start the simulation
2. Left-click and drag to draw obstacles (sand)
3. Use buttons to:
   - Clear: Reset the environment
   - Save: Store the trained model
   - Load: Use a previously trained model

## Project Structure
- `map.py`: Main simulation environment
- `ai.py`: Deep Q-Learning implementation

The car uses three sensors to detect obstacles and learns to navigate through reinforcement learning.
