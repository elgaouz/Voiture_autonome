# Self-driving Car AI Simulation

A PyTorch-based implementation of a self-driving car using Deep Q-Learning. The car learns to navigate through a custom environment, avoiding obstacles while trying to reach its target.

## Features
- Deep Q-Learning implementation for autonomous navigation
- Interactive environment with drawable obstacles
- Real-time visualization of car sensors and decision making
- Save/Load functionality for trained models
- Performance tracking and visualization

## Interface
![Interface Elements](interface/Capture%20d’écran%202024-12-03%20115603.png)
- **Red Circle**: The car with its direction indicator
- **Blue Lines**: Three sensors detecting obstacles
- **Green Circle**: Target destination
- **Dark Blue Lines**: Obstacles (sand) drawn by user
- **Control Buttons**:
  - Clear: Reset the environment
  - Save: Store trained model
  - Load: Use previously trained model

## How to Use
1. Run `map.py` to start the simulation
2. Left-click and drag to draw obstacles (sand)
3. Watch the car learn to navigate to the green target
4. Use buttons to manage the simulation

## Requirements
- Python 3.x
- PyTorch
- Pygame
- Numpy
- Matplotlib

## Project Structure
- `map.py`: Main simulation environment
- `ai.py`: Deep Q-Learning implementation

The car uses three sensors to detect obstacles and learns to navigate through reinforcement learning.