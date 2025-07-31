# Car Game RL Environment

This repository contains a Reinforcement Learning (RL) environment for a car racing game implemented using Pygame and trained using the Stable Baselines3 DQN algorithm. The environment simulates a car navigating through lanes to avoid obstacles, with the goal of maximizing the score by passing vehicles.

## Table of Contents

- [Overview](#overview)
- [Files](#files)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Evaluating the Model](#evaluating-the-model)
  - [Running the Game with AI](#running-the-game-with-ai)
- [Environment Details](#environment-details)
- [Dependencies](#dependencies)
- [Directory Structure](#directory-structure)
- [License](#license)

## Overview

The project implements a lane-based car game where an RL agent learns to navigate through traffic. The agent is trained using the Deep Q-Network (DQN) algorithm from Stable Baselines3. The game environment is built with Pygame and uses Gymnasium for RL compatibility. The agent observes the player's lane and distances to nearby vehicles, choosing actions to move left, stay, or move right.

## Files

- **`car_game_env.py`**: Defines the `CarGameEnv` class, a Gymnasium environment for the car game, handling game logic, rendering, and RL integration.
- **`train.py`**: Script to train the DQN model, save the best model, and optionally record training videos.
- **`eval.py`**: Script to evaluate the trained DQN model and record videos of the agent's performance.
- **`game.py`**: Runs the game with the trained RL agent, including a graphical interface, score logging, and live performance plotting.
- **`images/`**: Directory containing vehicle images (`car.png`, `pickup_truck.png`, `semi_trailer.png`, `taxi.png`, `van.png`, `crash.png`) used for rendering.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Hharkheem/Car-Game-Reinforcement-Learning
   cd Car-Game-Reinforcement-Learning
   ```
2. Install dependencies (see [Dependencies](#dependencies)).
3. Ensure the `images/` directory is in the project root with the required image files.

## Usage

### Training the Model

Run `train.py` to train the DQN model:

```bash
python train.py
```

- The model trains for 50 million timesteps.
- The best model is saved in `best_model/best_model.zip`.
- Training videos are saved in `training_videos/` (every 10th episode).
- Logs are saved in `logs/`, and TensorBoard logs are in `tensorboard/`.

### Evaluating the Model

Run `eval.py` to evaluate the trained model:

```bash
python eval.py
```

- Loads the model from `car_rl_model.zip`.
- Records evaluation videos in `eval_videos/`.

### Running the Game with AI

Run `game.py` to play the game with the trained RL agent:

```bash
python game.py
```

- Loads the best model from `best_model/best_model.zip`.
- Displays the game with Pygame, showing the AI-controlled car.
- Logs scores to `scores.csv`.
- Press `R` to restart after a game over, or `ESC` to quit.

## Environment Details

- **Action Space**: Discrete (3 actions: move left, stay, move right).
- **Observation Space**: Box (4 floats: player lane, distance to closest vehicle in each of the three lanes).
- **Reward Structure**:
  - +0.1 for surviving each step.
  - +2 for passing a vehicle.
  - -1/distance penalty for proximity to vehicles.
  - -20 for crashing.
- **Rendering**: Supports `human` (Pygame window) and `rgb_array` (for video recording) modes.
- **Game Mechanics**:
  - The player car starts in the center lane at y=400.
  - Vehicles spawn randomly in one of three lanes.
  - Speed increases every 5 points.
  - Game ends on collision with another vehicle.

## Dependencies

- Python 3.8+
- `pygame`
- `gymnasium`
- `stable-baselines3`
- `numpy`
- `matplotlib` (for live plotting in `game.py`)
- `opencv-python` (for video recording in `train.py` and `eval.py`)

Install dependencies using:

```bash
pip install pygame gymnasium stable-baselines3 numpy matplotlib opencv-python
```

## Directory Structure

```
car-game-rl/
├── images/
│   ├── car.png
│   ├── pickup_truck.png
│   ├── semi_trailer.png
│   ├── taxi.png
│   ├── van.png
│   ├── crash.png
├── best_model/
│   ├── best_model.zip
├── eval_videos/
├── training_videos/
├── logs/
├── tensorboard/
├── car_game_env.py
├── train.py
├── eval.py
├── game.py
├── scores.csv
├── README.md
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
