# config.py

# Environment definitions
EMPTY = 0
WALL = 1
ROBOT = 2
OBJECTIVE = 3

# Simulation parameters
GRID_SIZE = 10              # The grid is GRID_SIZE x GRID_SIZE.
NUM_ACTIONS = 5             # 0: no move, 1: left, 2: right, 3: up, 4: down

# Training parameters
LEARNING_RATE = 0.001
NUM_EPISODES = 50000        # Total number of episodes for training.
MAX_STEPS = 100             # Maximum number of steps per episode.
ENTROPY_BETA = 0.2          # Entropy regularization coefficient.

# Model configuration
USE_CNN = True              # Toggle CNN feature extraction. If False, uses raw flattened input.

# Reinforcement learning parameters
GAMMA = 0.99                # Discount factor for future rewards.

# Reward function parameters
distance_penalty_factor = 1.0
bonus = 100.0               # Bonus reward for reaching the objective.
penalty_per_step = 1.0      # Penalty applied per time step.

penalty_per_step = 1.0