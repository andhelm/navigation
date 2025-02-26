# config.py

# Environment definitions
EMPTY = 0
WALL = 1
ROBOT = 2
OBJECTIVE = 3

# Training and simulation parameters
NUM_MOVES = 20
NUM_ACTIONS = 5  # 0: no move, 1: left, 2: right, 3: up, 4: down
# You can add other parameters such as grid size, learning rates, etc.
GRID_SIZE = 10
LEARNING_RATE = 0.001
NUM_ENVIRONMENTS = 50000
ENTROPY_BETA = 0.2  # Adjust as needed
BATCH_SIZE = 32
USE_CNN = True

GAMMA = 0.99
distance_penalty_factor = 1.0
bonus = 100.0
penalty_per_step = 1.0