# rewards/reward_functions.py
import numpy as np
from config import EMPTY, WALL, ROBOT, OBJECTIVE, GAMMA, \
    distance_penalty_factor  # also include other constants like bonus, penalty_per_step


def compute_reward(initial_state, moves):
    """
    Computes the discounted cumulative reward for a given sequence of moves starting from the initial state.
    The discount factor is applied based on the number of steps taken.

    Parameters:
        initial_state (np.array): The starting arena grid.
        moves (list or np.array): A sequence of moves (integers: 0 no move, 1 left, etc.)

    Returns:
        np.float32: The discounted cumulative reward.
    """
    # Locate the robot and objective in the initial state.
    robot_pos = tuple(np.argwhere(initial_state == ROBOT)[0])
    objective_pos = tuple(np.argwhere(initial_state == OBJECTIVE)[0])
    current_pos = robot_pos
    steps = 0

    # Parameters (you can also move these to config.py):
    penalty_per_step = 1.0
    bonus = 100.0

    # Simulate the robot's movement.
    reached = False
    for move in moves:
        steps += 1
        new_pos = list(current_pos)
        if move == 1:  # left
            new_pos[1] -= 1
        elif move == 2:  # right
            new_pos[1] += 1
        elif move == 3:  # up
            new_pos[0] -= 1
        elif move == 4:  # down
            new_pos[0] += 1
        new_pos = tuple(new_pos)
        # Check if new position is valid (inside grid and not a wall).
        if (0 <= new_pos[0] < initial_state.shape[0] and
                0 <= new_pos[1] < initial_state.shape[1] and
                initial_state[new_pos] != WALL):
            current_pos = new_pos
        # Stop if objective reached.
        if current_pos == objective_pos:
            reached = True
            break

    if reached:
        # A higher reward for reaching quickly.
        final_reward = bonus - penalty_per_step * steps
    else:
        # Penalize based on steps and final Manhattan distance.
        distance = abs(current_pos[0] - objective_pos[0]) + abs(current_pos[1] - objective_pos[1])
        final_reward = - (penalty_per_step * steps + distance_penalty_factor * distance)

    discounted_reward = (GAMMA ** steps) * final_reward
    return np.float32(discounted_reward)
