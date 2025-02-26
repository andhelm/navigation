# simulation.py
import numpy as np
import random
from config import GRID_SIZE, EMPTY, WALL, ROBOT, OBJECTIVE


def is_goal_reached(state):
    """
    Check if the robot has reached the objective.

    Parameters:
        state (np.array): 2D grid representing the environment.

    Returns:
        bool: True if the robot's position equals the objective's position.
    """
    robot_indices = np.argwhere(state == ROBOT)
    objective_indices = np.argwhere(state == OBJECTIVE)
    if robot_indices.size == 0 or objective_indices.size == 0:
        return False
    # For a well-formed environment, there is one robot and one objective.
    return np.array_equal(robot_indices[0], objective_indices[0])


def update_state(current_state, action):
    """
    Update the current grid based on the chosen action.

    Actions:
      0: No move
      1: Left
      2: Right
      3: Up
      4: Down

    The function moves the robot if the move is valid (inside boundaries and not into a wall).
    It returns the new state and a reward.

    Parameters:
        current_state (np.array): 2D grid (e.g., (GRID_SIZE, GRID_SIZE)).
        action (int): Action chosen.

    Returns:
        new_state (np.array): Updated grid after applying the action.
        reward (float): Reward for the action.
    """
    new_state = current_state.copy()

    # Find the robot's current position.
    robot_indices = np.argwhere(current_state == ROBOT)
    if robot_indices.size == 0:
        # If no robot is found, return state unchanged and a penalty.
        return current_state, -5.0
    robot_pos = tuple(robot_indices[0])

    # Compute new position.
    new_pos = list(robot_pos)
    if action == 1:  # Left
        new_pos[1] -= 1
    elif action == 2:  # Right
        new_pos[1] += 1
    elif action == 3:  # Up
        new_pos[0] -= 1
    elif action == 4:  # Down
        new_pos[0] += 1
    # Action 0: no move; new_pos remains the same.
    new_pos = tuple(new_pos)

    # Check boundaries.
    if not (0 <= new_pos[0] < current_state.shape[0] and 0 <= new_pos[1] < current_state.shape[1]):
        return current_state, -2.0  # Penalty for invalid move.

    # Check if new position is a wall.
    if current_state[new_pos] == WALL:
        return current_state, -2.0  # Penalty for invalid move.

    # Determine reward.
    if current_state[new_pos] == OBJECTIVE:
        reward = 10.0  # Bonus for reaching the objective.
    else:
        reward = -1.0  # Small time-step penalty.

    # Update state: remove robot from current position and place it at new position.
    new_state[robot_pos] = EMPTY
    new_state[new_pos] = ROBOT
    return new_state, reward


def run_episode(model, initial_state, max_steps=100):
    """
    Run a single episode using the given model.
    At each step, the model predicts the optimal next move given the current state.
    The state is updated after each move, and rewards are accumulated.

    Parameters:
        model: A trained policy network that outputs a probability distribution over 5 actions.
        initial_state (np.array): The starting grid (shape: (GRID_SIZE, GRID_SIZE)).
        max_steps (int): Maximum steps allowed in the episode.

    Returns:
        path (list): List of positions (tuples) the robot visited.
        total_reward (float): Cumulative reward for the episode.
    """
    current_state = initial_state.copy()
    path = []
    total_reward = 0.0

    for step in range(max_steps):
        # Record the current robot position.
        robot_indices = np.argwhere(current_state == ROBOT)
        if robot_indices.size == 0:
            # No robot found; break out.
            break
        current_pos = tuple(robot_indices[0])
        path.append(current_pos)

        if is_goal_reached(current_state):
            break

        # Prepare input: reshape to (1, GRID_SIZE, GRID_SIZE, 1).
        state_input = current_state.reshape(1, current_state.shape[0], current_state.shape[1], 1).astype('float32')
        # Get predicted action probabilities.
        action_probs = model.predict(state_input)[0]  # shape: (NUM_ACTIONS,)
        # Sample an action from the distribution.
        action = np.random.choice(range(5), p=action_probs)

        # Update the environment state.
        current_state, reward = update_state(current_state, action)
        total_reward += reward

    return path, total_reward


if __name__ == '__main__':
    # For testing purposes, generate an environment using your arena generator.
    from arenas.random_env import generate_arena

    initial_state = generate_arena()
    print("Initial state shape:", initial_state.shape)  # Should be (GRID_SIZE, GRID_SIZE)

    # For testing, create a dummy model that predicts random actions.
    import tensorflow as tf

    dummy_model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(initial_state.shape[0], initial_state.shape[1], 1)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    dummy_model.compile(optimizer='adam', loss='categorical_crossentropy')

    path, total_reward = run_episode(dummy_model, initial_state)
    print("Path:", path)
    print("Total reward:", total_reward)
