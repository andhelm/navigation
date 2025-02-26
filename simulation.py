# simulation.py
import numpy as np
import random
import tensorflow as tf
from config import GRID_SIZE, EMPTY, WALL, ROBOT, OBJECTIVE, GAMMA

def is_goal_reached(state):
    robot_indices = np.argwhere(state == ROBOT)
    objective_indices = np.argwhere(state == OBJECTIVE)
    if robot_indices.size == 0 or objective_indices.size == 0:
        return False
    return np.array_equal(robot_indices[0], objective_indices[0])

def update_state(current_state, action):
    new_state = current_state.copy()
    robot_indices = np.argwhere(current_state == ROBOT)
    if robot_indices.size == 0:
        return current_state, -5.0
    robot_pos = tuple(robot_indices[0])
    new_pos = list(robot_pos)
    if action == 1:
        new_pos[1] -= 1
    elif action == 2:
        new_pos[1] += 1
    elif action == 3:
        new_pos[0] -= 1
    elif action == 4:
        new_pos[0] += 1
    new_pos = tuple(new_pos)
    if not (0 <= new_pos[0] < current_state.shape[0] and 0 <= new_pos[1] < current_state.shape[1]):
        return current_state, -2.0
    if current_state[new_pos] == WALL:
        return current_state, -2.0
    if current_state[new_pos] == OBJECTIVE:
        reward = 10.0
    else:
        reward = -1.0
    new_state[robot_pos] = EMPTY
    new_state[new_pos] = ROBOT
    return new_state, reward

def run_episode_and_collect_gradients(model, initial_state, max_steps=100, discount=True):
    state = initial_state.copy()
    log_probs = []
    rewards = []

    for t in range(max_steps):
        if is_goal_reached(state):
            break

        state_input = state.reshape(1, state.shape[0], state.shape[1], 1).astype('float32')
        action_probs = model(state_input, training=True)  # shape: (1, NUM_ACTIONS)
        action_probs = tf.squeeze(action_probs, axis=0)     # shape: (NUM_ACTIONS,)

        logits = tf.math.log(action_probs + 1e-8)
        sampled_action_tensor = tf.random.categorical(tf.reshape(logits, (1, -1)), num_samples=1)
        sampled_action_tensor = tf.squeeze(sampled_action_tensor, axis=0)  # tf.int32 tensor

        # Compute log probability in a differentiable manner.
        log_prob = tf.gather(tf.math.log(action_probs + 1e-8), sampled_action_tensor)
        log_probs.append(log_prob)

        # For state update, convert sampled_action_tensor to a Python int.
        action_int = int(sampled_action_tensor.numpy())
        state, reward = update_state(state, action_int)
        rewards.append(reward)

    # Compute rewards: discount if requested.
    if discount:
        discounted_rewards = []
        cumulative = 0.0
        for r in reversed(rewards):
            cumulative = r + GAMMA * cumulative
            discounted_rewards.insert(0, cumulative)
    else:
        discounted_rewards = rewards

    log_probs = tf.stack(log_probs)
    discounted_rewards = tf.convert_to_tensor(discounted_rewards, dtype=tf.float32)
    return log_probs, discounted_rewards

if __name__ == '__main__':
    from arenas.random_env import generate_arena
    initial_state = generate_arena()
    print("Initial state shape:", initial_state.shape)

    dummy_model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(initial_state.shape[0], initial_state.shape[1], 1)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    dummy_model.compile(optimizer='adam', loss='categorical_crossentropy')

    log_probs, discounted_rewards = run_episode_and_collect_gradients(dummy_model, initial_state)
    print("Log probs:", log_probs.numpy())
    print("Discounted rewards:", discounted_rewards.numpy())
