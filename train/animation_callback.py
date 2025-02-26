# train/animation_callback.py
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tensorflow as tf

from config import EMPTY, WALL, ROBOT, OBJECTIVE, NUM_MOVES, GRID_SIZE


def simulate_path(env, model):
    """
    Simulate a solution path in a given environment using the current model.
    Returns a list of positions representing the path.
    """
    # Work on a copy of the environment.
    env_copy = env.copy()

    # Ensure the environment is 2D.
    if env_copy.ndim != 2:
        total_elements = env_copy.size
        side = int(np.sqrt(total_elements))
        if side * side == total_elements:
            env_copy = env_copy.reshape((side, side))
            print(f"Reshaped environment to {env_copy.shape}")
        else:
            print(f"Warning: Environment shape {env_copy.shape} is not 2D.")
            return []

    robot_indices = np.argwhere(env_copy == ROBOT)
    objective_indices = np.argwhere(env_copy == OBJECTIVE)

    if robot_indices.size == 0:
        print("Warning: No ROBOT marker found in environment.")
        return []
    if objective_indices.size == 0:
        print("Warning: No OBJECTIVE marker found in environment.")
        return []

    robot_pos = tuple(robot_indices[0])
    objective_pos = tuple(objective_indices[0])
    path = [robot_pos]
    current_pos = robot_pos

    # Prepare input for the model.
    state_input = tf.expand_dims(tf.expand_dims(tf.cast(env_copy, tf.float32), axis=-1), axis=0)
    action_probs = model(state_input, training=False)
    # Remove the batch dimension; shape becomes (NUM_MOVES, num_actions)
    action_probs = tf.squeeze(action_probs, axis=0).numpy()
    moves = np.argmax(action_probs, axis=1)  # for debugging
    print("Moves for debugging:", moves)

    def move_from(pos, action):
        new_pos = list(pos)
        # 0: no move, 1: left, 2: right, 3: up, 4: down
        if action == 1:
            new_pos[1] -= 1
        elif action == 2:
            new_pos[1] += 1
        elif action == 3:
            new_pos[0] -= 1
        elif action == 4:
            new_pos[0] += 1
        return tuple(new_pos)

    # Simulate moves using the most probable action for each time step.
    for i in range(NUM_MOVES):
        action = np.argmax(action_probs[i])
        next_pos = move_from(current_pos, action)
        if (0 <= next_pos[0] < env_copy.shape[0] and
                0 <= next_pos[1] < env_copy.shape[1] and
                env_copy[next_pos] != WALL):
            current_pos = next_pos
            path.append(current_pos)
            if current_pos == objective_pos:
                break
        else:
            path.append(current_pos)
    #print("Final path:", path)
    return path


class AnimationCallback(tf.keras.callbacks.Callback):
    def __init__(self, env_list, save_dir="epoch_plots"):
        """
        Initialize the callback with a list of sample environments.

        Parameters:
            env_list: list of numpy arrays representing sample environments.
            save_dir: directory where plots will be saved.
        """
        super().__init__()
        self.env_list = env_list
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        # Use only the first four environments.
        sample_envs = self.env_list[:4]

        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        axs = axs.flatten()
        # Custom colormap: 0: lightgray, 1: blue, 2: green, 3: red.
        color_list = ["lightgray", "blue", "green", "red"]
        custom_cmap = mcolors.ListedColormap(color_list)

        for idx, env in enumerate(sample_envs):
            ax = axs[idx]
            # Check that environment is 2D. If not, try to reshape.
            if env.ndim != 2:
                total_elements = env.size
                side = int(np.sqrt(total_elements))
                if side * side == total_elements:
                    env = env.reshape((side, side))
                    print(f"Reshaped environment at index {idx} to {env.shape}")
                else:
                    print(f"Warning: Environment at index {idx} has invalid shape {env.shape}.")
                    ax.text(0.5, 0.5, "Invalid env shape", horizontalalignment='center',
                            verticalalignment='center', transform=ax.transAxes, color='red')
                    ax.axis('off')
                    continue

            path = simulate_path(env, self.model)
            try:
                ax.imshow(env, cmap=custom_cmap, vmin=0, vmax=3)
            except Exception as e:
                print(f"Error displaying environment at index {idx}: {e}")
                ax.text(0.5, 0.5, "Display error", horizontalalignment='center',
                        verticalalignment='center', transform=ax.transAxes, color='red')
                ax.axis('off')
                continue

            if path:
                xs = [p[1] for p in path]
                ys = [p[0] for p in path]
                ax.plot(xs, ys, marker='o', color='red')
            else:
                ax.text(0.5, 0.5, "Invalid env", horizontalalignment='center',
                        verticalalignment='center', transform=ax.transAxes, color='red')
            ax.set_title(f"Env {idx + 1} - Epoch {epoch + 1}")
            ax.axis('off')

        plt.tight_layout()
        # Save the figure to the specified directory.
        filename = os.path.join(self.save_dir, f"epoch_{epoch + 1:03d}.png")
        plt.savefig(filename)
        print(f"Saved epoch plot to {filename}")
        # Display the figure as before.
        plt.show(block=False)
        plt.pause(2)
        plt.close(fig)
