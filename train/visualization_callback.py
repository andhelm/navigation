# train/visualization_callback.py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tensorflow as tf
from simulation import run_episode_and_collect_gradients  # or run_episode if that's what you prefer
from config import GRID_SIZE, EMPTY, WALL, ROBOT, OBJECTIVE

class EpisodeVisualizationCallback(tf.keras.callbacks.Callback):
    def __init__(self, env, model, interval=10):
        """
        Parameters:
          env: A sample environment (2D numpy array of shape (GRID_SIZE, GRID_SIZE))
               used for visualization.
          model: The current policy network.
          interval: How often (in episodes) to visualize the solution path.
        """
        super().__init__()
        self.env = env
        self._model = model
        self.interval = interval

    def on_epoch_end(self, epoch, logs=None):
        if (epoch % self.interval) == 0:
            # Run an episode using the current model.
            # Here we set discount=False just to see the raw path.
            path, total_reward = run_episode_and_collect_gradients(self._model, self.env, max_steps=100, discount=False)

            # Define a custom colormap.
            color_list = ["lightgray", "blue", "green", "red"]
            custom_cmap = mcolors.ListedColormap(color_list)

            plt.figure(figsize=(6, 6))
            plt.imshow(self.env, cmap=custom_cmap, vmin=0, vmax=3)
            if path is not None and len(path) > 0:
                xs = [pos[1] for pos in path]
                ys = [pos[0] for pos in path]
                plt.plot(xs, ys, marker="o", color="red")

            plt.title(f"Epoch {epoch + 1}, Reward: {total_reward:.2f}")
            plt.axis("off")
            plt.show(block=False)
            plt.pause(2)
            plt.close()
