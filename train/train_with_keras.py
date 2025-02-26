# train/train_with_keras.py
import tensorflow as tf
import numpy as np
from agent.model import build_model
from arenas.random_env import generate_arena
from simulation import run_episode_and_collect_gradients
from config import GRID_SIZE, NUM_ACTIONS, LEARNING_RATE, GAMMA, NUM_EPISODES, MAX_STEPS, ENTROPY_BETA
from train.visualization_callback import EpisodeVisualizationCallback


def train(model, optimizer, num_episodes, max_steps=100, vis_callback=None):
    for episode in range(num_episodes):
        initial_state = generate_arena()  # Generate a new environment.
        log_probs, discounted_rewards = run_episode_and_collect_gradients(model, initial_state, max_steps)
        if tf.size(log_probs) == 0:
            continue

        baseline = tf.reduce_mean(discounted_rewards)
        advantage = discounted_rewards - baseline
        loss = -tf.reduce_sum(log_probs * advantage)

        with tf.GradientTape() as tape:
            # Recompute the loss inside the tape so it connects to model parameters.
            log_probs, discounted_rewards = run_episode_and_collect_gradients(model, initial_state, max_steps)
            baseline = tf.reduce_mean(discounted_rewards)
            advantage = tf.stop_gradient(discounted_rewards - baseline)
            loss = -tf.reduce_sum(log_probs * advantage)
            final_loss = loss
        gradients = tape.gradient(final_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if episode % 10 == 0:
            total_reward = tf.reduce_sum(discounted_rewards)
            print(f"Episode {episode}: Loss = {loss.numpy():.4f}, Total Reward = {total_reward.numpy():.4f}")
            # Manually call the visualization callback.
            if vis_callback is not None:
                vis_callback.on_epoch_end(episode, logs={"reward": total_reward.numpy()})


def run_training():
    input_shape = (GRID_SIZE, GRID_SIZE, 1)
    model = build_model(input_shape, num_actions=NUM_ACTIONS)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # Generate a fixed sample environment for visualization.
    sample_env = generate_arena()
    # Pass the model to the callback directly.
    vis_cb = EpisodeVisualizationCallback(sample_env, model, interval=10)

    train(model, optimizer, num_episodes=NUM_EPISODES, max_steps=MAX_STEPS, vis_callback=vis_cb)


if __name__ == '__main__':
    run_training()
