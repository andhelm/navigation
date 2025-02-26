# train/train_with_keras.py
import numpy as np
import tensorflow as tf
from agent.model import build_model
from arenas.environment import generate_random_environments
from rewards.reward_functions import compute_reward
from train.animation_callback import AnimationCallback

from config import NUM_MOVES, NUM_ACTIONS, NUM_ENVIRONMENTS, GRID_SIZE, LEARNING_RATE, ENTROPY_BETA, BATCH_SIZE


class RLModel(tf.keras.Model):
    def __init__(self, base_model, **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model

    def train_step(self, data):
        # data: a batch of environment grids (shape: (BATCH_SIZE, GRID_SIZE, GRID_SIZE))
        env = data
        # Add a channel dimension so shape becomes (BATCH_SIZE, GRID_SIZE, GRID_SIZE, 1)
        env_input = tf.expand_dims(tf.cast(env, tf.float32), axis=-1)

        with tf.GradientTape() as tape:
            # Forward pass: output shape (BATCH_SIZE, NUM_MOVES, NUM_ACTIONS)
            action_probs = self.base_model(env_input, training=True)
            # Do NOT squeeze here, as we want to keep the batch dimension.

            # Use TF ops to sample actions for each environment in the batch:
            logits = tf.math.log(action_probs + 1e-8)
            reshaped_logits = tf.reshape(logits, (-1, NUM_ACTIONS))
            sampled_actions = tf.random.categorical(reshaped_logits, num_samples=1)
            sampled_actions = tf.reshape(sampled_actions, (-1, NUM_MOVES))  # shape: (BATCH_SIZE, NUM_MOVES)

            # Compute one-hot encodings for the sampled actions:
            actions_onehot = tf.one_hot(sampled_actions,
                                        depth=NUM_ACTIONS)  # shape: (BATCH_SIZE, NUM_MOVES, NUM_ACTIONS)

            # Compute log probabilities for each move, for each environment:
            log_probs = tf.math.log(
                tf.reduce_sum(action_probs * actions_onehot, axis=-1) + 1e-8)  # shape: (BATCH_SIZE, NUM_MOVES)

            # Sum log probabilities over moves for each environment:
            total_log_prob = tf.reduce_sum(log_probs, axis=1)  # shape: (BATCH_SIZE,)

            # Compute rewards for each environment in the batch.
            def compute_reward_wrapper(env_sample, actions_sample):
                return compute_reward(env_sample, actions_sample)

        rewards = tf.map_fn(
            lambda x: tf.py_function(
                func=compute_reward_wrapper,
                inp=[x[0], x[1]],
                Tout=tf.float32),
            (env, sampled_actions),
            fn_output_signature=tf.float32
        )
        # rewards now contains the discounted cumulative rewards for each environment.

        # Compute baseline as the average reward in the batch.
        avg_reward = tf.reduce_mean(rewards)
        advantage = rewards - avg_reward

        entropy = -tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-8), axis=-1)
        total_entropy = tf.reduce_sum(entropy, axis=1)
        loss = -tf.reduce_mean(total_log_prob * advantage + ENTROPY_BETA * total_entropy)

        gradients = tape.gradient(loss, self.base_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.base_model.trainable_variables))

        # Return the average loss and average reward for the batch.
        return {"loss": loss, "reward": tf.reduce_mean(rewards)}

        gradients = tape.gradient(loss, self.base_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.base_model.trainable_variables))

        # Return the average loss and average reward for the batch.
        return {"loss": loss, "reward": tf.reduce_mean(rewards)}

    def call(self, inputs, training=False):
        return self.base_model(inputs, training=training)


def run_training():
    # Generate a set of environments for training.
    envs = generate_random_environments(num_envs=NUM_ENVIRONMENTS)
    # Create a tf.data.Dataset from these environments.
    dataset = tf.data.Dataset.from_tensor_slices(envs).batch(BATCH_SIZE)

    # Build the base model.
    input_shape = (GRID_SIZE, GRID_SIZE, 1)
    base_model = build_model(input_shape, num_moves=NUM_MOVES, num_actions=NUM_ACTIONS)

    # Wrap it in our custom RLModel.
    rl_model = RLModel(base_model)
    rl_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))

    # Use one sample environment for animation.
    sample_envs = envs[:4]  # first four environments
    anim_cb = AnimationCallback(sample_envs, save_dir="epoch_plots")

    # Train using model.fit with our animation callback.
    rl_model.fit(dataset, epochs=50, callbacks=[anim_cb])

if __name__ == '__main__':
    run_training()
