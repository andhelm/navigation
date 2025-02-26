# agent/model.py
import tensorflow as tf
from tensorflow.keras import layers, models
from config import GRID_SIZE, NUM_ACTIONS, USE_CNN

def build_model(input_shape, num_actions=NUM_ACTIONS):
    """
    Build and compile a neural network model that predicts the optimal next move.

    Parameters:
        input_shape (tuple): Shape of the input arena (e.g., (GRID_SIZE, GRID_SIZE, 1)).
        num_actions (int): Number of possible actions.

    Returns:
        tf.keras.Model: The compiled Keras model.
    """
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))

    if USE_CNN:
        # Use convolutional layers for spatial feature extraction.
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
    else:
        # Simply flatten the input for a raw input approach.
        model.add(layers.Flatten())

    # Fully connected layers.
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    # Output layer: predicts probability distribution over actions.
    model.add(layers.Dense(num_actions, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    # Example: Build model for a GRID_SIZE x GRID_SIZE arena.
    input_shape = (GRID_SIZE, GRID_SIZE, 1)
    model = build_model(input_shape)
    model.summary()
