# arenas/environment.py
import os
import numpy as np
from .random_env import generate_arena

from config import NUM_ENVIRONMENTS

def generate_random_environments(num_envs=NUM_ENVIRONMENTS):
    """
    Generate a list of random environments.

    Parameters:
        num_envs (int): Number of random environments to generate.

    Returns:
        List[np.array]: A list of randomly generated environments.
    """
    environments = [generate_arena() for _ in range(num_envs)]

    arena = generate_arena()
    print(arena.shape)  # Should print (GRID_SIZE, GRID_SIZE)

    return environments

def save_environments(environments, save_dir="data/environments"):
    """
    Save a list of environments as .npy files for later use.

    Parameters:
        environments (list of np.array): The list of environment arrays.
        save_dir (str): The directory where environments will be saved.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for idx, env in enumerate(environments):
        filename = os.path.join(save_dir, f"environment_{idx}.npy")
        np.save(filename, env)

if __name__ == "__main__":
    # Example usage: Generate 100 random environments and save them.
    envs = generate_random_environments(NUM_ENVIRONMENTS)
    save_environments(envs)

    print("Environments generated and saved.")
