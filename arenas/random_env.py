import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as colors

# Constants for the arena
from config import EMPTY, WALL, ROBOT, OBJECTIVE, GRID_SIZE

def generate_arena():
    # Start with an empty grid
    arena = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    # Add walls (randomly scattered)
    num_walls = random.randint(GRID_SIZE, GRID_SIZE * 2)  # Random number of walls
    num_walls=0
    for _ in range(num_walls):
        x, y = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
        arena[x][y] = WALL

    # Ensure the robot and objective are not on walls
    while True:
        robot_x, robot_y = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
        if arena[robot_x][robot_y] == EMPTY:
            arena[robot_x][robot_y] = ROBOT
            break

    while True:
        objective_x, objective_y = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
        if arena[objective_x][objective_y] == EMPTY:
            arena[objective_x][objective_y] = OBJECTIVE
            break

    return arena


def visualize_arena(arena):
    # Define a custom colormap: index order corresponds to EMPTY, WALL, ROBOT, OBJECTIVE
    cmap = colors.ListedColormap(['lightgray', 'blue', 'green', 'red'])

    # Display the arena with the custom colormap
    plt.imshow(arena, cmap=cmap, interpolation='nearest')

    # Create legend patches for each type
    legend_patches = [
        mpatches.Patch(color='lightgray', label='Empty'),
        mpatches.Patch(color='blue', label='Wall'),
        mpatches.Patch(color='green', label='Robot'),
        mpatches.Patch(color='red', label='Obj.')
    ]

    # Add the legend to the plot
    plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title("Arena Visualization")
    plt.show()