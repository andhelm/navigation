# agent/robot.py
import numpy as np


class Robot:
    def __init__(self, start_pos):
        """
        Initialize the robot with a starting position.

        Parameters:
            start_pos (tuple): The starting (row, col) of the robot.
        """
        self.position = start_pos

    def move(self, direction, arena):
        """
        Move the robot in the specified direction if the move is valid.

        Parameters:
            direction (int): Direction code where
                             0 = no move, 1 = left, 2 = right, 3 = up, 4 = down.
            arena (np.array): The current arena grid (used for boundary checks).

        Returns:
            tuple: Updated position of the robot.
        """
        new_pos = list(self.position)
        if direction == 1:  # left
            new_pos[1] -= 1
        elif direction == 2:  # right
            new_pos[1] += 1
        elif direction == 3:  # up
            new_pos[0] -= 1
        elif direction == 4:  # down
            new_pos[0] += 1
        # If direction is 0, no movement is done.

        new_pos = tuple(new_pos)
        # Check boundaries (assume arena is a 2D numpy array)
        if (0 <= new_pos[0] < arena.shape[0] and 0 <= new_pos[1] < arena.shape[1]):
            self.position = new_pos
        return self.position

    def jump(self, new_position, arena):
        """
        Jump directly to a new position if valid.

        Parameters:
            new_position (tuple): Target position for the jump.
            arena (np.array): The current arena grid (used for boundary checks).

        Returns:
            tuple: Updated position of the robot.
        """
        if (0 <= new_position[0] < arena.shape[0] and 0 <= new_position[1] < arena.shape[1]):
            self.position = new_position
        return self.position
