"""
This file implements the Maze class
"""
import re
from enum import Enum


# We allow 4 possible actions in the maze - left, right, up and down
class Action(Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3


# This is the Maze class, and its job is to capture the information about the world that our Agent is navigating.
class Maze:
    # First, we initialize the maze using a text file. The text file has the following information
    # 1. The size of the maze (Rows x Columns)
    # 2. The cost of every action - left, right, up and down
    # 3. The layout of the maze. A is the starting point, B is the goal, and * marks a pit
    def __init__(self, f_name):
        # Parse the text file input
        self.start = None  # record the position of the starting location
        self.end = set()  # record the position of the target location(s)
        self.pits = set()  # set of co-ordinates that are pits. Our agent cannot walk into one.
        self.height = None
        self.width = None
        # Each action need not have the same cost. Here, we capture the cost of the actions we can take
        self.action_costs = []

        # Here we read the file with the maze and populate it
        with open(f_name, 'r') as file:
            self.width, self.height = [int(x) for x in file.readline().split()]

            self.action_costs = [int(x) for x in file.readline().split()]
            if len(self.action_costs) != 4:
                raise ValueError("Need to specify the cost of all four actions.")

            row = 0
            for line in file:
                line = re.sub(r'\n', '', line)
                for col, c in enumerate(line):
                    if c == 'A':
                        if self.start is not None:
                            raise ValueError("Cannot have two Start positions")
                        self.start = self.image_coordinate_to_cartesian(row, col)
                    elif c == 'B':
                        self.end.add(self.image_coordinate_to_cartesian(row, col))
                    elif c == '*':
                        self.pits.add(self.image_coordinate_to_cartesian(row, col))
                    elif c != '.':
                        print(c)
                        raise ValueError("Unknown character")
                row += 1

        if self.start is None:
            raise ValueError("No starting point")

        if len(self.end) == 0:
            raise ValueError("No ending points")

        if self.width <= 0:
            raise ValueError("Invalid width")

        if self.height <= 0:
            raise ValueError("Invalid height")

    # Helper function to make coordinates easier to work with
    def image_coordinate_to_cartesian(self, row, col):
        return col, self.height - 1 - row

    # Given an action. return the cost of an action
    def action_cost(self, action: Action):
        return self.action_costs[action.value]

    # Now, we define a series of functions that let us get the permitted actions from a given location in the maze
    # First we define a method to compute where we would be after taking an action from a given location
    @staticmethod
    def resulting_coord(starting_coord, action: Action):
        if action == Action.LEFT:
            return starting_coord[0] - 1, starting_coord[1]
        if action == Action.RIGHT:
            return starting_coord[0] + 1, starting_coord[1]
        if action == Action.UP:
            return starting_coord[0], starting_coord[1] + 1
        if action == Action.DOWN:
            return starting_coord[0], starting_coord[1] - 1

    # Next, we define a function that checks if a coordinate is valid
    # A valid coordinate must be within the boundary of the grid, and not a pit
    def valid_coord(self, coord):
        x, y = coord
        in_boundary = (x >= 0) and (x < self.width) and (y >= 0) and (y < self.height)
        not_in_pit = coord not in self.pits
        return in_boundary and not_in_pit

    # Finally, we define the function that provides a list of valid actions given a coordinate
    # For all valid actions, they are ordered by "LEFT", "RIGHT", "UP", "DOWN".
    def valid_ordered_action(self, coord):
        valid_actions = []
        for action in [Action.LEFT, Action.RIGHT, Action.UP, Action.DOWN]:
            result_coord = Maze.resulting_coord(coord, action)
            if self.valid_coord(result_coord):
                valid_actions.append(action)
        return valid_actions
