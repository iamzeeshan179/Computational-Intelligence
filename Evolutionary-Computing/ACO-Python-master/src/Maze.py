import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import numpy as np
from Coordinate import Coordinate
from Direction import Direction
from SurroundingPheromone import SurroundingPheromone

# Class that holds all the maze data. This means the pheromones, the open and blocked tiles in the system as
# well as the starting and end coordinates.
class Maze:

    # Constructor of a maze
    # @param walls int array of tiles accessible (1) and non-accessible (0)
    # @param width width of Maze (horizontal)
    # @param length length of Maze (vertical)
    def __init__(self, walls, width, length):
        self.walls = walls
        self.length = length
        self.width = width
        self.start = None
        self.end = None
        self.pheromones = np.ones([self.width, self.length], dtype=object)
        self.initialize_pheromones()

    # Initialize pheromones to a start value.
    # move[direction] is 1 if accessible, 0 when inaccessible
    def initialize_pheromones(self):
        for i in range(self.width):
            for j in range(self.length):
                curr_pos = Coordinate(i, j)
                move = {}
                for direction in Direction:
                    next_pos = curr_pos.add_direction(direction)
                    if self.in_bounds(next_pos):
                        move[direction] = self.walls[next_pos.x][next_pos.y]
                    else:
                        move[direction] = 0
                self.leave_pheromones(curr_pos, move)
        return

    # Reset the maze for a new shortest path problem.
    def reset(self):
        self.initialize_pheromones()

    # Leave initial pheromones at accessible positions
    def leave_pheromones(self, pos, move):
        self.pheromones[pos.x][pos.y] = SurroundingPheromone(move[Direction.north],
                                                             move[Direction.east],
                                                             move[Direction.south],
                                                             move[Direction.west])

    # Update the pheromones along a certain route according to a certain Q
    # @param r The route of the ants
    # @param Q Normalization factor for amount of dropped pheromone
    def add_pheromone_route(self, route, q):
        pos = route.start
        for direction in route.route:
            surrounding_pheromone = self.pheromones[pos.x][pos.y]
            new_ph = (q / len(route.route)) + surrounding_pheromone.get(direction)
            surrounding_pheromone.set(direction, new_ph)
            pos = pos.add_direction(direction)
        return

    # Update pheromones for a list of routes
    # @param routes A list of routes
    # @param Q Normalization factor for amount of dropped pheromone
    def add_pheromone_routes(self, routes, q):
        for r in routes:
            self.add_pheromone_route(r, q)

    # Evaporate pheromone
    # @param rho evaporation factor
    def evaporate(self, rho):
        for i in range(self.width):
            for j in range(self.length):
                surrounding_pheromone = self.pheromones[i][j]
                for direction in Direction:
                    new_ph = (1 - rho) * surrounding_pheromone.get(direction)
                    surrounding_pheromone.set(direction, new_ph)
        return

    # Width getter
    # @return width of the maze
    def get_width(self):
        return self.width

    # Length getter
    # @return length of the maze
    def get_length(self):
        return self.length

    # Returns a the amount of pheromones on the neighbouring positions (N/S/E/W).
    # @param position The position to check the neighbours of.
    # @return the pheromones of the neighbouring positions.
    def get_surrounding_pheromone(self, position):
        return self.pheromones[position.x][position.y]

    # Pheromone getter for a specific position. If the position is not in bounds returns 0
    # @param pos Position coordinate
    # @return pheromone at point
    def get_pheromone(self, pos):
        return self.pheromones[pos.x][pos.y].total_surrounding_pheromone

    # Check whether a coordinate lies in the current maze.
    # @param position The position to be checked
    # @return Whether the position is in the current maze
    def in_bounds(self, position):
        return position.x_between(0, self.width) and position.y_between(0, self.length)

    # Representation of Maze as defined by the input file format.
    # @return String representation
    def __str__(self):
        string = ""
        string += str(self.width)
        string += " "
        string += str(self.length)
        string += " \n"
        for y in range(self.length):
            for x in range(self.width):
                string += str(self.walls[x][y])
                string += " "
            string += "\n"
        return string

    # Method that builds a mze from a file
    # @param filePath Path to the file
    # @return A maze object with pheromones initialized to 0's inaccessible and 1's accessible.
    @staticmethod
    def create_maze(file_path):
        try:
            f = open(file_path, "r")
            lines = f.read().splitlines()
            dimensions = lines[0].split(" ")
            width = int(dimensions[0])
            length = int(dimensions[1])

            # make the maze_layout
            maze_layout = []
            for x in range(width):
                maze_layout.append([])

            for y in range(length):
                line = lines[y + 1].split(" ")
                for x in range(width):
                    if line[x] != "":
                        state = int(line[x])
                        maze_layout[x].append(state)
            print("Ready reading maze file " + file_path)
            return Maze(maze_layout, width, length)
        except FileNotFoundError:
            print("Error reading maze file " + file_path)
            traceback.print_exc()
            sys.exit()
