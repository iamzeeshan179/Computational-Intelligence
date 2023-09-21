import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import random
from Route import Route
from Direction import Direction


# Class that represents the ants functionality.
class Ant:

    # Constructor for ant taking a Maze and PathSpecification.
    # @param maze Maze the ant will be running in.
    # @param spec The path specification consisting of a start coordinate and an end coordinate.
    def __init__(self, maze, path_specification, alpha, beta):
        self.maze = maze
        self.start = path_specification.get_start()
        self.end = path_specification.get_end()
        self.current_position = self.start
        self.rand = random
        self.alpha = alpha
        self.beta = beta
        # locally keep track of visited coordinates by ant
        self.visited = set()

    # Method that performs a single run through the maze by the ant.
    # @return The route the ant found through the maze.
    def find_route(self):
        route = Route(self.start)
        pheromones = self.maze.pheromones
        pos = self.start
        self.visited.add(pos)
        # Traverse maze until it reaches finishing point
        while pos != self.end:
            surrounding_pheromone = pheromones[pos.x][pos.y]
            next_dir = self.get_next_direction_not_visited(surrounding_pheromone, pos)
            if next_dir is None:
                # if it has to return to a visited node
                next_dir = self.get_next_direction(surrounding_pheromone, pos)

            route.add(next_dir)
            pos = pos.add_direction(next_dir)
            self.visited.add(pos)
        return route

    # Get next direction for the ant to move without excluding the visited directions
    # @return The selected direction based on probabilities
    def get_next_direction(self, surrounding_pheromone, pos):
        directions = self.get_dir_array(pos, remove_visited_dir=False)
        weights = self.get_direction_weights(surrounding_pheromone, pos, directions)
        selected_direction = random.choices(directions, weights)[0]
        return selected_direction

    # Get next direction that has not yet been visited by the ant
    # if no such direction then empty the visited set to find route without loop
    # @return The selected direction based on probabilities
    def get_next_direction_not_visited(self, surrounding_pheromone, pos):
        directions = self.get_dir_array(pos)
        weights = self.get_direction_weights(surrounding_pheromone, pos, directions)
        if sum(weights) == 0:
            return self.visited.clear()
        selected_direction = random.choices(directions, weights)[0]
        return selected_direction

    # Returns an array for movable directions
    # @param remove_visited_dir bool To remove visited directions
    # @return An array for movable directions
    def get_dir_array(self, pos, remove_visited_dir=True):
        directions = []
        for direction in Direction:
            if remove_visited_dir:
                position = pos.add_direction(direction)
                if position not in self.visited:
                    directions.append(direction)
            else:
                directions.append(direction)
        return directions

    # Returns an array of weights for every direction
    def get_direction_weights(self, surrounding_pheromone, pos, directions):
        weights = []
        for direction in directions:
            prob = self.calculate_direction_probability(surrounding_pheromone, pos, direction)
            weights.append(prob)
        return weights

    # Calculate direction probability given surrounding pheromones
    # Probability of 1 when end point reached
    # @return probability value of selecting a given direction
    def calculate_direction_probability(self, surrounding_pheromone, pos, direction):
        t = surrounding_pheromone.get(direction)  # get value[dir] 1
        next_pos = pos.add_direction(direction)  # move
        dist = next_pos.get_distance(self.end)  # distance from next_pos to finishing position

        if dist == 0:
            return 1

        n = 1/dist
        return (t ** self.alpha) * (n ** self.beta)
