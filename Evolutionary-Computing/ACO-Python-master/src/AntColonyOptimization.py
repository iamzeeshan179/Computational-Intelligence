import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
# import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
from Maze import Maze
from PathSpecification import PathSpecification
from Ant import Ant

# Class representing the first assignment. Finds shortest path between two points in a maze according to a specific
# path specification.
class AntColonyOptimization:

    # Constructs a new optimization object using ants.
    # @param maze the maze .
    # @param antsPerGen the amount of ants per generation.
    # @param generations the amount of generations.
    # @param Q normalization factor for the amount of dropped pheromone
    # @param evaporation the evaporation factor.
    def __init__(self, maze, ants_per_gen, generations, q, evaporation):
        self.maze = maze
        self.ants_per_gen = ants_per_gen
        self.generations = generations
        self.q = q
        self.evaporation = evaporation

    # Loop that starts the shortest path process
    # @param spec Specification of the route we wish to optimize
    # @return ACO optimized route
    def find_shortest_route(self, path_specification):
        self.maze.reset()
        route_lengths = []
        final_route = None
        for i in range(self.generations):
            all_routes = []
            for j in range(self.ants_per_gen):
                ant = Ant(self.maze, path_specification, alpha=1, beta=0.5)
                route = ant.find_route()
                if final_route is None or route.shorter_than(final_route):
                    final_route = route
                all_routes.append(route)
            self.maze.evaporate(self.evaporation)
            self.maze.add_pheromone_routes(all_routes, self.q)
            avg_length = self.average_route_length(all_routes)
            route_lengths.append(avg_length)
            print("==> Generation: " + str(i + 1) + " completed"
                  + " avg:" + str(avg_length))
        print(route_lengths)
        self.plot_performance(route_lengths)
        return final_route

    def average_route_length(self, all_routes):
        lengths = []
        for route in all_routes:
            # Get route length per ant
            lengths.append(route.size())
        avg_length = sum(lengths) / self.ants_per_gen
        return avg_length

    def plot_performance(self, route_lengths):
        plt.plot(route_lengths)
        plt.xlabel("Generation")
        plt.ylabel("Avg route length")
        plt.title("Medium Maze=>"
                  + " Gens: " + str(self.generations)
                  + ", Ants: " + str(self.ants_per_gen)
                  + ", Q: " + str(self.q)
                  + ", Evaporation: " + str(self.evaporation)
                  )
        plt.show()
        plt.savefig("./../data/aco-performance.png")


# Driver function for Assignment 1
if __name__ == "__main__":
    # parameters
    ants_per_gen = 10
    gens = 100
    q = 200
    evap = 0.1
    diff = "medium"

    # construct the optimization objects
    maze = Maze.create_maze("./../data/" + diff + " maze.txt")
    spec = PathSpecification.read_coordinates("./../data/" + diff + " coordinates.txt")
    aco = AntColonyOptimization(maze, ants_per_gen, gens, q, evap)

    # save starting time
    start_time = int(round(time.time() * 1000))

    # run optimization
    shortest_route = aco.find_shortest_route(spec)

    # print time taken
    print("Time taken: " + str((int(round(time.time() * 1000)) - start_time) / 1000.0))

    # save solution
    shortest_route.write_to_file("./../data/" + diff + "_solution.txt")

    # print route size
    print("Route size: " + str(shortest_route.size()))