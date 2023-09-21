import random
import numpy as np

class MyEGreedy:

    def __init__(self):
        print("Made EGreedy")

    def get_random_action(self, agent, maze):
        possible_actions = maze.get_valid_actions(agent)
        random_action = random.choice(possible_actions)
        return random_action
        # TODO to select an action at random in State s

    def get_best_action(self, agent, maze, q_learning):
        possible_actions = maze.get_valid_actions(agent)
        action_values = q_learning.get_action_values(agent.get_state(maze), possible_actions)

        all_zero = True

        for value in action_values:
            if value != 0:
                all_zero = False

        if all_zero:
            return self.get_random_action(agent, maze)

        else:
            max_action_value = max(action_values)
            max_action_value_index = action_values.index(max_action_value)
            best_action = possible_actions[max_action_value_index]
            return best_action

    def get_egreedy_action(self, agent, maze, q_learning, epsilon):
        random_value = random.random()
        if (random_value < epsilon):
            return self.get_random_action(agent, maze)
        else:
            return self.get_best_action(agent, maze, q_learning)

        # TODO to select between random or best action selection based on epsilon.
