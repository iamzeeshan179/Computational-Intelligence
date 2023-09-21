from Maze import Maze
from Agent import Agent
from MyQLearning import MyQLearning
from MyEGreedy import MyEGreedy
import sys
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# Get Average Steps per Trial for every run
def get_trial_averages(data):
    avg = []
    for trials in data.items():
        totalSteps = sum(trials[1])
        maxTrials = len(trials[1])
        avg.append(totalSteps / maxTrials)
    return avg


def plot_trial_averages(averages):
    plt.plot(averages)
    plt.title('Trials vs 10 run averages (Toy Maze), alpha: ' + str(alfa))
    plt.xlabel('Trials')
    plt.ylabel('Average Steps')
    plt.savefig('graph1.jpg')
    plt.show()


def plot_rewards(reward_5, reward_10):
    print("reward_5:" + str(rewards_5))
    print("reward_10:" + str(rewards_10))
    plt.title("Reward capturing, gamma:" + str(gamma))
    x = [5, 10]
    y = [reward_5, reward_10]
    plt.barh(x, y)
    plt.yticks(x)
    plt.ylabel("Reward value")
    plt.xlabel("Counts")
    plt.show()


if __name__ == "__main__":
    # load the maze
    file = "../data/toy_maze.txt"
    maze = Maze(file)

    # Set the reward at the bottom right to 10
    # toy maze
    maze.set_reward(maze.get_state(9, 9), 10)
    # maze.set_reward(maze.get_state(24, 14), 10)

    # ex 2.3.1 set reward at top right to 5
    maze.set_reward(maze.get_state(9, 0), 5)

    # create a robot at starting and reset location (0,0) (top left)
    robot = Agent(0, 0)

    # make a selection object (you need to implement the methods in this class)
    selection = MyEGreedy()

    # make a Qlearning object (you need to implement the methods in this class)
    learn = MyQLearning()

    # Parameters
    NUM_ITER = 10  # runs
    alfa = 0.7
    gamma = 0.81  # <0.8 favor reward 5
    epsilon = 1
    beta = 0.99

    rewards_5 = 0  # counter for reward of value 5 captured by agent
    rewards_10 = 0  # counter for reward of value 10 captured by agent
    data = {}  # stores steps taken per trial for every run

    for iter in range(NUM_ITER):  # runs
        total_steps = 0
        trial_count = 0
        step = 0
        epsilon = 1

        # Agent Cycle
        while step < 30000:
            current_state = robot.get_state(maze)

            # random action or a best action (determined probabilistically by the value of epsilon)
            action = selection.get_egreedy_action(robot, maze, learn, epsilon)

            # perform action and update Q-Table
            next_state = robot.do_action(action, maze)
            next_possible_actions = maze.get_valid_actions(robot)
            reward = maze.get_reward(next_state)
            learn.update_q(current_state, action, reward, next_state, next_possible_actions,
                           alfa, gamma)

            # agent finds a reward, this completes a trial
            if reward != 0:
                if reward == 10:
                    rewards_10 += 1
                if reward == 5:
                    rewards_5 += 1
                trial_count += 1  # trial completed
                goal_steps = robot.nr_of_actions_since_reset  # steps taken to reach goal

                # store steps taken per trial
                if trial_count not in data:
                    data[trial_count] = []  # create a list if trial data does not exist
                data[trial_count].append(goal_steps)

                robot.reset()

                # update epsilon to reduce exploration factor
                epsilon = epsilon * beta

            step += 1

    avg = get_trial_averages(data)
    plot_trial_averages(avg)
    plot_rewards(rewards_5, rewards_10)
