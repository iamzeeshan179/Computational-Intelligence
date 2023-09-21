import numpy as np
import matplotlib.pyplot as plt


# Perceptron class for single perceptron computation
class Perceptron:
    # Constructor for perceptron with learning rate and number of iterations
    def __init__(self, learning_rate=0.1, n_iterations=100):
        self.lr = learning_rate
        self.epochs = n_iterations
        self.weights = None
        self.bias = None

    # Sigmoid function for activation
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Activation step function - works better than sigmoid here
    def activation_function(self, x):
        if x >= 0:
            return 1
        else:
            return 0

    # Trains the model and return the mean square error costs
    def train(self, input_data, labels):
        # random samples of input size from a uniform distribution over (0,1)
        self.weights = np.random.rand(input_data.shape[1])
        self.bias = np.random.rand(1)

        costs = []
        for epoch in range(self.epochs):
            for idx, x_i in enumerate(input_data):
                self.update_params(x_i, labels[idx])
            costs.append(self.mse(input_data, labels))

        print("Final Weights:")
        print(self.weights)
        print("Final Bias:")
        print(self.bias)

        return costs

    # Update the parameters weights and bias
    def update_params(self, input_item, label):
        prediction = self.predict(input_item)
        error = label - prediction
        self.weights += self.lr * input_item * error
        self.bias += self.lr * error

    # Returns the mean squared error of prediction against expected label
    def mse(self, input_data, labels):
        predicted = []
        for idx, item in enumerate(input_data):
            predicted.append(self.predict(item))
        errors = np.array(predicted) - labels
        return np.mean(errors ** 2)

    # Makes a predictions based on the input item
    def predict(self, item):
        result = np.dot(self.weights, item) + self.bias
        # Use either sigmoid or step function here
        return self.activation_function(result)

    # Plots a graph of errors over epochs for given gate
    def plot_graph(self, errors, gate):
        plt.title("Perceptron for " + gate + ", learning rate = " + str(self.lr))
        plt.ylabel("Mean Squared Error")
        plt.xlabel("Epochs")
        plt.xticks(np.arange(len(errors)))
        plt.plot(errors)
        plt.show()


if __name__ == "__main__":
    input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_OR = np.array([0, 1, 1, 1])
    y_AND = np.array([0, 0, 0, 1])
    y_XOR = np.array([0, 1, 1, 0])

    OR_ptron = Perceptron(0.1, 25)
    OR_ptron.plot_graph(OR_ptron.train(input_data, y_OR), "OR")

    AND_ptron = Perceptron(0.1, 25)
    AND_ptron.plot_graph(AND_ptron.train(input_data, y_AND), "AND")

    XOR_ptron = Perceptron(0.1, 25)
    XOR_ptron.plot_graph(XOR_ptron.train(input_data, y_XOR), "XOR")
