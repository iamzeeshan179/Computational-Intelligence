from NeuralNetwork import NeuralNetwork
import numpy as np
# import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Import Files
file_features = open("data/features.txt")
file_targets = open("data/targets.txt")
file_test_features = open("data/unknown.txt")
inputs = np.loadtxt(file_features, delimiter=",").T
outputs = (np.loadtxt(file_targets, delimiter=" ", dtype="int").reshape(1, -1) - 1)[0]  # adjust index
test_features = np.loadtxt(file_test_features, delimiter=",").T

# Data-dividing fractions
test_fraction = 0.1  # Feel free to change
validation_fraction = 0.1  # Feel free to change
training_fraction = 1 - test_fraction - validation_fraction


# One-hot vector generator
# For each unique value in the original categorical column, a new column is created in this method.
# These dummy variables are then filled up with zeros and ones.
def one_hot_vector(x, num_classes):
    return np.eye(num_classes)[x].T


# Plots a graph
def plot_graph(arr, title, x_label, y_label, file_name):
    plt.title(title)
    plt.plot(arr)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(file_name)
    plt.show()


# Plots a graph for NN performance
def plot_performance(x_arr, y_arr, title, x_label, y_label, file_name):
    plt.title(title)
    plt.plot(x_arr, y_arr)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(np.arange(len(x_arr)))
    plt.savefig(file_name)
    plt.show()


# Makes a prediction for the trained NN based on test data
def make_predictions(NN, test_data, submission=False):
    # Run forward pass and predictions on test data
    _, _, _, A2 = NN.forward_prop(test_data)

    if not submission:
        predictions = np.argmax(A2, axis=0)  # Returns 0 based index
    else:
        predictions = (np.argmax(A2, axis=0) + 1)  # Return original class index

    return predictions


# Generates a confusion matrix for validation set
def confusionMatrix(labels, predictions):
    k = len(np.unique(labels))
    matrix = np.zeros((k, k))

    for i in range(len(labels)):
        matrix[labels[i]][predictions[i]] += 1

    print("Confusion Matrix:")
    return matrix


# Tests the NN on unseen data and its expected labels
# Returns a confusion matrix and accuracy score for the trained model
def predict_and_score(NN, x_test, y_test):
    predictions = make_predictions(NN, x_test, submission=False)
    labels = np.argmax(y_test, axis=0)

    print("Running test_data")
    print(confusionMatrix(labels, predictions))
    acc = np.array(labels == predictions).sum() / float(len(labels))
    print("Accuracy is: " + str(acc))
    return acc


# Takes the split arrays and returns the train and test sets
# holdout is the index extracted for test
def partition(training, target, holdout):
    # Used for Test
    x_test = training[holdout]
    y_test = target[holdout]

    x_temp = np.delete(training, holdout)
    y_temp = np.delete(target, holdout)

    # test rows removed and rest is used for training
    x_train = np.concatenate(x_temp)
    y_train = np.concatenate(y_temp)

    return x_train.T, y_train.T, x_test.T, y_test.T


# K Fold Cross-Validation to optimize model by finding neurons with best results
# Plots a graph of NN performance against number of neurons
def k_fold(training_data, target_data, start, end, k):
    x = np.array_split(training_data.T, k)
    y = np.array_split(target_data.T, k)

    final_accuracies = []
    neurons = np.arange(start, end + 1)

    for i in neurons:
        accuracies = []
        for j in range(k):
            nn = NeuralNetwork(features=10, neurons=i, outputs=7, learning_rate=0.1, beta=0.5,
                               iterations=200)
            x_train, y_train, x_validate, y_validate = partition(x, y, j)

            # NN parameters
            batch_size = 128
            num_trains = len(x[0])  # Training inputs
            num_batches = num_trains // batch_size

            acc, losslog = nn.train(x_train, y_train, x_validate, y_validate, num_trains, num_batches,
                                    batch_size)
            accuracies.append(sum(acc) / len(acc))
        print("=> Completed fold for {} neurons, accuracy: {:.4f}".format(i, accuracies[-1]))
        final_accuracies.append(sum(accuracies) / len(accuracies))

    best_accuracy_idx = np.argmax(final_accuracies)
    best_neurons = neurons[best_accuracy_idx]
    print("=> Best fit was {} Neurons with avg accuracy: {:.4f}".format(best_neurons,
                                                                        final_accuracies[best_accuracy_idx]))
    plot_performance(neurons, final_accuracies, "NN performance", "Neurons", "Avg Accuracy", "plots/nn-performance.png")


def repeated_testing(NN, x_train, y_train, x_test, y_test, k):
    accuracies = []
    batch_size = 128
    num_trains = X_train.shape[1]
    num_batches = num_trains // batch_size
    trials = np.arange(1, k + 1)
    for i in range(k):
        acc, losslog = NN.train(x_train, y_train, x_test, y_test, num_trains, num_batches, batch_size)
        accuracies.append(sum(acc) / len(acc))
    print("mean accuracy: " + str(np.mean(accuracies)))
    plot_performance(trials, accuracies, "Test set Accuracy", "Trial#", "Avg Accuracy",
                     "plots/accuracy_test_set.png")


# Save the predictions in a suitable format for submission
def save_predictions(predictions):
    np.savetxt('Group_61_classes.txt', [predictions], delimiter=',', fmt='%d')


if __name__ == "__main__":
    # Split the Data
    split_train = int(inputs.shape[1] * training_fraction)
    split_test = int(inputs.shape[1] * (1 - test_fraction))

    X_train, X_validation, X_test = inputs[:, 0:split_train], inputs[:, split_train:split_test], inputs[:, split_test:]
    Y_train, Y_validation, Y_test = one_hot_vector(outputs[0: split_train], 7), one_hot_vector(
        outputs[split_train:split_test], 7), one_hot_vector(outputs[split_test:], 7)

    # This is the batch size we will use during training for SGD
    batch_size = 128
    # this is the number of training elements we have
    num_trains = X_train.shape[1]
    # Calculating the number of training iterations base on the number of training samples and the batch size
    num_batches = num_trains // batch_size

    # # Ex 11 - Run K-fold cross validation to adjust hyper-parameters
    # # Neurons = Range(7, 30), K = 10
    # k_fold(X_train, Y_train, 7, 30, 10)

    # # Training - This part runs a basic test instance of Neural Network
    test_NN = NeuralNetwork(features=10, neurons=8, outputs=7, learning_rate=0.01, beta=0.5,
                            iterations=100)

    accuracy, loss_log = test_NN.train(X_train, Y_train, X_validation, Y_validation, num_trains, num_batches,
                                       batch_size)
    print("----Initial Training Completed-----")

    plot_graph(accuracy, "Validation Accuracy", "Epoch", "Accuracy", "plots/accuracy_before_tuning.jpg")
    # plot_graph(loss_log, "Cross Entropy Loss per Batch", "Batch", "Cross Entropy Loss", "plots/loss_before_tuning.jpg")

    # Returns confusion matrix and accuracy of model
    predict_and_score(test_NN, X_test, Y_test)  # Accuracy ~0.84

    # # Optimization - Runs a Tuned instance of Neural Network
    optimized_NN = NeuralNetwork(features=10, neurons=22, outputs=7, learning_rate=0.01, beta=0.5,
                            iterations=200)

    # repeated_testing(test_NN, X_train, Y_train, X_validation, Y_validation, 10)
    accuracy, loss_log = optimized_NN.train(X_train, Y_train, X_test, Y_test, num_trains, num_batches, batch_size)
    print("----Optimized Training Completed-----")
    print(accuracy[-1])
    plot_graph(accuracy, "Test Set Accuracy", "Epoch", "Accuracy", "plots/accuracy_test.jpg")

    # # Evaluation - Returns confusion matrix and accuracy of model
    predict_and_score(optimized_NN, X_test, Y_test)  # Accuracy ~0.93

    # # Ex 15 - Make Predictions on unknown classes
    predictions = make_predictions(optimized_NN, test_features, submission=True)
    save_predictions(predictions)
