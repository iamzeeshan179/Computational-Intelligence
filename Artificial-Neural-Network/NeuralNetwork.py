import numpy as np
import time


# Neural Network Class
class NeuralNetwork:
    # Constructor for NN
    def __init__(self, features, neurons, outputs, learning_rate, beta, iterations):
        self.lr = learning_rate
        self.epochs = iterations
        self.beta = beta
        self.W1 = np.random.rand(neurons, features)
        self.W2 = np.random.rand(outputs, neurons)
        self.b1 = np.random.rand(neurons)
        self.b2 = np.random.rand(outputs)

    # Random shuffling the training data every training epoch
    def shuffle_data(self, inputs, outputs, num_trains):
        np.random.seed(np.random.randint(num_trains))
        indices = np.random.permutation(num_trains)
        x_train, y_train = inputs[:, indices], outputs[:, indices]
        return x_train, y_train

    # Sigmoid activation function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Softmax function to return probabilities
    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0).reshape(1, -1)

    # Calculate Cross Entropy Loss for predicted against expect label
    def cross_entropy_loss(self, y, y_pred):
        return -(1.0 / y.shape[1]) * np.sum(np.multiply(y, np.log(y_pred)))

    # Forward-propagation
    def forward_prop(self, inputs):
        z1 = np.matmul(self.W1, inputs)
        a1 = self.sigmoid(z1)
        z2 = np.matmul(self.W2, a1)
        a2 = self.softmax(z2)
        return z1, a1, z2, a2

    # Backwards-propagation and update parameters
    def update(self, batch_size, A1, A2, V_dW1, V_dW2, X, Y):
        # Calculating the derivative of 𝐿 with respect to Z2
        dZ2 = A2 - Y

        # calculating the derivative of 𝐿 with respect to W2
        dW2 = (1.0 / batch_size) * np.matmul(dZ2, A1.T)

        # calculating the derivative of L with respect to A1
        dA1 = np.matmul((A2 - Y).T, self.W2)

        # Calculating the derivative of 𝐿 with respect to Z1
        dZ1 = dA1 * A1.T * (1 - A1).T

        # Calculating the derivative of 𝐿 with respect to W1
        dW1 = np.matmul(dZ1.T, X.T)

        # Update the learning velocities for SGD with momentum
        V_dW1 = self.beta * V_dW1 + (1 - self.beta) * dW1
        V_dW2 = self.beta * V_dW2 + (1 - self.beta) * dW2

        # Updating the model weights
        self.W1 = self.W1 - (self.lr * V_dW1)
        self.W2 = self.W2 - (self.lr * V_dW2)

    # Train the NN model
    def train(self, X_train, Y_train, X_validation, Y_validation, num_trains, num_batches, batch_size):
        # Logging the training loss in a list
        loss_log = []

        # Logging the validation accuracy in a list
        val_acc = []

        # Zeros initialize the momentum for SGD
        V_dW1 = np.zeros(self.W1.shape)
        V_dW2 = np.zeros(self.W2.shape)

        for i in range(self.epochs):
            start_t = time.time()

            # Random shuffling the training data every training epoch
            X_train_shuffled, Y_train_shuffled = self.shuffle_data(X_train, Y_train, num_trains)

            for j in range(num_batches):

                # Get mini-batch samples for training
                start_idx = j * batch_size  # start index of the batch size
                end_idx = min(j * batch_size + batch_size, X_train.shape[1] - 1)  # end index of the batch size
                X, Y = X_train_shuffled[:, start_idx: end_idx], Y_train_shuffled[:, start_idx: end_idx]

                # Size of actual mini-batch, it could be smaller than batch_size for the last batch
                mini_batch = end_idx - start_idx

                # Forward-propagation
                z1, a1, z2, a2 = self.forward_prop(X)

                # calculating the cross entropy loss
                loss = self.cross_entropy_loss(Y, a2)

                # appending the loss in the log
                loss_log.append(loss)

                # Back-propagate and update parameters
                self.update(mini_batch, a1, a2, V_dW1, V_dW2, X, Y)

                # if j % 100 == 0:
                #     print("[Epoch/Iterations]:[{}/{}], loss: {}".format(i, j, loss))

                if loss < 0.1:
                    break  # If loss<0.1 then exit loop

            # Get accuracy on validation set
            _, _, _, temp = self.forward_prop(X_validation)
            predictions = np.argmax(temp, axis=0)
            labels = np.argmax(Y_validation, axis=0)

            acc = np.array(labels == predictions).sum() / float(len(labels))
            val_acc.append(acc)
            # print("=> Elapsed time epoch #{} : {:.2f} seconds".format(i, time.time() - start_t))
            # print("=> Current Accuracy: {:.4f} ".format(acc))
        return val_acc, loss_log
