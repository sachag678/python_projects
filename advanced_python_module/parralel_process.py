"""Process Parallel."""
import numpy as np


def single_layer_perceptron(inputs, weights):
    """Multiply the weights with the inputs."""
    return np.dot(inputs, weights)

if __name__ == '__main__':
    list_of_weights = []
    for i in range(5):
            list_of_weights.append(np.random.rand(3, 1))
    inputs = np.array([3, 1, 2])
    output = []
    for i in range(5):
        output.append(single_layer_perceptron(inputs, list_of_weights[i]))

    average_weights = np.array(list_of_weights).mean(axis=0)

    print('Average weights:', single_layer_perceptron(inputs, average_weights))
    print('Average output:', np.array(output).mean())
