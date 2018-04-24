"""Process Parallel."""
import numpy as np


def single_layer_perceptron(inputs, weights):
    """Multiply the weights with the inputs."""
    return np.dot(inputs, weights)


def multi_layer_perceptron(inputs, weights1, weights2):
    """Multi layer perceptron."""
    layer1 = np.dot(inputs, weights1)
    layer1 = np.reshape(layer1, [3, 1])
    return np.dot(weights2, layer1)


def test_slp_wagging_vs_bagging():
    """Test SLP."""
    list_of_weights = []
    for i in range(5):
            list_of_weights.append(np.random.rand(3, 1))
    inputs = np.array([3, -5, 2])
    output = []
    for i in range(5):
        output.append(single_layer_perceptron(inputs, list_of_weights[i]))

    average_weights = np.array(list_of_weights).mean(axis=0)

    print('Single Layer Perceptron -----------------------------------------')
    print('Average weights:', single_layer_perceptron(inputs, average_weights)[0])
    print('Average output:', np.array(output).mean())
    print('Makes sense, since it is a linear function.')


def test_mlp_wagging_vs_bagging():
    """Test MLP."""
    print('Multi Layer Perceptron---------------------------------------')
    inputs = np.array([3, -5, 2])
    w1 = []
    w2 = []
    for i in range(5):
            w1.append(np.random.rand(3, 3))
            w2.append(np.random.rand(1, 3))

    output = [multi_layer_perceptron(inputs, w1[i], w2[i]) for i in range(5)]

    average_w1 = np.array(w1).mean(axis=0)
    average_w2 = np.array(w2).mean(axis=0)

    print('Average weights:', multi_layer_perceptron(inputs, average_w1, average_w2)[0][0])
    print('Average output:', np.array(output).mean())

if __name__ == '__main__':
    test_slp_wagging_vs_bagging()
    test_mlp_wagging_vs_bagging()
