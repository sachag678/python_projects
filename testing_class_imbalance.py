"""Testing."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm


def generate_data_from_two_different_gaussians(majority, minority):
    """
    Generate data from two differnt gaussians and labels them with 1 and 0.

    The dataset is imbalanced with 1000, and 100 for the maj and min class.
    """
    dist1 = np.random.normal(0, 1, [minority, 1])
    dist2 = np.random.normal(3, 2, [majority, 1])

    x = np.concatenate((dist1, dist2))
    zeros = np.zeros((minority, 1))
    ones = np.ones((majority, 1))

    y = np.concatenate((zeros, ones))
    # plt.hist(dist1, normed=True)
    # plt.hist(dist2, normed=True)
    # plt.show()

    data = np.concatenate((x, y), axis=1)
    df = pd.DataFrame(data, columns=['X', 'Y'])
    return df


def learn_gaussian_params(data):
    """Learn params of a Gaussian Distribution."""
    mu = np.array(data).mean()
    sigma = np.array(data).std()
    return mu, sigma


def likelihood(datapoint, mu, sigma):
    """Calculate the likelihood."""
    return norm.pdf(datapoint, mu, sigma)

if __name__ == '__main__':
    major = 1000
    minor = 100
    data = generate_data_from_two_different_gaussians(major, minor)

    mu0, sigma0 = learn_gaussian_params(data['X'][0:minor - 1])
    mu1, sigma1 = learn_gaussian_params(data['X'][minor:major + minor - 1])

    datapoint = 2.4
    probxgiven0 = likelihood(datapoint, mu0, sigma0)
    probxgiven1 = likelihood(datapoint, mu1, sigma1)

    prob0 = minor / (major + minor)
    prob1 = major / (major + minor)

    print(probxgiven0 * prob0, probxgiven1 * prob1)
