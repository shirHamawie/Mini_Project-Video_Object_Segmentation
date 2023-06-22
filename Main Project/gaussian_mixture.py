from scipy.stats import multivariate_normal
from dataclasses import dataclass
import numpy as np


@dataclass
class GaussianMixture:
    def __init__(self, means, covariances, weights=None):
        self.means = means
        self.covariances = covariances
        self.weights = weights if weights else [1/len(means)]*len(means)
        self.gaussians = [multivariate_normal(mean, covariance) for mean, covariance in zip(means, covariances)]

    def pdf(self, x):
        pdfs = np.array([weight * gaussian.pdf(x) for weight, gaussian in zip(self.weights, self.gaussians)])
        return np.sum(pdfs)

    def max_pdf(self, x):
        pdfs = np.array([gaussian.pdf(x) for gaussian in self.gaussians])
        return np.max(pdfs)


def test_gaussian_mixture_pdf():
    # Test case 1: Single Gaussian
    means = [[0, 0]]
    covariances = [[[1, 0], [0, 1]]]
    weights = [1]
    gm = GaussianMixture(means, covariances, weights)
    x = [1, 1]
    expected_pdf = multivariate_normal(means[0], covariances[0]).pdf(x)
    assert np.isclose(gm.pdf(x), expected_pdf)

    # Test case 2: Multiple Gaussians
    means = [[0, 0], [1, 1], [-1, -1]]
    covariances = [[[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]]
    weights = [0.5, 0.3, 0.2]
    gm = GaussianMixture(means, covariances, weights)
    x = [1, 1]
    expected_pdf = sum([weight * multivariate_normal(mean, covariance).pdf(x) for weight, mean, covariance in zip(weights, means, covariances)])
    assert np.isclose(gm.pdf(x), expected_pdf)


if __name__ == '__main__':
    pass
