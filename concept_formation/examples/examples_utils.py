"""
This module contains utility functions used in the example scripts. They are
implemented separately because they use scipy and numpy and we want to remove
external dependencies from within the core library.
"""
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
from math import sqrt

from scipy.stats import sem
from scipy.stats import t
from scipy import linalg
import numpy as np

from concept_formation.utils import mean

def moving_average(a, n=3) :
    """A function for computing the moving average, so that we can smooth out the
    curves on a graph.
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def lowess(x, y, f=1./3., iter=3, confidence=0.95):
    """
    Performs Lowess smoothing

    Code adapted from: https://gist.github.com/agramfort/850437

    lowess(x, y, f=2./3., iter=3) -> yest

    Lowess smoother: Robust locally weighted regression.
    The lowess function fits a nonparametric regression curve to a scatterplot.
    The arrays x and y contain an equal number of elements; each pair
    (x[i], y[i]) defines a data point in the scatterplot. The function returns
    the estimated (smooth) values of y.

    The smoothing span is given by f. A larger value for f will result in a
    smoother curve. The number of robustifying iterations is given by iter. The
    function will run faster with a smaller number of iterations.

    .. todo:: double check that the confidence bounds are correct
    """
    n = len(x)
    r = int(np.ceil(f*n))
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
    w = np.clip(np.abs((x[:,None] - x[None,:]) / h), 0.0, 1.0)
    w = (1 - w**3)**3
    yest = np.zeros(n)
    delta = np.ones(n)
    for iteration in range(iter):
        for i in range(n):
            weights = delta * w[:,i]
            b = np.array([np.sum(weights*y), np.sum(weights*y*x)])
            A = np.array([[np.sum(weights), np.sum(weights*x)],
                   [np.sum(weights*x), np.sum(weights*x*x)]])
            beta = linalg.solve(A, b)
            yest[i] = beta[0] + beta[1]*x[i]
 
        residuals = y - yest
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta**2)**2

    h = np.zeros(n)
    for x_idx, x_val in enumerate(x):
        r2 = np.array([v*v for i, v in enumerate(residuals) if x[i] == x_val])
        n = len(r2)
        se = sqrt(mean(r2)) / sqrt(len(r2))
        h[x_idx] = se * t._ppf((1+confidence)/2., n-1)

    return yest, yest-h, yest+h

def avg_lines(x, y, confidence=0.95):
    n = len(x)
    mean = np.zeros(n)
    lower = np.zeros(n)
    upper = np.zeros(n)

    for x_idx, x_val in enumerate(x):
        ys = np.array([v for i,v in enumerate(y) if x[i] == x_val])
        m,l,u = mean_confidence_interval(ys)
        mean[x_idx] = m
        lower[x_idx] = l
        upper[x_idx] = u

    return mean, lower, upper

def mean_confidence_interval(data, confidence=0.95):
    """
    Given a list or vector of data, this returns the mean, lower, and upper
    confidence intervals to the level of confidence specified (default = 95%
    confidence interval).
    """
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h
