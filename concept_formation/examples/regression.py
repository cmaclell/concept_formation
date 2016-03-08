from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
from random import shuffle

import numpy as np
import matplotlib.pyplot as plt

from concept_formation.cobweb import CobwebTree
from concept_formation.cobweb3 import Cobweb3Tree

# Generate sample data
np.random.seed(0)
X = np.sort(5 * np.random.rand(40, 1), axis=0)
T = np.linspace(0, 5, 500)[:, np.newaxis]
T_y = np.sin(T).ravel()
y = np.sin(X).ravel()

# Add noise to targets
y[::5] += 1 * (0.5 - np.random.rand(8))

# Create dictionaries
# Note that the y value is stored as a hidden variable because
# in this case we only want to use the X value to make predictions.
training_data = [{'X': v[0], '_y': y[i]} for i,v in enumerate(X)]
shuffle(training_data)

# Build test data
test_data = [{'X': v[0]} for i,v in enumerate(T)]

# Fit cobweb models
cbt = CobwebTree()
cb3t = Cobweb3Tree()
cbt.fit(training_data, iterations=1)
cb3t.fit(training_data, iterations=1)

# Predict test data
cby = [cbt.categorize(e).predict('_y') for e in test_data]
cb3y = [cb3t.categorize(e).predict('_y') for e in test_data]

# Plot the results
plt.scatter(X, y, c='k', label='training data')
plt.plot(T, cby, c='g', label='Cobweb')
plt.plot(T, cb3y, c='b', label='Cobweb3')
plt.axis('tight')
plt.legend(loc=3)
plt.title("TrestleRegressor")

plt.show()
