from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
from random import shuffle
from random import seed

import numpy as np
import matplotlib.pyplot as plt

from concept_formation.cobweb import CobwebTree
from concept_formation.cobweb3 import Cobweb3Tree


# Generate sample data
np.random.seed(0)
seed(0)
X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel()
y2 = np.sin(X).ravel()

T = np.linspace(0, 5, 50)[:, np.newaxis]
T_y = np.sin(T).ravel()

# Add noise to targets
y[::5] += 1 * (0.5 - np.random.rand(8))
y2[::5] += 1 * (0.5 - np.random.rand(8))

# Create dictionaries
# Note that the y value is stored as a hidden variable because
# in this case we only want to use the X value to make predictions.
training_data = [{'X': v[0], '_y': y[i]} for i, v in enumerate(X)]
shuffle(training_data)

# Build test data
test_data = [{'X': v[0]} for i, v in enumerate(T)]
# test_data = [{'X': float(v)} for i,v in enumerate(X)]

# Fit cobweb models
cbt = CobwebTree()
cb3t = Cobweb3Tree()

cbt.fit(training_data, iterations=1)
cb3t.fit(training_data, iterations=1)
# print(cb3t.root)

child = cb3t.categorize({'X': 4.16})
# print(child.predict('X'))
# print(child.predict('y'))

curr = child
# print(curr)
while curr.parent is not None:
    curr = curr.parent
    # print(curr)

# Predict test data
cby = [cbt.categorize(e).predict('_y') for e in test_data]
cb3y = [cb3t.categorize(e).predict('_y') for e in test_data]

# Plot the results
plt.scatter(X, y, c='k', label='training data')
plt.plot(T, cby, c='g', label='Cobweb')
plt.plot(T, cb3y, c='b', label='Cobweb3')
plt.axis('tight')
plt.legend(loc=3)
plt.title("COBWEB and COBWEB3 Regressors")

plt.show()
