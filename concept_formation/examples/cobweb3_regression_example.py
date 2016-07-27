from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from concept_formation.trestle import TrestleTree
import matplotlib.pyplot as plt
from random import seed

# Create a random dataset
rng = np.random.RandomState(1)
seed(0)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))

# Fit regression models (Decision Tree and TRESTLE)
# For TRESTLE the y attribute is hidden, so only the X is used to make
# predictions. 
dtree = DecisionTreeRegressor(max_depth=3)
dtree.fit(X, y)
ttree = TrestleTree()
training_data = [{'x': float(X[i][0]), '_y': float(y[i])} for i,v in
                 enumerate(X)]
ttree.fit(training_data, iterations=1)

# Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_dtree = dtree.predict(X_test)
y_trestle = [ttree.categorize({'x': float(v)}).predict('_y') for v in X_test]

# Plot the results
plt.figure()
plt.scatter(X, y, c="k", label="Data")
plt.plot(X_test, y_trestle, c="g", label="TRESTLE", linewidth=2)
plt.plot(X_test, y_dtree, c="r", label="Decison Tree (Depth=3)", linewidth=2)
plt.xlabel("Data")
plt.ylabel("Target")
plt.title("TRESTLE/Decision Tree Regression")
plt.legend(loc=3)
plt.show()
