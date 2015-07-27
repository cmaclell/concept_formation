from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from concept_formation.trestle import TrestleTree
import matplotlib.pyplot as plt

# Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))

# Fit regression model
# Note, TRESTLE is order dependent, so the fit function shuffles the data.
# More stable results might be attained by increasing the number of iterations.
dtree = DecisionTreeRegressor(max_depth=2)
dtree.fit(X, y)
ttree = TrestleTree()
training_data = [{'x': float(X[i][0]), 'y': float(y[i])} for i,v in enumerate(X)]
ttree.fit(training_data)

# Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_dtree = dtree.predict(X_test)
y_trestle = [ttree.infer_missing({'x': float(v)})['y'] for i,v in
             enumerate(X_test)]

# Plot the results
plt.figure()
plt.scatter(X, y, c="k", label="Data")
plt.plot(X_test, y_trestle, c="g", label="TRESTLE", linewidth=2)
plt.plot(X_test, y_dtree, c="r", label="Decison Tree (Depth=2)", linewidth=2)
plt.xlabel("Data")
plt.ylabel("Target")
plt.title("TRESTLE/Decision Tree Regression")
plt.legend()
plt.show()
