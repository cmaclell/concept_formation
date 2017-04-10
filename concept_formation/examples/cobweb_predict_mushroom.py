from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from random import seed

from concept_formation.examples.examples_utils import avg_lines
from concept_formation.evaluation import incremental_evaluation
from concept_formation.cobweb import CobwebTree
from concept_formation.dummy import DummyTree
from concept_formation.datasets import load_mushroom

seed(0)
num_runs = 30 
num_examples = 30 
mushrooms = load_mushroom()

naive_data = incremental_evaluation(DummyTree(), mushrooms,
                                  run_length=num_examples,
                                  runs=num_runs, attr="classification")
cobweb_data = incremental_evaluation(CobwebTree(), mushrooms,
                                  run_length=num_examples,
                                  runs=num_runs, attr="classification")

cobweb_x, cobweb_y = [], []
naive_x, naive_y = [], []

for opp in range(len(cobweb_data[0])):
  for run in range(len(cobweb_data)):
    cobweb_x.append(opp)
    cobweb_y.append(cobweb_data[run][opp])

for opp in range(len(naive_data[0])):
  for run in range(len(naive_data)):
    naive_x.append(opp)
    naive_y.append(naive_data[run][opp])

cobweb_x = np.array(cobweb_x)
cobweb_y = np.array(cobweb_y)
naive_x = np.array(naive_x)
naive_y = np.array(naive_y)

cobweb_y_smooth, cobweb_lower_smooth, cobweb_upper_smooth = avg_lines(cobweb_x, cobweb_y)
naive_y_smooth, naive_lower_smooth, naive_upper_smooth = avg_lines(naive_x, naive_y)

plt.fill_between(cobweb_x, cobweb_lower_smooth, cobweb_upper_smooth, alpha=0.5,
                 facecolor="green")
plt.fill_between(naive_x, naive_lower_smooth, naive_upper_smooth, alpha=0.5,
                 facecolor="red")

plt.plot(cobweb_x, cobweb_y_smooth, label="COBWEB", color="green")
plt.plot(naive_x, naive_y_smooth, label="Naive Predictor", color="red")

plt.gca().set_ylim([0.0,1.0])
plt.gca().set_xlim([0,max(naive_x)-1])
plt.title("Incremental Mushroom Edibility Prediction Accuracy")
plt.xlabel("# of Training Examples")
plt.ylabel("Avg. Probability of True Class (Accuracy)")
plt.legend(loc=4)

plt.show()

