from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from random import seed

from concept_formation.examples.examples_utils import avg_lines
from concept_formation.evaluation import incremental_evaluation
from concept_formation.trestle import TrestleTree
from concept_formation.dummy import DummyTree
from concept_formation.datasets import load_quadruped
from concept_formation.preprocessor import ObjectVariablizer

seed(0)

num_runs = 5
num_examples = 15
animals = load_quadruped(num_examples)

variablizer = ObjectVariablizer()
animals = [variablizer.transform(t) for t in animals]

for animal in animals:
    animal['type'] = animal['_type']
    del animal['_type']

naive_data = incremental_evaluation(DummyTree(), animals,
                                  run_length=num_examples,
                                  runs=num_runs, attr="type")
trestle_data = incremental_evaluation(TrestleTree(), animals,
                                  run_length=num_examples,
                                  runs=num_runs, attr="type")

trestle_x, trestle_y = [], []
naive_x, naive_y = [], []
human_x, human_y = [], []

for opp in range(len(trestle_data[0])):
  for run in range(len(trestle_data)):
    trestle_x.append(opp)
    trestle_y.append(trestle_data[run][opp])

for opp in range(len(naive_data[0])):
  for run in range(len(naive_data)):
    naive_x.append(opp)
    naive_y.append(naive_data[run][opp])

trestle_x = np.array(trestle_x)
trestle_y = np.array(trestle_y)
naive_x = np.array(naive_x)
naive_y = np.array(naive_y)

trestle_y_avg, _, _ = avg_lines(trestle_x, trestle_y)
naive_y_avg, _, _ = avg_lines(naive_x, naive_y)

plt.plot(trestle_x, trestle_y_avg, label="TRESTLE", color="green")
plt.plot(naive_x, naive_y_avg, label="Naive Predictor", color="red")

plt.gca().set_ylim([0.00,1.0])
plt.gca().set_xlim([0,max(naive_x)-1])
plt.title("Incremental Quadruped Prediction Accuracy")
plt.xlabel("# of Training Examples")
plt.ylabel("Avg. Probability of True Quadruped Type (Accuracy)")
plt.legend(loc=4)

plt.show()
