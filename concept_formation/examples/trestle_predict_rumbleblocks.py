from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
import matplotlib.pyplot as plt
from random import seed
import numpy as np

from concept_formation.examples.examples_utils import lowess
from concept_formation.evaluation import incremental_evaluation
from concept_formation.trestle import TrestleTree
from concept_formation.dummy import DummyTree
from concept_formation.datasets import load_rb_s_07
from concept_formation.datasets import load_rb_s_07_human_predictions
from concept_formation.preprocessor import ObjectVariablizer

seed(5)

num_runs = 30
num_examples = 29
towers = load_rb_s_07()

variablizer = ObjectVariablizer()
towers = [variablizer.transform(t) for t in towers]

naive_data = incremental_evaluation(DummyTree(), towers,
                                    run_length=num_examples,
                                    runs=num_runs, attr="success")
cobweb_data = incremental_evaluation(TrestleTree(), towers,
                                     run_length=num_examples,
                                     runs=num_runs, attr="success")

human_data = []
key = None
human_predictions = load_rb_s_07_human_predictions()
for line in human_predictions:
    line = line.rstrip().split(",")
    if key is None:
        key = {v: i for i, v in enumerate(line)}
        continue
    x = int(line[key['order']])-1
    y = (1 - abs(int(line[key['correctness']]) -
                 int(line[key['prediction']])))
    human_data.append((x, y))

human_data.sort()

cobweb_x, cobweb_y = [], []
naive_x, naive_y = [], []
human_x, human_y = [], []

for opp in range(len(cobweb_data[0])):
    for run in range(len(cobweb_data)):
        cobweb_x.append(opp)
        cobweb_y.append(cobweb_data[run][opp])

for opp in range(len(naive_data[0])):
    for run in range(len(naive_data)):
        naive_x.append(opp)
        naive_y.append(naive_data[run][opp])

for x, y in human_data:
    human_x.append(x)
    human_y.append(y)

cobweb_x = np.array(cobweb_x)
cobweb_y = np.array(cobweb_y)
naive_x = np.array(naive_x)
naive_y = np.array(naive_y)
human_x = np.array(human_x)
human_y = np.array(human_y)

cobweb_y_smooth, cobweb_lower_smooth, cobweb_upper_smooth = lowess(cobweb_x,
                                                                   cobweb_y)
naive_y_smooth, naive_lower_smooth, naive_upper_smooth = lowess(naive_x,
                                                                naive_y)
human_y_smooth, human_lower_smooth, human_upper_smooth = lowess(human_x,
                                                                human_y)

plt.fill_between(cobweb_x, cobweb_lower_smooth, cobweb_upper_smooth, alpha=0.5,
                 facecolor="green")
plt.fill_between(naive_x, naive_lower_smooth, naive_upper_smooth, alpha=0.5,
                 facecolor="red")
plt.fill_between(human_x, human_lower_smooth, human_upper_smooth, alpha=0.3,
                 facecolor="blue")

plt.plot(cobweb_x, cobweb_y_smooth, label="TRESTLE", color="green")
plt.plot(naive_x, naive_y_smooth, label="Naive Predictor", color="red")
plt.plot(human_x, human_y_smooth, label="Human Predictions", color="blue")

plt.gca().set_ylim([0.00, 1.0])
plt.gca().set_xlim([0, max(naive_x)-1])
plt.title("Incremental Tower Success Prediction Accuracy")
plt.xlabel("# of Training Examples")
plt.ylabel("Avg. Probability of True Success Label (Accuracy)")
plt.legend(loc=4)

plt.show()
