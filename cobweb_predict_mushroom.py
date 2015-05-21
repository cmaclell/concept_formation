import json
import numpy as np
import matplotlib.pyplot as plt

from concept_formation.utils import lowess
from concept_formation.predict import incremental_prediction
from concept_formation.cobweb import CobwebTree
from concept_formation.dummy import DummyTree

num_runs = 30 
num_examples = 30 

with open('data_files/mushrooms.json') as fin:
    mushrooms = json.load(fin)

############################## GENERATE PREDICTIONS ##########################

naive_data = incremental_prediction(DummyTree(), mushrooms,
                                  run_length=num_examples,
                                  runs=num_runs, attr="classification")
cobweb_data = incremental_prediction(CobwebTree(), mushrooms,
                                  run_length=num_examples,
                                  runs=num_runs, attr="classification")

############################## PLOT RESULTS ##################################

naive_data.sort()
cobweb_data.sort()

cobweb_x, cobweb_y = [], []
naive_x, naive_y = [], []

for x,y in cobweb_data:
    cobweb_x.append(x)
    cobweb_y.append(y)
for x,y in naive_data:
    naive_x.append(x)
    naive_y.append(y)

cobweb_x = np.array(cobweb_x)
cobweb_y = np.array(cobweb_y)
naive_x = np.array(naive_x)
naive_y = np.array(naive_y)

cobweb_y_smooth, cobweb_lower_smooth, cobweb_upper_smooth = lowess(cobweb_x, cobweb_y)
naive_y_smooth, naive_lower_smooth, naive_upper_smooth = lowess(naive_x, naive_y)

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
