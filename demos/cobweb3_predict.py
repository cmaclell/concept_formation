from random import normalvariate
from random import shuffle, uniform
import numpy as np
import matplotlib.pyplot as plt

from utils import moving_average
from predict import incremental_prediction
from dummy import DummyTree
from cobweb3 import Cobweb3Tree

window = 50 
num_runs = 10 
num_examples = 200 

############################ GENERATE SIMULATED DATA ########################
num_clusters = 2 
num_samples = 300
sigma = 0.15 

mean = [uniform(-100, 100) for i in range(num_clusters)]
label = ['bo', 'go', 'ro', 'co', 'mo', 'yo', 'ko']
shuffle(label)
label = label[0:num_clusters]

simulated_data = []
for i in range(num_clusters):
    simulated_data += [{'x': normalvariate(0, sigma), 'label': 'l1'} for
                       j in range(num_samples)]
    simulated_data += [{'x': normalvariate(10, sigma), 'label': 'l2'} for
                       j in range(num_samples)]

############################## GENERATE PREDICTIONS ##########################

naive_accuracy = incremental_prediction(DummyTree(), simulated_data,
                                  run_length=num_examples,
                                  runs=num_runs, attr="x")
cobweb_accuracy = incremental_prediction(Cobweb3Tree(), simulated_data,
                                  run_length=num_examples,
                                  runs=num_runs, attr="x")

############################## PLOT RESULTS ##################################

cobweb_data = [[] for i in range(len(cobweb_accuracy[0]))]
naive_data = [[] for i in range(len(naive_accuracy[0]))]

for run in cobweb_accuracy:
    for i,v in enumerate(run):
        cobweb_data[i].append(v)
for run in naive_accuracy:
    for i,v in enumerate(run):
        naive_data[i].append(v)

cobweb_y = np.array([np.mean(l) for l in cobweb_data])
naive_y = np.array([np.mean(l) for l in naive_data])

cobweb_y_smooth = moving_average(cobweb_y, window)
naive_y_smooth = moving_average(naive_y, window)

x = np.array([i for i in range(len(cobweb_y_smooth))])

cobweb_lower = np.array([np.mean(l) - 2 * np.std(l) for l in cobweb_data])
naive_lower = np.array([np.mean(l) - 2 * np.std(l) for l in naive_data])

cobweb_lower_smooth = moving_average(cobweb_lower, window)
naive_lower_smooth = moving_average(naive_lower, window)

cobweb_upper = np.array([np.mean(l) + 2 * np.std(l) for l in cobweb_data])
naive_upper = np.array([np.mean(l) + 2 * np.std(l) for l in naive_data])

cobweb_upper_smooth = moving_average(cobweb_upper, window)
naive_upper_smooth = moving_average(naive_upper, window)

plt.fill_between(x, cobweb_lower_smooth, cobweb_upper_smooth, alpha=0.5,
                 facecolor="green")
plt.fill_between(x, naive_lower_smooth, naive_upper_smooth, alpha=0.5,
                 facecolor="red")

plt.plot(x, cobweb_y_smooth, label="COBWEB", color="green")
plt.plot(x, naive_y_smooth, label="Naive Predictor", color="red")

#plt.gca().set_ylim([0,1.05])
plt.gca().set_xlim([0,len(naive_y_smooth)-1])
plt.title("Incremental Mushroom Edibility Prediction Accuracy")
plt.xlabel("# of Training Examples")
plt.ylabel("Accuracy")
plt.legend(loc=4)

plt.show()

