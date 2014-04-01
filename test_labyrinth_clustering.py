import pickle
from sklearn import metrics

human = pickle.load(open('human.pickle', 'rb'))
machine = pickle.load(open('clustering.pickle', 'rb'))

labels_true = []
labels_pred = []

for guid in machine:
    if guid in human:
        labels_true.append(human[guid])
        labels_pred.append(machine[guid])

print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels_pred))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels_pred))
print("V-Measure: %0.3f" % metrics.v_measure_score(labels_true, labels_pred))
print("ARI: %0.3f" % metrics.adjusted_rand_score(labels_true, labels_pred))
print("# human clusters: %i" % len(set(labels_true)))
print("# machine clusters: %i" % len(set(labels_pred)))

