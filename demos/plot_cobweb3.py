import time
from cobweb3 import *
from random import normalvariate
from random import shuffle, uniform
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

tree = Cobweb3Tree()

num_clusters = 4 
num_samples = 30
sigma = 1

xmean = [uniform(-8, 8) for i in range(num_clusters)]
ymean = [uniform(-8, 8) for i in range(num_clusters)]
label = ['bo', 'go', 'ro', 'co', 'mo', 'yo', 'ko']
shuffle(label)
label = label[0:num_clusters]

data = []
actual = []
clusters = []
for i in range(num_clusters):
    data += [{'x': normalvariate(xmean[i], sigma), 'y':
              normalvariate(ymean[i], sigma), '_label': label[i]} for j in
             range(num_samples)]
    actual.append(Ellipse([xmean[i], ymean[i]], width=4*sigma,
                          height=4*sigma, angle=0))
shuffle(data)
trained = []

plt.ion()
plt.show()

# draw the actual sampling distribution
for c in actual:
    c.set_alpha(0.08)
    c.set_facecolor("blue")
    plt.gca().add_patch(c)

for datum in data:
    #train the tree on the sampled datum
    tree.ifit(datum)
    trained.append(datum)

    # remove old cluster circles
    for c in clusters:
        c.remove()

    # 4 * std gives two std on each side (~95% confidence)
    clusters = [Ellipse([cluster.av_counts['x'].unbiased_mean(),
                         cluster.av_counts['y'].unbiased_mean()],
                        width=4*cluster.av_counts['x'].unbiased_std(),
                        height=4*cluster.av_counts['y'].unbiased_std(), 
                        angle=0) for cluster in tree.root.children]

    # draw the cluster circles
    for c in clusters:
        c.set_alpha(0.1)
        c.set_facecolor('red')
        plt.gca().add_patch(c)

    # draw the new point
    plt.plot([datum['x']], [datum['y']], datum['_label'])

    plt.axis([-10, 10, -10, 10])
    plt.draw()
    time.sleep(0.0001)

plt.ioff()
plt.show()
