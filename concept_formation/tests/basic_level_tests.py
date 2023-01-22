from random import randint
from timeit import timeit
from random import shuffle
from random import random
from random import choice

from tqdm import tqdm
from collections import Counter
from collections import defaultdict
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from concept_formation.visualize import visualize
from concept_formation.cobweb import CobwebTree
from concept_formation.datasets import load_mushroom

def generate_dataset(n_inst, n_attr, n_val):
    instances = []
    for i in range(n_inst):
        i = {}
        for j in range(n_attr):
            i[str(j)] = randint(1, n_val)
        instances.append(i)
    return instances


def time(n_inst, n_attr, n_val):
    return timeit('tree.fit(x)',
                  setup=('from __main__ import generate_dataset; '
                         'from concept_formation.cobweb import CobwebTree; '
                         'tree = CobwebTree(); '
                         'x = generate_dataset(%i, %i, %i)' % (n_inst, n_attr,
                                                               n_val)),
                  number=1)


if __name__ == "__main__":
    # # 5 attributes
    # sizes = [10, 30, 60, 120, 180, 220, 500]
    # times = [time(i, 5, 5) for i in sizes]
    # plt.plot(sizes, times, 'ro')
    # plt.plot(sizes, times, 'r-')

    # # 10 attributes
    # times = [time(i, 10, 5) for i in sizes]
    # plt.plot(sizes, times, 'bo')
    # plt.plot(sizes, times, 'b-')

    # # 20 attributes
    # times = [time(i, 20, 5) for i in sizes]
    # plt.plot(sizes, times, 'go')
    # plt.plot(sizes, times, 'g-')

    # red_patch = mpatches.Patch(color='red', label='# attr=5')
    # blue_patch = mpatches.Patch(color='blue', label='# attr=10')
    # green_patch = mpatches.Patch(color='green', label='# attr=20')
    # plt.legend(handles=[red_patch, blue_patch, green_patch], loc=2)

    # plt.xlabel('Number of training instances (5 possible values / attr)')
    # plt.ylabel('Runtime in Seconds')
    # plt.show()

    mushrooms = load_mushroom(300)

    # for e in mushrooms:
    #     for i in range(10):
    #         e["classification{}".format(i)] = e['classification']

    tree = CobwebTree()
    tree.fit(mushrooms)
    visualize(tree)

    avs = defaultdict(Counter)
    for m in mushrooms:
        for attr in m:
            avs[attr][m[attr]] += 1

    accuracy_leaf = []
    accuracy_basic = []
    for i in tqdm(range(200)):
        tree = CobwebTree()
        shuffle(mushrooms)
        run_leaf = []
        run_basic = []
        for m in mushrooms:
            attrs = list(m)
            shuffle(attrs)
            target = attrs[0]
            # target = 'classification'

            # add some noise
            tmp = {a: choice(list(avs[a])) if random() < 0.3 else m[a] for a in m}

            leaf = tree.categorize({a:tmp[a] for a in tmp if a != target})

            pred_leaf = leaf.predict(target)
            run_leaf.append(int(pred_leaf == m[target]))

            try:
                pred_basic = leaf.get_basic_level().predict(target)
                # print("LEAF", leaf)
                # print("BASIC", leaf.get_basic_level())
            except:
                pred_basic = None
            run_basic.append(int(pred_basic == m[target]))

            tree.ifit(tmp)

        accuracy_leaf.append(run_leaf)
        accuracy_basic.append(run_basic)

    avg_leaf = [sum([run[i] for run in accuracy_leaf])/len(accuracy_leaf) for i in range(len(mushrooms))]
    avg_basic = [sum([run[i] for run in accuracy_basic])/len(accuracy_basic) for i in range(len(mushrooms))]
    plt.plot(avg_leaf, label="leaf", alpha=0.5)
    plt.plot(avg_basic, label="basic", alpha=0.5)
    ax = plt.gca()
    ax.legend()
    plt.show()


