from random import randint
from timeit import timeit

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches


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

def viz(n_inst, n_attr, n_val):
    from concept_formation.cobweb import CobwebTree
    from concept_formation.visualize import visualize

    tree = CobwebTree()
    x = generate_dataset(n_inst, n_attr, n_val)
    tree.fit(x)

    visualize(tree)


if __name__ == "__main__":

    viz(500, 10, 5) 

    # 5 attributes
    sizes = [10, 30, 60, 120, 180, 220, 500]
    times = [time(i, 5, 5) for i in sizes]
    plt.plot(sizes, times, 'ro')
    plt.plot(sizes, times, 'r-')

    # 10 attributes
    times = [time(i, 10, 5) for i in sizes]
    plt.plot(sizes, times, 'bo')
    plt.plot(sizes, times, 'b-')

    # 20 attributes
    times = [time(i, 20, 5) for i in sizes]
    plt.plot(sizes, times, 'go')
    plt.plot(sizes, times, 'g-')

    red_patch = mpatches.Patch(color='red', label='# attr=5')
    blue_patch = mpatches.Patch(color='blue', label='# attr=10')
    green_patch = mpatches.Patch(color='green', label='# attr=20')
    plt.legend(handles=[red_patch, blue_patch, green_patch], loc=2)

    plt.xlabel('Number of training instances (5 possible values / attr)')
    plt.ylabel('Runtime in Seconds')
    plt.show()
