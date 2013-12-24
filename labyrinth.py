from random import shuffle
import numpy as np
from cobweb3 import Cobweb3Tree

class Labyrinth(Cobweb3Tree):

    # TODO naming convention on these category utility functions is weird and
    # probably needs to be redone
    def _category_utility_component(self, parent):
        #TODO Stub
        return 0.0

    def _labyrinth(self, instance):
        pass

    def _category_utility(self):
        if len(self.children) == 0:
            return 0.0

        category_utility = 0.0

        for child in self.children:
            p_of_child = child.count / self.count
            category_utility += (p_of_child *
                                 (child._category_utility_nominal(self) +
                                  child._category_utility_numeric(self) +
                                  child._category_utility_component(self)))
        return category_utility / (1.0 * len(self.children))

    pass

if __name__ == "__main__":

    t = Labyrinth()
    instances = []

    for v in np.random.randn(10):
        r = {}
        r['x'] = v
        r['sample_mean'] = "0"
        instances.append(r)

    for v in (40 + np.random.randn(10)):
        r = {}
        r['x'] = v
        r['sample_mean'] = "40"
        instances.append(r)

    shuffle(instances)
    t.fit(instances)
    print(t)

    test = {}
    test['sample_mean'] = "40"
    print(t.predict(test))



