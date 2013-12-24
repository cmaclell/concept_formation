import math
import random
import numpy as np
from cobweb import CobwebTree

class Cobweb3Tree(CobwebTree):

    def _category_utility_numeric(self, parent):
        category_utility = 0.0
        for attr in self.av_counts:
            child_vals = []
            parent_vals = []

            for val in self.av_counts[attr]:
                if isinstance(val, float):
                    child_vals += [val] * int(self.av_counts[attr][val])
            for val in parent.av_counts[attr]:
                if isinstance(val, float):
                    parent_vals += [val] * int(parent.av_counts[attr][val])

            if len(child_vals) == 0:
                continue

            mean_child = sum(child_vals) / len(child_vals)
            std_child = math.sqrt(sum([(v - mean_child)**2 for v in
                                       child_vals])/len(child_vals))

            mean_parent = sum(parent_vals) / len(parent_vals)
            std_parent = math.sqrt(sum([(v - mean_parent)**2 for v in
                                       parent_vals])/len(parent_vals))

            if std_child < 1.0:
                std_child = 1.0

            if std_parent < 1.0:
                std_parent = 1.0

            category_utility += (1.0/std_child - 1.0/std_parent)

        return category_utility / (2.0 * math.sqrt(math.pi))
         
    def _category_utility(self):
        if len(self.children) == 0:
            return 0.0

        category_utility = 0.0

        for child in self.children:
            p_of_child = child.count / self.count
            category_utility += (p_of_child *
                                 (child._category_utility_nominal(self) +
                                  child._category_utility_numeric(self)))
        return category_utility / (1.0 * len(self.children))

if __name__ == "__main__":
    t = Cobweb3Tree()

    instances = []

    for v in np.random.randn(10):
        r = {}
        r['val'] = v
        instances.append(r)

    for v in (40 + np.random.randn(10)):
        r = {}
        r['val'] = v
        instances.append(r)

    random.shuffle(instances)
    for i in instances:
        t._cobweb(i)

    t._pretty_print()
    print(t._category_utility())

