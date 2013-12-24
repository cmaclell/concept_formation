import math
from random import normalvariate
from random import choice
from random import random
from cobweb import CobwebTree

class Cobweb3Tree(CobwebTree):

    def _mean(self, values):
        return sum(values) / len(values)

    def _std(self, values):
        return math.sqrt(sum([(v - self._mean(values))**2 for v in
                                       values])/len(values))

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

            std_child = self._std(child_vals)
            std_parent = self._std(parent_vals)

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

    def _pretty_print(self, depth=0):
        """
        Prints the categorization tree.
        """
        ret = str(('\t' * depth) + "|-")

        attributes = []

        for attr in self.av_counts:
            float_values = []
            values = []

            for val in self.av_counts[attr]:
                if isinstance(val, float):
                    float_values.append(val)
                else:
                    values.append("'" + val + "': " +
                                  str(self.av_counts[attr][val]))

            if float_values:
                values.append("'mean':" + str(self._mean(float_values)))
                values.append("'std':" + str(self._std(float_values)))

            attributes.append("'" + attr + "': {" + ", ".join(values) + "}")
                  
        ret += "{" + ", ".join(attributes) + "}: " + str(self.count) + '\n'
        
        for c in self.children:
            ret += c._pretty_print(depth+1)

        return ret

    def predict(self, instance):
        """
        Given an instance predict any missing attribute values without
        modifying the tree.
        """
        prediction = {}

        # make a copy of the instance
        for attr in instance:
            prediction[attr] = instance[attr]

        concept = self._categorize(instance)
        
        for attr in concept.av_counts:
            if attr in prediction:
                continue
            
            nominal_values = []
            float_values = []

            for val in concept.av_counts[attr]:
                if isinstance(val, float):
                    float_values += [val] * concept.av_counts[attr][val]
                else:
                    nominal_values += [val] * concept.av_counts[attr][val]

            if random() < ((len(nominal_values) * 1.0) / (len(nominal_values) +
                                                          len(float_values))):
                prediction[attr] = choice(nominal_values)
            else:
                prediction[attr] = normalvariate(self._mean(float_values),
                                                 self._std(float_values))

        return prediction

if __name__ == "__main__":
    from random import shuffle
    import numpy as np

    t = Cobweb3Tree()
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


