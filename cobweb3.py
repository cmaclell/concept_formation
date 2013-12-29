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

    def _expected_correct_guesses(self):
        """
        Computes the number of attribute values that would be correctly guessed
        in the current concept. This extension supports both nominal and
        numeric attribute values.
        """
        # acuity the smallest allowed standard deviation; default = 1.0 
        acuity = 1.0
        correct_guesses = 0.0

        for attr in self.av_counts:
            float_values = []
            for val in self.av_counts[attr]:
                if isinstance(val, float):
                    float_values += [val] * int(self.av_counts[attr][val])
                else:
                    prob = ((1.0 * self.av_counts[attr][val]) / self.count)
                    correct_guesses += (prob * prob)

            if len(float_values) == 0:
                continue

            std = self._std(float_values)

            if std < acuity:
                std = acuity

            correct_guesses += (1.0 / (2.0 * math.sqrt(math.pi) * std))

        return correct_guesses
         
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
                    values.append("'" + str(val) + "': " +
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

        concept = self._cobweb_categorize(instance)
        
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

    t = Cobweb3Tree()

    #instances = []

    #for v in np.random.randn(10):
    #    r = {}
    #    r['x'] = v
    #    r['sample_mean'] = "0"
    #    instances.append(r)

    #for v in (40 + np.random.randn(10)):
    #    r = {}
    #    r['x'] = v
    #    r['sample_mean'] = "40"
    #    instances.append(r)

    t.train_from_json("cobweb3_test.json")
    t._verify_counts()
    print(t)
    print()

    test = {}
    test['sample_mean'] = "40"
    print(t.predict(test))


