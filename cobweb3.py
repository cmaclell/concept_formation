import math
import json
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

    def _get_probability(self, attr, val):
        if attr not in self.av_counts:
            return 0.0

        if isinstance(val, float):
            # acuity the smallest allowed standard deviation; default = 1.0 
            acuity = 1.0
            float_values = []

            for val in self.av_counts[attr]:
                if isinstance(val, float):
                    float_values += [val] * self.av_counts[attr][val]

            if len(float_values) == 0:
                return 0.0

            mean = self._mean(float_values)
            std = self._std(float_values)
            if std < acuity:
                std = acuity

            point = abs((val - mean) / (std))
            #print(point)
            return (1.0 - math.erf(point / math.sqrt(2)))#/2.0
        
        if val in self.av_counts[attr]:
            return (1.0 * self.av_counts[attr][val]) / self.count

        return 0.0

    def _output_json(self):
        output = {}
        output['name'] = self.concept_name
        output['size'] = self.count
        output['children'] = []

        temp = {}
        for attr in self.av_counts:
            float_vals = []
            for value in self.av_counts[attr]:
                if isinstance(value, float):
                    float_vals.append(value)
                else:
                    temp[attr + " = " + str(value)] = self.av_counts[attr][value]
            if len(float_vals) > 0:
                mean = attr + "_mean = %0.2f (%0.2f)" % (self._mean(float_vals),
                                                self._std(float_vals))
                temp[mean] = len(float_vals)

        for child in self.children:
            output['children'].append(child._output_json())

        output['counts'] = temp

        return output

if __name__ == "__main__":

    n = 1 
    runs = []
    for i in range(0,n):
        print("run %i" % i)
        t = Cobweb3Tree()
        runs.append(t.sequential_prediction("cobweb3_test.json", 10))
        print(json.dumps(t._output_json()))

    print(runs)
    for i in range(0,len(runs[0])):
        a = []
        for r in runs:
            a.append(r[i])
        print("mean: %0.2f, std: %0.2f" % (Cobweb3Tree()._mean(a),
                                           Cobweb3Tree()._std(a)))

    #t = Cobweb3Tree()

    ##instances = []

    ##for v in np.random.randn(10):
    ##    r = {}
    ##    r['x'] = v
    ##    r['sample_mean'] = "0"
    ##    instances.append(r)

    ##for v in (40 + np.random.randn(10)):
    ##    r = {}
    ##    r['x'] = v
    ##    r['sample_mean'] = "40"
    ##    instances.append(r)

    #t.train_from_json("cobweb3_test.json")
    #t.verify_counts()
    #print(t)
    #print()

    #test = {}
    #test['sample_mean'] = "40"
    #print(t.predict(test))


