import math
import numpy
import hungarian
from labyrinth import Labyrinth

class Trestle(Labyrinth):

    def _hungarian_match(self, instance):
        from_name = [attr for attr in instance if isinstance(instance[attr],
                                                             Labyrinth)]
        to_name = []
        for attr in self.av_counts:
            for val in self.av_counts[attr]:
                if isinstance(val, Labyrinth):
                    to_name.append(attr)
                    break

        if(len(from_name) == 0 or
           len(to_name) == 0):
            return {}
        
        length = max(len(from_name), len(to_name))

        max_cost = 2.0
        
        cost_matrix = []
        for row_index in range(length):
            if row_index >= len(from_name):
                cost_matrix.append([max_cost] * length)
                continue 

            row = []
            for col_index in range(length):
                if col_index >= len(to_name):
                    row.append(max_cost)
                    continue

                cost = max_cost
                from_val = instance[from_name[row_index]]
                for val in self.av_counts[to_name[col_index]]:
                    #print((1.0 * self.av_counts[to_name[col_index]][val]) / self.count)
                    #print(from_val._probability_given(val))

                    cost -= (((1.0 * self.av_counts[to_name[col_index]][val]) /
                                          self.count) *
                                         from_val._probability_given(val))
                row.append(cost)
                    
            cost_matrix.append(row)

        a = numpy.array(cost_matrix)

        # may be able to eliminate this duplicate
        b = numpy.array(cost_matrix)
        assignment = hungarian.lap(a)[0]
        mapping = {}
        
        for index, val in enumerate(assignment):
            if (index >= len(from_name)):
                continue
            elif (b[index][val] == max_cost):
                mapping[from_name[index]] = "component" + self._gensym()
            else:
                mapping[from_name[index]] = to_name[val]
                
        return mapping 

    def _match(self, instance):
        """ 
        Define the specialized matching function to rename components
        and relations to maximize the match between instance and the
        current concept (self).
        """
        mapping = self._hungarian_match(instance)
        temp_instance = self._rename(instance, mapping)
        return temp_instance

    def _replace(self, old, new):
        """
        Traverse the tree and replace all references to concept old with
        concept new.
        """
        temp_counts = {}
        for attr in self.av_counts:
            temp_counts[attr] = {}
            for val in self.av_counts[attr]:
                x = val
                if val == old:
                    x = new
                if x not in temp_counts[attr]:
                    temp_counts[attr][x] = 0
                temp_counts[attr][x] += self.av_counts[attr][val] 

        self.av_counts = temp_counts

        for c in self.children:
            c._replace(old,new)

    def _split(self, best):
        """
        Specialized version of split for labyrinth. This removes all references
        to a particular concept from the tree. It replaces these references
        with a reference to the parent concept
        """
        self.children.remove(best)
        for child in best.children:
            self.children.append(child)

        # replace references to deleted concept with parent concept
        self.__class__.root._replace(best, self)

    def _probability_given(self, other):
        """
        The probability that the current node would be reached from another
        provided node.
        """
        if self == other:
            return 1.0
        elif len(other.children) == 0:
            return 0.0

        probs = [((child.count / (other.count * 1.0)) *
                 self._probability_given(child)) for child in other.children]
        
        return max(probs)

    def _probability_instance(self, instance):
        """
        Returns the probability of the instance's attribute values given
        the current concept.
        """
        prob = 0.0
        for attr in instance:
            prob += self._probability_attribute_value(attr, instance[attr])
        return prob

    def _probability_attribute_value(self, attr, val):
        """
        Returns the probability of a given attribute value pair in the current
        concept (self).
        """
        if attr not in self.av_counts:
            return 0.0

        if isinstance(val, float):
            float_values = []
            for v in self.av_counts[attr]:
                if isinstance(v, float):
                    float_values.append(v)
                
            # handle the float values
            if len(float_values) == 0:
                return 0.0
            mean = self._mean(float_values)
            std = self._std(float_values)

            # return the probability that the point (or a point further from
            # the mean) would be generated given the mean and std
            point = math.abs((val - mean) / (std))
            return (1.0 - math.erf(point / math.sqrt(2)))/2.0

        elif isinstance(val, Trestle):
            for v in self.av_counts[attr]:
                prob = 0.0
                if isinstance(v, Trestle):
                    prob += (((1.0 * self.av_counts[attr][v]) / self.count) *
                             val._probability_given(v))
            return prob

        else:
            if val not in self.av_counts[attr]:
                return 0.0
            return (1.0 * self.av_counts[attr][val]) / self.count

    def _cobweb_categorize(self, instance):
        """
        Sorts an instance in the categorization tree defined at the current
        node without modifying the counts of the tree.

        Uses the new and best operations; when new is the best operation it
        returns the current node otherwise it recurses on the best node. 
        """
        if not self.children:
            return self

        best1, best2 = self._two_best_children(instance)
        action_cu, best_action = self._get_best_operation(instance, best1,
                                                          best2, ["best",
                                                                  "new"]) 
        best1_cu, best1 = best1

        if best_action == "new":
            return self
        elif best_action == "best":
            # Only recurse if we increase the probability in the best.
            if (best1._probability_instance(instance) >
                self._probability_instance(instance)):
                return best1._cobweb_categorize(instance)
            
            return self

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
                    if isinstance(val, Labyrinth):
                        prob = 0.0
                        for val2 in self.av_counts[attr]:
                            if isinstance(val2, Labyrinth):
                                prob += (((1.0 * self.av_counts[attr][val2]) /
                                          self.count) *
                                         val._probability_given(val2))
                    else:
                        prob = ((1.0 * self.av_counts[attr][val]) / self.count)
                    correct_guesses += (prob * prob)

            # handle the float values
            if len(float_values) == 0:
                continue
            std = self._std(float_values)
            if std < acuity:
                std = acuity
            correct_guesses += (1.0 / (2.0 * math.sqrt(math.pi) * std))

        return correct_guesses

if __name__ == "__main__":

    t = Trestle()
    t.train_from_json("labyrinth_test.json")
    #t.train_from_json("towers_trestle.json")
    t.verify_counts()
    print(t)
    
    print("Predicting")

    test = {}
    right_stack = {}
    comp1 = {}
    comp1["color"] = "blue"
    comp1["shape"] = "odd"
    comp2 = {}
    comp2["color"] = "red"
    comp2["shape"] = "circular"
    comp3 = {}
    #comp3["color"] = "grey"
    comp3["shape"] = "square"
    relation1 = ["left-of", "component1", "component2"]
    relation2 = ["left-of", "component1", "component3"]
    relation3 = ["on", "component3", "component2"]
    right_stack["component1"] = comp1 
    right_stack["component2"] = comp2 
    right_stack["component3"] = comp3 
    right_stack["relation1"] = relation1 
    right_stack["relation2"] = relation2 
    right_stack["relation3"] = relation3 
    test["Rightstack-2"] = right_stack
    print(t.predict(test))

