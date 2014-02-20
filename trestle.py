import json
import math
import numpy
import hungarianNative
#import hungarian # depreciated
from labyrinth import Labyrinth

class Trestle(Labyrinth):

    def _trestle(self, instance):
        """
        Wraps the Labyrinth algorithm and includes context information.
        """
        temp_instance = {}
        for attr in instance:
            if isinstance(instance[attr], dict):
                temp_instance[attr] = self._trestle_categorize(instance[attr])
            elif isinstance(instance[attr], list):
                temp_instance[attr] = tuple(instance[attr])
            else:
                temp_instance[attr] = instance[attr]

        temp_instance = self._match(temp_instance)
        context = self._cobweb_categorize(temp_instance)

        temp_instance = {}
        for attr in instance:
            if isinstance(instance[attr], dict):
                temp2 = {}
                for a in instance[attr]:
                    temp2[a] = instance[attr][a]
                    temp2['parent_context'] = context
                temp_instance[attr] = self._trestle(temp2)
            elif isinstance(instance[attr], list):
                temp_instance[attr] = tuple(instance[attr])
            else:
                temp_instance[attr] = instance[attr]

        #print(temp_instance)
        temp_instance = self._match(temp_instance)
        return self._cobweb(temp_instance)

    def _trestle_categorize(self, instance):
        temp_instance = {}
        for attr in instance:
            if isinstance(instance[attr], dict):
                temp_instance[attr] = self._trestle_categorize(instance[attr])
            elif isinstance(instance[attr], list):
                temp_instance[attr] = tuple(instance[attr])
            else:
                temp_instance[attr] = instance[attr]

        temp_instance = self._match(temp_instance)
        context = self._cobweb_categorize(temp_instance)

        temp_instance = {}
        for attr in instance:
            if isinstance(instance[attr], dict):
                temp2 = {}
                for a in instance[attr]:
                    temp2[a] = instance[attr][a]
                    temp2['parent_context'] = context
                temp_instance[attr] = self._trestle_categorize(temp2)
            elif isinstance(instance[attr], list):
                temp_instance[attr] = tuple(instance[attr])
            else:
                temp_instance[attr] = instance[attr]

        temp_instance = self._match(temp_instance)
        return self._cobweb_categorize(temp_instance)

    # DEPRECIATED
    #def _common_ancestor(self, other_concept):
    #    temp = self
    #    while temp != None:
    #        if temp._is_parent(other_concept):
    #            return temp 
    #        temp = temp.parent
    #    print("ERROR Concepts not being compared in the same tree")

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

        # some reasonably large constant when dealing with really small
        # probabilities
        max_cost = 100.0
        
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

                reward = 0.0
                from_val = instance[from_name[row_index]]
                for val in self.av_counts[to_name[col_index]]:
                    #print((1.0 * self.av_counts[to_name[col_index]][val]) / self.count)
                    #print(from_val._probability_given(val))

                    #print((((1.0 * self.av_counts[to_name[col_index]][val]) /
                    #        self.count) * from_val._probability_given(val) *
                    #       val._probability_given(from_val)))
                    reward += (((1.0 * self.av_counts[to_name[col_index]][val])
                                / self.count) *
                               from_val._probability_given(val) *
                               val._probability_given(from_val))
                    #ancestor = from_val._common_ancestor(val)
                    #reward += (((1.0 * self.av_counts[to_name[col_index]][val]) /
                    #          self.count) *
                    #         from_val._probability_given(ancestor) *
                    #         val._probability_given(ancestor))
                    
                    # Additional bonus for part of a relational match
                    for attr in instance:
                        if not isinstance(attr, tuple):
                            continue
                        for attr2 in self.av_counts:
                            if not isinstance(attr2, tuple):
                                continue
                            if len(attr) != len(attr2):
                                continue
                            if attr[0] != attr2[0]:
                                continue
                            for i in range(1, len(attr)):
                                if (attr[i] == from_name[row_index] and attr2[i]
                                    == to_name[col_index]):
                                    reward += ((1.0 *
                                                self.av_counts[attr2][True] /
                                                self.count) * (1.0 /
                                                               len(attr2)))

                row.append(max_cost - (reward*reward))
                    
            cost_matrix.append(row)

        a = numpy.array(cost_matrix)

        # may be able to eliminate this duplicate
        b = numpy.array(cost_matrix)

        #depreciated c library approach
        #assignment1 = hungarian.lap(a)[0]

        ### substitute hungarian method ####
        assignment = hungarianNative.hungarian(a)

        #print("MATCHING")
        #print(b)
        #print(instance)
        #print(from_name)
        #print(self)
        #print(to_name)
        #print(assignment)
        
        ### substitute hungarian method ####
        mapping = {}
        
        for index, val in enumerate(assignment):
            if (index >= len(from_name)):
                continue
            elif (val >= len(to_name)):
                mapping[from_name[index]] = "component" + self._gensym()
            else:
                mapping[from_name[index]] = to_name[val]
            #print("match cost: %0.3f" % b[index][val])

            #elif (b[index][val] == max_cost):
            #    mapping[from_name[index]] = "component" + self._gensym()

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

    def _get_root(self):
        if self.parent == None:
            return self
        else:
            return self.parent._get_root()

    def _split(self, best):
        """
        Specialized version of split for labyrinth. This removes all references
        to a particular concept from the tree. It replaces these references
        with a reference to the parent concept
        """
        super(Trestle, self)._split(best)

        # replace references to deleted concept with parent concept
        self._get_root()._replace(best, self)

    def _probability_given(self, other):
        """
        The probability of the current node given we are at the other node. If
        self is a parent of other, then there is 100% prob. if other is a
        parent, than we need to compute the likelihood that it would be the
        current node. 
        """
        if self == other:
            return 1.0
        if (self.count >= other.count):
            if self._is_parent(other):
                return 1.0
            else:
                return 0.0
        elif (self.count < other.count):
            if other._is_parent(other):
                return (1.0 * self.count) / other.count

    #def _probability_given(self, other):
    #    """
    #    The probability that the current node would be reached from another
    #    provided node.
    #    """
    #    if self == other:
    #        return 1.0
    #    elif self.parent == None:
    #        return 0.0

    #    prob = ((self.count / (self.parent.count * 1.0)) *
    #            self.parent._probability_given(other))
    #    
    #    if(prob > 1.0):
    #        print(prob)
    #        print(self)
    #        print(self.parent)
    #    assert(prob <= 1.0)
    #    
    #    return prob

    # TODO I don't think I need to predict the relational attributes?
    #def _prob_attr_value(self, instance, attr, val):
    #    concept = self._labyrinth_categorize(instance)

    #    if isinstance(val, list):
    #        temp_instance = {}

    #        for attr in instance:
    #            if isinstance(instance[attr], dict):
    #                temp_instance[attr] = self._labyrinth(instance[attr])
    #            elif isinstance(instance[attr], list):
    #                temp_instance[tuple(instance[attr])] = True
    #            else:
    #                temp_instance[attr] = instance[attr]

    #        mapping = self._hungarian_match(temp_instance)
    #        #print(mapping)

    #        new_val = []
    #        for i in range(len(val)):
    #            if i == 0:
    #                new_val.append(val[i])
    #                continue
    #            if val[i] in mapping:
    #                new_val.append(mapping[val[i]])
    #            else:
    #                new_val.append(val[i])
    #        attr = tuple(new_val)
    #        val = True
    #        print(attr)
    #        print(concept)
    #        #print(concept._get_probability(attr,val))
    #        return concept._get_probability(attr, val)

    #    return concept._get_probability(attr, val)

    def _get_probability(self, attr, val):
        if attr not in self.av_counts:
            return 0.0

        if isinstance(val, Trestle):
            prob = 0.0
            for val2 in self.av_counts[attr]:
                if isinstance(val2, Trestle):

                    prob += (((1.0 * self.av_counts[attr][val2]) /
                              self.count) *
                             val._probability_given(val2))
                    #ancestor = val._common_ancestor(val2)

                    #prob += (((1.0 * self.av_counts[attr][val2]) /
                    #          self.count) *
                    #         val._probability_given(ancestor) *
                    #         val2._probability_given(ancestor))
            return prob

        if isinstance(val, float):
            # acuity the smallest allowed standard deviation; default = 1.0 
            acuity = 1.0
            float_values = []

            for fv in self.av_counts[attr]:
                if isinstance(fv, float):
                    float_values += [fv] * self.av_counts[attr][fv]

            if len(float_values) == 0:
                return 0.0

            mean = self._mean(float_values)
            std = self._std(float_values)
            if std < acuity:
                std = acuity

            point = abs((val - mean) / (std))
            return (1.0 - math.erf(point / math.sqrt(2)))#/2.0

        if val in self.av_counts[attr]:
            return (1.0 * self.av_counts[attr][val]) / self.count

        return 0.0

    def _probability_instance(self, instance):
        """
        Returns the probability of the instance's attribute values given
        the current concept.
        """
        prob = 1.0
        for attr in instance:
            prob *= self._get_probability(attr, instance[attr])
        return prob

    #def _prob_attribute_value(self, attr, val):
    #    """
    #    Returns the probability of a given attribute value pair in the current
    #    concept (self).

    #    NOTE: not used in category utility... just for prediction error.
    #    """
    #    if attr not in self.av_counts:
    #        return 0.0

    #    if isinstance(val, float):
    #        float_values = []
    #        for v in self.av_counts[attr]:
    #            if isinstance(v, float):
    #                float_values.append(v)
    #            
    #        # handle the float values
    #        if len(float_values) == 0:
    #            return 0.0
    #        mean = self._mean(float_values)
    #        std = self._std(float_values)

    #        # return the probability that the point (or a point further from
    #        # the mean) would be generated given the mean and std
    #        point = abs((val - mean) / (std))
    #        return (1.0 - math.erf(point / math.sqrt(2)))/2.0

    #    elif isinstance(val, Trestle):
    #        for v in self.av_counts[attr]:
    #            prob = 0.0
    #            if isinstance(v, Trestle):
    #                prob += (((1.0 * self.av_counts[attr][v]) / self.count) *
    #                         val._probability_given(v))
    #        return prob

    #    else:
    #        if val not in self.av_counts[attr]:
    #            return 0.0
    #        return (1.0 * self.av_counts[attr][val]) / self.count

    #def _cobweb_categorize(self, instance):
    #    """
    #    Sorts an instance in the categorization tree defined at the current
    #    node without modifying the counts of the tree.

    #    Uses the new and best operations; when new is the best operation it
    #    returns the current node otherwise it recurses on the best node. 
    #    """
    #    if not self.children:
    #        return self

    #    best1, best2 = self._two_best_children(instance)
    #    action_cu, best_action = self._get_best_operation(instance, best1,
    #                                                      best2, ["best",
    #                                                              "new"]) 
    #    best1_cu, best1 = best1

    #    if best_action == "new":
    #        return self
    #    elif best_action == "best":
    #        # Only recurse if we increase the probability in the best.
    #        if (best1._probability_instance(instance) >
    #            self._probability_instance(instance)):
    #            return best1._cobweb_categorize(instance)
    #        
    #        return self

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
                    if isinstance(val, Trestle):
                        prob = 0.0
                        for val2 in self.av_counts[attr]:
                            if isinstance(val2, Trestle):
                                prob += (((1.0 * self.av_counts[attr][val2]) /
                                          self.count) *
                                         val._probability_given(val2) *
                                         val2._probability_given(val))
                                #ancestor = val._common_ancestor(val2)

                                #prob += (((1.0 * self.av_counts[attr][val2]) /
                                #          self.count) *
                                #         val._probability_given(ancestor) *
                                #         val2._probability_given(ancestor))
                    else:
                        prob = ((1.0 * self.av_counts[attr][val]) / self.count)
                    #prob = ((1.0 * self.av_counts[attr][val]) / self.count)
                    correct_guesses += (prob * prob)

            # handle the float values
            if len(float_values) == 0:
                continue
            std = self._std(float_values)
            if std < acuity:
                std = acuity
            correct_guesses += (1.0 / (2.0 * math.sqrt(math.pi) * std))

        return correct_guesses

    #def _prob_attr_value(self, instance, attr, val):
    #    concept = self._trestle_categorize(instance)
    #    return concept._get_probability(attr, val)

    #def ifit(self, instance):
    #    self._trestle(instance)

if __name__ == "__main__":

    print(Trestle().cluster("data_files/rb_com_11_noCheck.json", 20, 100))
    #print(Trestle().cluster("towers_trestle.json", 15))
    #print(Trestle().cluster("data_files/rb_s_07.json", 1, 3))
    #Labyrinth().predictions("data_files/rb_s_07.json", 15, 4)
    #Trestle().predictions("towers_small_trestle.json", 10, 1)

