import math
import numpy
import hungarianNative
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
        max_cost = 1000.0
        
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
                    reward += (((1.0 * self.av_counts[to_name[col_index]][val])
                                / self.count) *
                               from_val._probability_given(val))
                    #*
                    #           val._probability_given(from_val))
                    
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

        # Note: "a" is modified by hungarian.
        a = numpy.array(cost_matrix)
        assignment = hungarianNative.hungarian(a)
        #print(assignment)

        mapping = {}
        for index in assignment:
            if index >= len(from_name):
                continue
            elif assignment[index] >= len(to_name):
                  mapping[from_name[index]] = "component" + self._gensym()
            else:
                mapping[from_name[index]] = to_name[assignment[index]]

        #print(mapping)
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

    def _remove(self):
        """
        Specialized version of remove for trestle. This removes all references
        to a particular concept from the tree. It replaces these references
        with a reference to the parent concept
        """
        # call recursively
        for c in self.children:
            c._remove()

        # replace references to deleted concept with parent concept
        self._get_root()._replace(self, self.parent)

        # deletes the child
        self.parent.children.remove(self)

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

        if (self.count > other.count):
            if self._is_parent(other):
                return 1.0
            else:
                return 0.0

        elif (self.count < other.count):
            if other._is_parent(self):
                return (1.0 * self.count) / other.count

        return 0.0

    def _get_probability(self, attr, val):
        """
        Gets the probability of a particular attribute value. This has been
        modified to support numeric and nominal values. The acuity is set as a
        global parameter now. 
        """
        if attr not in self.av_counts:
            return 0.0

        if isinstance(val, Trestle):
            prob = 0.0
            for val2 in self.av_counts[attr]:
                if isinstance(val2, Trestle):

                    prob += (((1.0 * self.av_counts[attr][val2]) /
                              self.count) *
                             val._probability_given(val2))
                    #* 
                    #         val2._probability_given(val))

            return prob

        if isinstance(val, float):
            float_values = []

            for av in self.av_counts[attr]:
                if isinstance(av, float):
                    float_values += [av] * self.av_counts[attr][av]

            mean = self._mean(float_values)
            std = self._unbiased_std(float_values)

            # assign 100% accuracy to the mean
            return (math.exp(-((val - mean) * (val - mean)) / (2.0 * std * std)))
        
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

    def _get_parents(self):
        parents = {}
        parents[self] = 1.0
        current = self
        while current.parent:
            parents[current.parent] = (self.count * 1.0) / current.parent.count
            current = current.parent
        return parents

    def _expected_correct_guesses(self):
        """
        Computes the number of attribute values that would be correctly guessed
        in the current concept. This extension supports both nominal and
        numeric attribute values. The acuity parameter should be set based on
        the domain cobweb is being used on. The acuity is set as a global
        parameter now. 
        """
        correct_guesses = 0.0

        for attr in self.av_counts:
            float_values = []
            for val in self.av_counts[attr]:
                if isinstance(val, float):
                    float_values += [val] * int(self.av_counts[attr][val])
                elif isinstance(val, Trestle):
                    prob = 0.0

                    parents = val._get_parents()
                    for val2 in self.av_counts[attr]:
                        if val2 in parents:
                            prob += (((1.0 * self.av_counts[attr][val2]) /
                                     self.count) * parents[val2])

                    # only consider 6 biggest additional values
                    #prob += (((1.0 * self.av_counts[attr][val]) / self.count))
                    #values = sorted([val2 for val2 in
                    #                 self.av_counts[attr] if val2 != val and
                    #                 isinstance(val2, Trestle)],
                    #                key=lambda v: v.count)
                    #values = values[:6]
                    #for val2 in values:

                    #for val2 in self.av_counts[attr]:
                    #    if isinstance(val2, Trestle):
                    #    
                    #        prob += (((1.0 * self.av_counts[attr][val2]) /
                    #                  self.count) *
                    #                 val._probability_given(val2))

                else:
                    prob = ((1.0 * self.av_counts[attr][val]) / self.count)

                correct_guesses += (prob * prob)

            if len(float_values) == 0:
                continue

            std = self._unbiased_std(float_values)
            correct_guesses += (1.0 / (2.0 * math.sqrt(math.pi) * std))

        return correct_guesses

    #def _prob_attr_value(self, instance, attr, val):
    #    concept = self._trestle_categorize(instance)
    #    return concept._get_probability(attr, val)

    #def ifit(self, instance):
    #    self._trestle(instance)

if __name__ == "__main__":

    t = Trestle()
    print(t.cluster("data_files/rb_com_11_noCheck.json", 60))

    #print(Trestle().cluster("towers_trestle.json", 15))
    #print(Trestle().cluster("data_files/rb_s_07.json", 1, 3))
    #Labyrinth().predictions("data_files/rb_s_07.json", 15, 4)
    #Trestle().predictions("towers_small_trestle.json", 10, 1)

