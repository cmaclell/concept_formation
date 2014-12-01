import itertools
import hungarianNative
import numpy
import json
from utils import ContinuousValue
from cobweb3 import Cobweb3Tree, Cobweb3Node
from structure_mapper import standardizeApartNames

class LabyrinthTree(Cobweb3Tree):

    def __init__(self):
        self.root = LabyrinthNode()

    def ifit(self, instance):
        """
        A modification of ifit to call labyrinth instead.
        """
        return self.labyrinth(instance)

    def labyrinth(self, instance):
        """
        Recursively calls labyrinth on all of the components in a depth-first
        traversal. Once all of the components have been classified then then it
        classifies the current node.
        """
        instance = standardizeApartNames(instance)
        temp_instance = {}
        for attr in instance:
            if isinstance(instance[attr], dict):
                temp_instance[attr] = self.labyrinth(instance[attr])
            elif isinstance(instance[attr], list):
                temp_instance[tuple(instance[attr])] = True
            else:
                temp_instance[attr] = instance[attr]

        temp_instance = self.root.match(temp_instance)
        return self.cobweb(temp_instance)

    def labyrinth_categorize(self, instance):
        """
        The labyrinth categorize function, this labyrinth categorizes all the
        sub-components before categorizing itself.
        """
        instance = standardizeApartNames(instance)
        temp_instance = {}
        for attr in instance:
            if isinstance(instance[attr], dict):
                temp_instance[attr] = self.labyrinth_categorize(instance[attr])
            elif isinstance(instance[attr], list):
                temp_instance[attr] = tuple(instance[attr])
            else:
                temp_instance[attr] = instance[attr]

        temp_instance = self.root.match(temp_instance)
        return self.cobweb_categorize(temp_instance)

class LabyrinthNode(Cobweb3Node):

    def rename(self, instance, mapping):
        """
        Given a mapping (type = dict) rename the 
        components and relations and return the renamed
        instance.
        """
        # Ensure it is a complete mapping
        # Might be troublesome if there is a name collision
        for attr in instance:
            if not isinstance(instance[attr], LabyrinthNode):
                continue
            if attr not in mapping:
                mapping[attr] = attr

        temp_instance = {}
        relations = []

        # rename all attribute values
        for attr in instance:
            if isinstance(attr, tuple):
                relations.append(attr)
            elif isinstance(instance[attr], LabyrinthNode):
                mapping[attr]
                instance[attr]
                temp_instance[mapping[attr]] = instance[attr]
            else:
                temp_instance[attr] = instance[attr]

        #rename relations and add them to instance
        for relation in relations:
            temp = []
            for idx, val in enumerate(relation):
                if idx == 0 or val not in mapping:
                    temp.append(val)
                else:
                    temp.append(mapping[val])
            temp_instance[tuple(temp)] = True

        return temp_instance

    def probability_given(self, other):
        """
        The probability of the current node given we are at the other node. If
        self is a parent of other, then there is 100% prob. if other is a
        parent, than we need to compute the likelihood that it would be the
        current node. 
        """
        if self == other:
            return 1.0

        if (self.count > other.count):
            if self.is_parent(other):
                return 1.0
            else:
                return 0.0

        elif (self.count < other.count):
            if other.is_parent(self):
                return (1.0 * self.count) / other.count

        return 0.0

    def hungarianmatch(self, instance):
        """
        Compute the greedy match between the instance and the current concept.
        This algorithm is O(n^3) and will return the optimal match if there are
        no relations. However, when there are relations, then it is only
        computing the best greedy match (no longer optimal). 
        """

        # FOR DEBUGGING
        #print("INSTANCE")
        #for attr in instance:
        #    if isinstance(instance[attr], Labyrinth):
        #        print(attr + ": " + str(instance[attr].concept_name))

        #print("CONCEPT")
        #for attr in self.av_counts:
        #    for val in self.av_counts[attr]:
        #        if isinstance(val, Labyrinth):
        #            print(attr + ": " + str(val.concept_name))

        from_name = [attr for attr in instance if isinstance(instance[attr],
                                                             LabyrinthNode)]
        to_name = []
        for attr in self.av_counts:
            if isinstance(self.av_counts[attr], ContinuousValue):
                continue
            for val in self.av_counts[attr]:
                if isinstance(val, LabyrinthNode):
                    to_name.append(attr)
                    break

        if(len(from_name) == 0 or
           len(to_name) == 0):
            return {}
        
        length = max(len(from_name), len(to_name))

        # some reasonably large constant when dealing with really small
        # probabilities + bonuses for relations, which mean a given match may
        # be greater than 1.0.
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

                #normal labyrinth nominal style match
                #if from_val in self.av_counts[to_name[col_index]]:
                #    reward = (((1.0 * self.av_counts[to_name[col_index]][from_val]) /
                #              self.count))

                # match based on match to values with shared ancestory.
                for val in self.av_counts[to_name[col_index]]:
                    ancestor = self.common_ancestor(from_val, val)
                    reward += (((1.0 * self.av_counts[to_name[col_index]][val])
                                / self.count) *
                               from_val.probability_given(ancestor) *
                               val.probability_given(ancestor))

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
                            if (attr[i] == from_name[row_index] and 
                                attr2[i] == to_name[col_index]):
                                reward += ((1.0 *
                                            self.av_counts[attr2][True] /
                                            self.count) * (1.0 /
                                                           len(attr2)))
                row.append(max_cost - (reward))
                    
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
                  mapping[from_name[index]] = "component" + self.gensym()
            else:
                mapping[from_name[index]] = to_name[assignment[index]]

        #print(mapping)
        return mapping 

    def exhaustivematch(self, instance):
        """
        Compute matches exhaustively. This has O(n!) time complexity, so it
        won't scale very well.
        """
        from_name = [attr for attr in instance if isinstance(instance[attr],
                                                             LabyrinthNode)]
        to_name = []
        for attr in self.av_counts:
            for val in self.av_counts[attr]:
                if isinstance(val, LabyrinthNode):
                    to_name.append(attr)
                    break

        mappings = []
        if len(from_name) > len(to_name):
            from_lists = set([x[:len(to_name)] for x in
                             itertools.permutations(from_name)])
            for from_list in from_lists:
                mapping = {}
                for i in range(len(to_name)):
                    mapping[from_list[i]] = to_name[i]
                mappings.append(mapping)
        else:
            to_lists = set([x[:len(from_name)] for x in 
                            itertools.permutations(to_name)])
            for to_list in to_lists:
                mapping = {}
                for i in range(len(from_name)):
                    mapping[from_name[i]] = to_list[i]
                mappings.append(mapping)
    
        scored_mappings = []
        for mapping in mappings:
            temp = self._shallow_copy()
            temp.increment_counts(self.rename(instance, mapping))
            score = temp._expected_correct_guesses()
            scored_mappings.append((score, mapping))

        best_mapping = sorted(scored_mappings, key=lambda x: x[0])[0][1]
        return best_mapping

    def match(self, instance):
        """ 
        Define the specialized matching function to rename components
        and relations to maximize the match between instance and the
        current concept (self).
        """
        #mapping = self.exhaustivematch(instance)
        mapping = self.hungarianmatch(instance)
        temp_instance = self.rename(instance, mapping)
        return temp_instance

    def is_parent(self, other_concept):
        """
        Returns True if self is a parent of other concept.
        """
        temp = other_concept
        while temp != None:
            if temp == self:
                return True
            temp = temp.parent
        return False

    def common_ancestor(self, val1, val2):
        """
        Returns the nearest common ancestor of val1 and val2.
        """
        if val1.is_parent(val2):
            return val1
        ancestor = val2
        while ancestor.parent:
            if ancestor.is_parent(val1):
                return ancestor
            ancestor = ancestor.parent
        return ancestor

if __name__ == "__main__":

    tree = LabyrinthTree()

    with open('data_files/rb_s_07_continuous.json', "r") as json_data:
        instances = json.load(json_data)
    print(len(instances))
    instances = instances[0:20]
    print(set(tree.cluster(instances, 2)))

