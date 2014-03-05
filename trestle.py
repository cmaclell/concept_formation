import itertools
import hungarianNative
import numpy
import json
import copy
from itertools import combinations
from random import normalvariate
from random import choice
from random import random
from random import shuffle
from cobweb3 import Cobweb3

class Trestle(Cobweb3):

    def replace_value(self, attr, old, new):
        """
        Replaces the old value of an attribute with a new value.
        """
        if old == new:
            return
        if attr not in self.av_counts:
            return
        if old not in self.av_counts[attr]:
            return

        if new not in self.av_counts[attr]:
            self.av_counts[attr][new] = 0.0

        self.av_counts[attr][new] += self.av_counts[attr][old]
        del self.av_counts[attr][old]

    def depth(self):
        if self.parent:
            return 1 + self.parent.depth()
        return 0

    def attribute_generalize(self):
        """
        This function will hill climb over the possible av generalizations
        until no further category utility gain is possible.
        """
        current = self
        current_cu = super(Trestle, self).category_utility()

        for attr in self.av_counts:
            values = [v for v in self.av_counts[attr] if isinstance(v, Trestle)]
            
            #possible_generalizations = []
            #for v in values:
            #    count = 0
            #    curr = v
            #    possible_generalizations.append((count, curr))
            #    while curr.parent:
            #        count += 1
            #        possible_generalizations.append((count, curr.parent))
            #        curr = curr.parent

            possible_generalizations = set([self.common_ancestor(v1,v2)
                                                 for (v1,v2) in
                                                 combinations(values,2)])

            possible_generalizations = sorted(list(
                set(possible_generalizations)), key=lambda x: -1.0 * x.depth())
            #print(possible_generalizations2)

            while possible_generalizations:
                new = possible_generalizations.pop()
            #for new in possible_generalizations:
                temp = current.shallow_copy()
                
                #old_values = set([v for c in temp.children if attr in
                #                  c.av_counts for v in c.av_counts[attr]])
                old_values = [v for v in temp.av_counts[attr]]
                for old in old_values:
                    if not new.is_parent(old):
                        continue
                    temp.replace_value(attr, old, new)
                    for c in temp.children:
                        c.replace_value(attr, old, new)
                cu = super(Trestle, temp).category_utility()
                
                if cu >= current_cu:
                    for v in possible_generalizations:
                        if new.is_parent(v):
                            possible_generalizations.remove(v)
                    current_cu = cu
                    current = temp
                
        return (current, current_cu)

    def category_utility(self):
        """
        Performs attribute value generalization before computing traditional
        CU.
        """
        temp = self.shallow_copy()
        temp, cu = self.attribute_generalize()
        return cu

    def val_existance_check(self):
        """
        Used to ensure that all the values in the tree exist in the tree.
        """
        root = self.get_root()
        for attr in self.av_counts:
            for val in self.av_counts[attr]:
                if isinstance(val, Trestle):
                    assert val.get_root() == root
        for c in self.children:
            c.val_existance_check()

    def val_leaves_check(self):
        """
        Used to ensure all values are leaves.
        """
        for attr in self.av_counts:
            for val in self.av_counts[attr]:
                if isinstance(val, Trestle):
                    assert not val.children
        for c in self.children:
            c.val_leaves_check()

    def val_representation_check(self):
        """
        Used to ensure he level of representation of a component attribute is
        consistent.
        """
        for attr in self.av_counts:
            for val in self.av_counts[attr]:
                if isinstance(val, Trestle):
                    for val2 in self.av_counts[attr]:
                        if isinstance(val2, Trestle):
                            if val == val2:
                                continue
                            assert not val.is_parent(val2)
                            assert not val2.is_parent(val)
        for c in self.children:
            c.val_representation_check()

    def remove_reference(self, node):
        """
        Specialized version of remove for Trestle. This removes all references
        to a particular concept from the tree. It replaces these references
        with a reference to the parent concept
        """
        # call recursively
        for c in self.children:
            c.remove_reference(node)

        # replace references to deleted concept with parent concept
        self.get_root().replace(self, node)

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

    def Trestle(self, instance):
        """
        Recursively calls Trestle on all of the components in a depth-first
        traversal. Once all of the components have been classified then then it
        classifies the current node.
        """
        temp_instance = {}
        attributes = sorted([attr for attr in instance])
        shuffle(attributes)
        for attr in attributes:
        #for attr in instance:
            if isinstance(instance[attr], dict):
                temp_instance[attr] = self.Trestle(instance[attr])
            elif isinstance(instance[attr], list):
                pass
                #temp_instance[tuple(instance[attr])] = True
            else:
                temp_instance[attr] = instance[attr]

        # Ensure all components are leaves
        for attr in temp_instance:
            if (isinstance(temp_instance[attr], Trestle) and
                temp_instance[attr].children):
                temp_instance[attr] = temp_instance[attr].trestle_categorize(instance[attr])

        # should be able to match just at the root, if the matchings change
        # than the counts between parent and child will be thrown off which is
        # not allowed to happen so for now don't worry about it.
        # TODO check if this needs to be changed
        temp_instance = self.match(temp_instance)
        ret = self.cobweb(temp_instance)

        return ret

    def rename(self, instance, mapping):
        """
        Given a mapping (type = dict) rename the 
        components and relations and return the renamed
        instance.
        """
        # Ensure it is a complete mapping
        # Might be troublesome if there is a name collision
        for attr in instance:
            if not isinstance(instance[attr], Trestle):
                continue
            if attr not in mapping:
                mapping[attr] = attr

        temp_instance = {}
        relations = []

        # rename all attribute values
        for attr in instance:
            if isinstance(attr, tuple):
                relations.append(attr)
            elif isinstance(instance[attr], Trestle):
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

    def hungarian_match(self, instance):
        """
        Compute the greedy match between the instance and the current concept.
        This algorithm is O(n^3) and will return the optimal match if there are
        no relations. However, when there are relations, then it is only
        computing the best greedy match (no longer optimal). 
        """

        # FOR DEBUGGING
        #print("INSTANCE")
        #for attr in instance:
        #    if isinstance(instance[attr], Trestle):
        #        print(attr + ": " + str(instance[attr].concept_name))

        #print("CONCEPT")
        #for attr in self.av_counts:
        #    for val in self.av_counts[attr]:
        #        if isinstance(val, Trestle):
        #            print(attr + ": " + str(val.concept_name))

        from_name = [attr for attr in instance if isinstance(instance[attr],
                                                             Trestle)]
        to_name = []
        for attr in self.av_counts:
            for val in self.av_counts[attr]:
                if isinstance(val, Trestle):
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

                #normal Trestle nominal style match
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
                                                             Trestle)]
        to_name = []
        for attr in self.av_counts:
            for val in self.av_counts[attr]:
                if isinstance(val, Trestle):
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
        mapping = self.hungarian_match(instance)
        temp_instance = self.rename(instance, mapping)
        return temp_instance

    def output_json(self):
        """
        A modification of output_json from cobweb and cobweb3 to handle
        component values.
        """
        output = {}
        output["name"] = self.concept_name
        output["size"] = self.count
        if self.children:
            output["CU"] = self.category_utility()
        output["children"] = []

        # generalize av counts
        gself, cu = self.attribute_generalize()

        temp = {}
        for attr in self.av_counts:
            float_vals = []
            for value in self.av_counts[attr]:
                if isinstance(attr, tuple):
                    temp["[" + " ".join(attr) + "]"] = self.av_counts[attr][True]
                elif isinstance(value, float):
                    float_vals.append(value)
                elif isinstance(value, Trestle): 
                    temp[attr + " = " + value.concept_name] = self.av_counts[attr][value]
                else:
                    temp[attr + " = " + str(value)] = self.av_counts[attr][value]
            if len(float_vals) > 0:
                mean = attr + "_mean = %0.2f (%0.2f)" % (self._mean(float_vals),
                                                self._std(float_vals))
                temp[mean] = len(float_vals)

        temp2 = {}
        for attr in gself.av_counts:
            float_vals = []
            for value in gself.av_counts[attr]:
                if isinstance(attr, tuple):
                    temp2["[" + " ".join(attr) + "]"] = gself.av_counts[attr][True]
                elif isinstance(value, float):
                    float_vals.append(value)
                elif isinstance(value, Trestle): 
                    temp2[attr + " = " + value.concept_name] = gself.av_counts[attr][value]
                else:
                    temp2[attr + " = " + str(value)] = gself.av_counts[attr][value]
            if len(float_vals) > 0:
                mean = attr + "_mean = %0.2f (%0.2f)" % (self._mean(float_vals),
                                                self._std(float_vals))
                temp2[mean] = len(float_vals)
                
        for child in self.children:
            output["children"].append(child.output_json())

        output["counts"] = temp
        output['general counts'] = temp2

        return output

    def trestle_categorize(self, instance):
        """
        The Trestle categorize function, this Trestle categorizes all the
        sub-components before categorizing itself.
        """
        temp_instance = {}
        for attr in instance:
            if isinstance(instance[attr], dict):
                temp_instance[attr] = self.trestle_categorize(instance[attr])
            elif isinstance(instance[attr], list):
                temp_instance[attr] = tuple(instance[attr])
            else:
                temp_instance[attr] = instance[attr]

        # should be able to match just at the root, if the matchings change
        # than the counts between parent and child will be thrown off which is
        # not allowed to happen so for now don't worry about it.
        # TODO check if this needs to be changed
        temp_instance = self.match(temp_instance)
        return self.cobweb_categorize(temp_instance)

    def ifit(self, instance):
        """
        A modification of ifit to call Trestle instead.
        """
        return self.Trestle(instance)

    def pretty_print(self, depth=0):
        """
        Prints the categorization tree. Modified for component values.
        """
        tabs = "\t" * depth
        if self.parent:
            ret = str(('\t' * depth) + "|-" + "[" + self.concept_name + "(" +
                      self.parent.concept_name + ")]: " +
                      str(self.count) + "\n" + tabs + "  ")
        else:
            ret = str(('\t' * depth) + "|-" + "[" + self.concept_name + "(" +
                      "None)]: " +
                      str(self.count) + "\n" + tabs + "  ")

        attributes = []

        for attr in self.av_counts:
            float_values = []
            values = []

            for val in self.av_counts[attr]:
                if isinstance(val, float):
                    float_values.append(val)
                elif isinstance(val, Trestle):
                    values.append("'" + val.concept_name + "': " + 
                                  str(self.av_counts[attr][val]))
                else:
                    values.append("'" + str(val) + "': " +
                                  str(self.av_counts[attr][val]))

            if float_values:
                values.append("'mean':" + str(self._mean(float_values)))
                values.append("'std':" + str(self._std(float_values)))

            attributes.append("'" + str(attr) + "': {" + ", ".join(values) + "}")
                  
        ret += "{" + (",\n" + tabs + "   ").join(attributes) + "}\n"
        
        for c in self.children:
            ret += c.pretty_print(depth+1)

        return ret

    def concept_attr_value(self, instance, attr, val):
        """
        A modification to call Trestle categorize instead of cobweb
        categorize.
        """
        concept = self.trestle_categorize(instance)
        return concept.get_probability(attr, val)

    def flexible_prediction(self, instance, guessing=False):
        """
        A modification of flexible prediction to handle component values.
        The flexible prediction task is called on all subcomponents. To compute
        the accuracy for each subcomponent.
        """
        
        probs = []
        for attr in instance:
            #TODO add support for relational attribute values 
            if isinstance(instance[attr], list):
                continue
            if isinstance(instance[attr], dict):
                probs.append(self.flexible_prediction(instance[attr], guessing))
                continue
            temp = {}
            for attr2 in instance:
                if attr == attr2:
                    continue
                temp[attr2] = instance[attr2]
            if guessing:
                probs.append(self.get_probability(attr, instance[attr]))
            else:
                probs.append(self.concept_attr_value(temp, attr, instance[attr]))
        if len(probs) == 0:
            print(instance)
            return -1 
        return sum(probs) / len(probs)

    def predict(self, instance):
        """
        Given an instance predict any missing attribute values without
        modifying the tree. A modification for component values.
        """
        prediction = {}

        # make a copy of the instance
        # call recursively on structured parts
        for attr in instance:
            if isinstance(instance[attr], dict):
                prediction[attr] = self.predict(instance[attr])
            else:
                prediction[attr] = instance[attr]

        concept = self.trestle_categorize(prediction)
        #print(concept)
        #print(self)
        
        for attr in concept.av_counts:
            if attr in prediction:
                continue
           
            # sample to determine if the attribute should be included
            num_attr = sum([concept.av_counts[attr][val] for val in
                            concept.av_counts[attr]])
            if random() > (1.0 * num_attr) / concept.count:
                continue
            
            nominal_values = []
            component_values = []
            float_values = []

            for val in concept.av_counts[attr]:
                if isinstance(val, float):
                    float_values += [val] * concept.av_counts[attr][val]
                elif isinstance(val, Trestle):
                    component_values += [val] * concept.av_counts[attr][val] 
                else:
                    nominal_values += [val] * concept.av_counts[attr][val]

            rand = random()

            if rand < ((len(nominal_values) * 1.0) / (len(nominal_values) +
                                                      len(component_values) +
                                                      len(float_values))):
                prediction[attr] = choice(nominal_values)
            elif rand < ((len(nominal_values) + len(component_values) * 1.0) /
                         (len(nominal_values) + len(component_values) +
                          len(float_values))):
                prediction[attr] = choice(component_values).predict({})
            else:
                prediction[attr] = normalvariate(self._mean(float_values),
                                                 self._std(float_values))

        return prediction

    def sequential_prediction(self, filename, length, guessing=False):
        """
        Given a json file, perform an incremental sequential prediction task. 
        Try to flexibly predict each instance before incorporating it into the 
        tree. This will give a type of cross validated result.
        """
        json_data = open(filename, "r")
        instances = json.load(json_data)
        #instances = instances[0:length]
        for instance in instances:
            if "guid" in instance:
                del instance['guid']
        accuracy = []
        nodes = []
        for j in range(1):
            shuffle(instances)
            for n, i in enumerate(instances):
                if n >= length:
                    break
                accuracy.append(self.flexible_prediction(i, guessing))
                nodes.append(self.num_concepts())
                self.ifit(i)
        json_data.close()
        return accuracy, nodes

    def flatten_instance(self, instance):
        duplicate = copy.deepcopy(instance)
        for attr in duplicate:
            if isinstance(duplicate[attr], dict):
                duplicate[attr] = self.flatten_instance(duplicate[attr])
        return repr(sorted(duplicate.items()))

    def order_towers(self):
        """
        Given a number of towers with GUIDs added return a better
        training ordering.
        """
        L = []
        if not self.children:
            if 'guid' in self.av_counts:
                for guid in self.av_counts['guid']:
                    L.append(guid)
            return L
        else:
            sorted_c = sorted(self.children, key=lambda c: -1 * c.count)
            lists = []

            for c in sorted_c:
                lists.append(c.order_towers())

            while lists:
                for l in lists:
                    if l:
                        L.append(l.pop())
                        
                lists = [l for l in lists if l]
            return L

    def replace(self, old, new):
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
            c.replace(old,new)

    def verify_pointers(self):
        for c in self.children:
            assert c.parent == self
            c.verify_pointers()     

    def get_root(self):
        """
        Gets the root of the categorization tree.
        """
        if self.parent == None:
            return self
        else:
            return self.parent.get_root()

    def create_child_with_current_counts(self):
        """
        Creates a new child (to the current node) with the counts initialized by
        the current node's counts.
        """
        if self.count > 0:
            new = self.__class__(self)
            new.parent = self
            self.children.append(new)

            # TODO may be a more efficient way to do this, just ensure the
            # pointer stays a leaf in the main cobweb alg. for instance.

            self.get_root().replace(self, new)
            return new

    def split(self, best):
        """
        Specialized version of split for Trestle. This removes all references
        to a particular concept from the tree. It replaces these references
        with a reference to the parent concept
        """
        super(Trestle, self).split(best)
        
        # replace references to deleted concept with parent concept
        self.get_root().replace(best, self)

    def verify_component_values(self):
        for attr in self.av_counts:
            for val in self.av_counts[attr]:
                if isinstance(val, Trestle):
                    assert not val.children
        for c in self.children:
            c.verify_component_values()

    def verify_parent_pointers(self):
        for c in self.children:
            assert c.parent == self
            c.verify_parent_pointers()

    def cluster(self, filename, length):
        """
        Used to provide a clustering of a set of examples provided in a JSON
        file. It starts by incorporating the examples into the categorization
        tree multiple times. After incorporating the instances it then
        categorizes each example (without updating the tree) and returns the
        concept it was assoicated with.
        """
        json_data = open(filename, "r")
        instances = json.load(json_data)
        shuffle(instances)
        instances = instances[0:length]
        o_instances = copy.deepcopy(instances)
        for instance in instances:
            if "guid" in instance:
                del instance['guid']
        json_data.close()
        clusters = {}
        previous = {}
        g_instances = {}

        for i in instances:
            previous[self.flatten_instance(i)] = None

        # train initially
        for x in range(1):
            shuffle(instances)
            for n, i in enumerate(instances):
                print("training instance: " + str(n))
                self.ifit(i)

        #TODO for debugging
        #self.verify_counts()
        #self.verify_component_values()

        #add categorize for adding guids
        mapping = {}
        for idx, inst in enumerate(o_instances):
            print("categorizing instance: %i" % idx)
            instance = copy.deepcopy(inst)
            if "guid" in instance:
                del instance['guid']
            g_instances[inst['guid']] = instance

            mapping[inst['guid']] = self.trestle_categorize(instance)

        # add guids
        for g in mapping:
            curr = mapping[g]
            while curr:
                curr.av_counts['has-guid'] = {"1":True}
                if 'guid' not in curr.av_counts:
                    curr.av_counts['guid'] = {}
                curr.av_counts['guid'][g] = True
                curr = curr.parent
        
        ## get ordering
        #guid_order = self.order_towers()
        #self = self.__class__()

        ## second time sorting
        #count = 0
        #for guid in guid_order:
        #    count += 1
        #    print("training instance: " + str(count))
        #    self.ifit(g_instances[guid])

        ## add categorize for adding guids
        #mapping = {}
        #for idx, inst in enumerate(o_instances):
        #    print("categorizing instance: %i" % idx)
        #    instance = copy.deepcopy(inst)
        #    if "guid" in instance:
        #        del instance['guid']

        #    mapping[inst['guid']] = self.trestle_categorize(instance)

        ## add guids
        #for g in mapping:
        #    curr = mapping[g]
        #    while curr:
        #        curr.av_counts['has-guid'] = {"1":True}
        #        if 'guid' not in curr.av_counts:
        #            curr.av_counts['guid'] = {}
        #        curr.av_counts['guid'][g] = True
        #        curr = curr.parent
        
        for g in mapping:
            cluster = mapping[g]
            cluster = cluster.parent
            clusters[g] = cluster.concept_name

        print(json.dumps(self.output_json()))

        return clusters

if __name__ == "__main__":

    #Trestle().predictions("data_files/rb_com_11_noCheck.json", 15, 10)
    #print(Trestle().cluster("data_files/rb_com_11_noCheck.json", 300))
    print(Trestle().cluster("data_files/rb_test_continuous.json", 300))

    #Trestle().predictions("data_files/kelly-data.json", 5, 1)
    #print(Trestle().cluster("data_files/kelly-data.json", 10))
    #Trestle().baseline_guesser("data_files/rb_com_11_noCheck.json", 10, 1)

    #print(Trestle().cluster("data_files/rb_s_07.json", 10, 3))
    #print(Trestle().cluster("data_files/jenny_graph_data.json", 50, 1))

    #t = Trestle()
    #t.sequential_prediction("towers_small_trestle.json", 10)
    #print(t.predict({"success": "1"}))

