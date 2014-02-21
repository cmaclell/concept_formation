import itertools
import hungarianNative
import numpy
import json
import copy
from random import normalvariate
from random import choice
from random import random
from random import shuffle
from cobweb3 import Cobweb3Tree

class Labyrinth(Cobweb3Tree):

    def _is_parent(self, other_concept):
        temp = other_concept
        while temp != None:
            if temp == self:
                return True
            temp = temp.parent
        return False

    def _labyrinth(self, instance):
        """
        Recursively calls labyrinth on all of the components in a depth-first
        traversal. Once all of the components have been classified then then it
        classifies the current node.
        """
        temp_instance = {}
        attributes = sorted([attr for attr in instance])
        shuffle(attributes)
        for attr in attributes:
        #for attr in instance:
            if isinstance(instance[attr], dict):
                temp_instance[attr] = self._labyrinth(instance[attr])
            elif isinstance(instance[attr], list):
                pass
                #temp_instance[tuple(instance[attr])] = True
            else:
                temp_instance[attr] = instance[attr]

        # should be able to match just at the root, if the matchings change
        # than the counts between parent and child will be thrown off which is
        # not allowed to happen so for now don't worry about it.
        # TODO check if this needs to be changed
        temp_instance = self._match(temp_instance)
        ret = self._cobweb(temp_instance)

        return ret

    def _rename(self, instance, mapping):
        """
        Given a mapping (type = dict) rename the 
        components and relations and return the renamed
        instance.
        """
        # Ensure it is a complete mapping
        # Might be troublesome if there is a name collision
        for attr in instance:
            if not isinstance(instance[attr], Labyrinth):
                continue
            if attr not in mapping:
                mapping[attr] = attr

        temp_instance = {}
        relations = []

        # rename all attribute values
        for attr in instance:
            if isinstance(attr, tuple):
                relations.append(attr)
            elif isinstance(instance[attr], Labyrinth):
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

    def _hungarian_match(self, instance):
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
                if from_val in self.av_counts[to_name[col_index]]:
                    reward = (((1.0 * self.av_counts[to_name[col_index]][from_val]) /
                              self.count))

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

    def _exhaustive_match(self, instance):
        """
        Compute matches exhaustively. This has O(n!) time complexity, so it
        won't scale very well.
        """
        from_name = [attr for attr in instance if isinstance(instance[attr],
                                                             Labyrinth)]
        to_name = []
        for attr in self.av_counts:
            for val in self.av_counts[attr]:
                if isinstance(val, Labyrinth):
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
            temp._increment_counts(self._rename(instance, mapping))
            score = temp._expected_correct_guesses()
            scored_mappings.append((score, mapping))

        best_mapping = sorted(scored_mappings, key=lambda x: x[0])[0][1]
        return best_mapping

    def _match(self, instance):
        """ 
        Define the specialized matching function to rename components
        and relations to maximize the match between instance and the
        current concept (self).
        """
        #mapping = self._exhaustive_match(instance)
        mapping = self._hungarian_match(instance)
        temp_instance = self._rename(instance, mapping)
        return temp_instance

    def _output_json(self):
        """
        A modification of output_json from cobweb and cobweb3 to handle
        component values.
        """
        output = {}
        output["name"] = self.concept_name
        output["size"] = self.count
        output["children"] = []

        temp = {}
        for attr in self.av_counts:
            float_vals = []
            for value in self.av_counts[attr]:
                if isinstance(attr, tuple):
                    temp["[" + " ".join(attr) + "]"] = self.av_counts[attr][True]
                elif isinstance(value, float):
                    float_vals.append(value)
                elif isinstance(value, Labyrinth): 
                    temp[attr + " = " + value.concept_name] = self.av_counts[attr][value]
                else:
                    temp[attr + " = " + str(value)] = self.av_counts[attr][value]
            if len(float_vals) > 0:
                mean = attr + "_mean = %0.2f (%0.2f)" % (self._mean(float_vals),
                                                self._std(float_vals))
                temp[mean] = len(float_vals)
                
        for child in self.children:
            output["children"].append(child._output_json())

        output["counts"] = temp

        return output

    def _labyrinth_categorize(self, instance):
        """
        The labyrinth categorize function, this labyrinth categorizes all the
        sub-components before categorizing itself.
        """
        temp_instance = {}
        for attr in instance:
            if isinstance(instance[attr], dict):
                temp_instance[attr] = self._labyrinth_categorize(instance[attr])
            elif isinstance(instance[attr], list):
                temp_instance[attr] = tuple(instance[attr])
            else:
                temp_instance[attr] = instance[attr]

        # should be able to match just at the root, if the matchings change
        # than the counts between parent and child will be thrown off which is
        # not allowed to happen so for now don't worry about it.
        # TODO check if this needs to be changed
        temp_instance = self._match(temp_instance)
        return self._cobweb_categorize(temp_instance)

    def ifit(self, instance):
        """
        A modification of ifit to call labyrinth instead.
        """
        return self._labyrinth(instance)

    def _pretty_print(self, depth=0):
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
                elif isinstance(val, Labyrinth):
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
            ret += c._pretty_print(depth+1)

        return ret

    def _concept_attr_value(self, instance, attr, val):
        """
        A modification to call labyrinth categorize instead of cobweb
        categorize.
        """
        concept = self._labyrinth_categorize(instance)
        return concept._get_probability(attr, val)

    def _flexible_prediction(self, instance, guessing=False):
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
                probs.append(self._flexible_prediction(instance[attr], guessing))
                continue
            temp = {}
            for attr2 in instance:
                if attr == attr2:
                    continue
                temp[attr2] = instance[attr2]
            if guessing:
                probs.append(self._get_probability(attr, instance[attr]))
            else:
                probs.append(self._concept_attr_value(temp, attr, instance[attr]))
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

        concept = self._labyrinth_categorize(prediction)
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
                elif isinstance(val, Labyrinth):
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
                accuracy.append(self._flexible_prediction(i, guessing))
                nodes.append(self._num_concepts())
                self.ifit(i)
        json_data.close()
        return accuracy, nodes

    def _flatten_instance(self, instance):
        duplicate = copy.deepcopy(instance)
        for attr in duplicate:
            if isinstance(duplicate[attr], dict):
                duplicate[attr] = self._flatten_instance(duplicate[attr])
        return repr(sorted(duplicate.items()))

    def cluster(self, filename, length, iterations=100):
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
        diff = 1
        counter = 0
        
        previous = {}
        for i in instances:
            previous[self._flatten_instance(i)] = None

        while diff > 0 and counter < iterations:
            counter += 1
        #for j in range(iterations):
            #before = self._num_concepts()
            shuffle(instances)
            diff = 0
            for n, i in enumerate(instances):
                print("training instance: " + str(n))
                self.ifit(i)

            for n, i in enumerate(instances):
                print("categorizing instance: " + str(n))
                cluster = self._labyrinth_categorize(i).parent
                if (previous[self._flatten_instance(i)] != cluster):
                    diff += 1
                    previous[self._flatten_instance(i)] = cluster

            print(diff)
            #print(self._num_concepts())
            #diff = abs(before - self._num_concepts())
            #print(json.dumps(self._output_json()))
       

        #self._remove_singletons()
        mapping = {}
        for idx, inst in enumerate(o_instances):
            instance = copy.deepcopy(inst)
            if "guid" in instance:
                del instance['guid']
            #print(inst['guid'])
            #print(previous[self._flatten_instance(instance)].concept_name)
            mapping[inst['guid']] = self._labyrinth_categorize(instance)
            #concept = mapping[inst['guid']].parent.concept_name
            #print(concept)
            #clusters[inst['guid']] = self._labyrinth_categorize(instance).parent.concept_name
            #clusters[inst['guid']] = concept

        for g in mapping:
            curr = mapping[g]
            while curr:
                curr.av_counts['has-guid'] = {"1":True}
                if 'guid' not in curr.av_counts:
                    curr.av_counts['guid'] = {}
                curr.av_counts['guid'][g] = True
                curr = curr.parent
        
        for g in mapping:
            cluster = mapping[g]
            num = 1
            while num <= 1 and cluster.parent:
                cluster = cluster.parent
                num = 0
                for c in cluster.children:
                    if 'has-guid' in c.av_counts:
                        num += 1
            clusters[g] = cluster.concept_name

        print(json.dumps(self._output_json()))

        return clusters

if __name__ == "__main__":

    print(Labyrinth().cluster("data_files/rb_com_11_noCheck.json", 20, 100))
    #print(Labyrinth().cluster("data_files/rb_s_07.json", 10, 3))
    #print(Labyrinth().cluster("data_files/jenny_graph_data.json", 50, 1))
    #Labyrinth().predictions("data_files/rb_com_11_noCheck.json", 15, 3)
    #Labyrinth().baseline_guesser("data_files/rb_com_11_noCheck.json", 10, 1)

    #t = Labyrinth()
    #t.sequential_prediction("towers_small_trestle.json", 10)
    #print(t.predict({"success": "1"}))



