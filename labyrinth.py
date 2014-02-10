import itertools
import math
import hungarianNative
import numpy
import json
from random import normalvariate
from random import choice
from random import random
from random import shuffle
from cobweb3 import Cobweb3Tree

class Labyrinth(Cobweb3Tree):

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
                #pass
                temp_instance[tuple(instance[attr])] = True
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

                reward = 0.0
                from_val = instance[from_name[row_index]]
                if from_val in self.av_counts[to_name[col_index]]:
                    reward = (((1.0 * self.av_counts[to_name[col_index]][from_val]) /
                              self.count))

                row.append(max_cost - (reward*reward))
                    
            cost_matrix.append(row)

        a = numpy.array(cost_matrix)

        # may be able to eliminate this duplicate
        b = numpy.array(cost_matrix)

        #depreciated c library approach
        #assignment1 = hungarian.lap(a)[0]

        ### substitute hungarian method ####
        assignment = hungarianNative.hungarian(a)
        
        ### substitute hungarian method ####
        mapping = {}
        
        for index, val in enumerate(assignment):
            if (index >= len(from_name)):
                continue
            elif (val >= len(to_name)):
                mapping[from_name[index]] = "component" + self._gensym()
            #elif (b[index][val] == max_cost):
            #    mapping[from_name[index]] = "component" + self._gensym()
            else:
                mapping[from_name[index]] = to_name[val]
                
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

    #def _cobweb(self, instance):
    #    super(Labyrinth, self)._cobweb(instance)

    def _output_json(self):
        output = {}
        output['name'] = self.concept_name
        output['size'] = self.count
        output['children'] = []

        temp = {}
        for attr in self.av_counts:
            if isinstance(attr, tuple):
                continue
            float_vals = []
            for value in self.av_counts[attr]:
                if isinstance(value, float):
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
            output['children'].append(child._output_json())

        output['counts'] = temp

        return output

    #def _output_json(self):
    #    output = {}
    #    output['name'] = self.concept_name
    #    output['size'] = self.count
    #    for attr in self.av_counts:
    #        name = attr
    #        if isinstance(name, tuple):
    #           name = "(" + " ".join(name) + ")"
    #        for value in self.av_counts[attr]:
    #            vname = value
    #            if isinstance(vname, Labyrinth):
    #                vname = vname.concept_name
    #            output[name] = str(vname) #+ "(" + str(self.av_counts[attr][value]) + ")"
    #    if len(self.children) > 0:
    #        output['children'] = []
    #        for child in self.children:
    #            output['children'].append(child._output_json())
    #    else:
    #        if 'guid' in self.av_counts:
    #            for v in self.av_counts['guid']:
    #                output['guid'] = v

    #    return output


    def _labyrinth_categorize(self, instance):
        temp_instance = {}
        for attr in instance:
            if isinstance(instance[attr], dict):
                temp_instance[attr] = self._labyrinth_categorize(instance[attr])
            elif isinstance(instance[attr], list):
                #pass
                temp_instance[attr] = tuple(instance[attr])
            else:#COVERTEN
                temp_instance[attr] = instance[attr]

        # should be able to match just at the root, if the matchings change
        # than the counts between parent and child will be thrown off which is
        # not allowed to happen so for now don't worry about it.
        # TODO check if this needs to be changed
        temp_instance = self._match(temp_instance)
        return self._cobweb_categorize(temp_instance)

    def ifit(self, instance):
        self._labyrinth(instance)

    def _pretty_print(self, depth=0):
        """
        Prints the categorization tree.
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

    def _get_probability(self, attr, val):
        if attr not in self.av_counts:
            return 0.0

        if isinstance(val, Labyrinth):
            raise Exception("can't get probability of component value.")

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
            #print(point)
            return (1.0 - math.erf(point / math.sqrt(2)))#/2.0
        
        if val in self.av_counts[attr]:
            return (1.0 * self.av_counts[attr][val]) / self.count

        return 0.0

    def _prob_attr_value(self, instance, attr, val):
        concept = self._labyrinth_categorize(instance)

        #TODO don't think I need this.  
        #if isinstance(val, list):
        #    temp_instance = {}

        #    for attr in instance:
        #        if isinstance(instance[attr], dict):
        #            temp_instance[attr] = self._labyrinth(instance[attr])
        #        elif isinstance(instance[attr], list):
        #            temp_instance[tuple(instance[attr])] = True
        #        else:
        #            temp_instance[attr] = instance[attr]

        #    mapping = self._exhaustive_match(temp_instance)
        #    print(mapping)

        #    new_val = []
        #    for i in range(len(val)):
        #        if i == 0:
        #            new_val.append(val[i])
        #            continue
        #        new_val.append(mapping[val[i]])
        #    attr = tuple(new_val)
        #    val = True
        #    return concept._get_probability(attr, val)

        return concept._get_probability(attr, val)

    def _specific_prediction(self, instance, attr):
        if attr in instance:
            #TODO add support for relational attribute values 
            if isinstance(instance[attr], list):
                return
            if isinstance(instance[attr], dict):
                #probs.append(self._flexible_prediction(instance[attr]))
                return
            temp = {}
            for attr2 in instance:
                if attr == attr2:
                    continue
                temp[attr2] = instance[attr2]
            return self._prob_attr_value(temp, attr, instance[attr])

    def _flexible_prediction(self, instance):
        probs = []
        for attr in instance:
            #TODO add support for relational attribute values 
            if isinstance(instance[attr], list):
                continue
            if isinstance(instance[attr], dict):
                #temp = self._flexible_prediction(instance[attr])
                #if temp == -1:
                #    print(instance)
                #else:
                #    probs.append(temp)
                probs.append(self._flexible_prediction(instance[attr]))
                continue
            temp = {}
            for attr2 in instance:
                if attr == attr2:
                    continue
                temp[attr2] = instance[attr2]
            probs.append(self._prob_attr_value(temp, attr, instance[attr]))
        if len(probs) == 0:
            print(instance)
            return -1 
        return sum(probs) / len(probs)

    def predict(self, instance):
        """
        Given an instance predict any missing attribute values without
        modifying the tree.
        """
        prediction = {}

        # make a copy of the instance
        # call recursively on structured parts
        for attr in instance:
            if isinstance(instance[attr], dict):
                prediction[attr] = self.predict(instance[attr])
            else:
                prediction[attr] = instance[attr]

        prediction = self._match(prediction)
        concept = self._labyrinth_categorize(prediction)
        #print(concept)
        
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

    def cluster(self, filename, length):
        json_data = open(filename, "r")
        instances = json.load(json_data)
        json_data.close()
        instances = instances[0:length]
        clusters = []
        for j in range(3):
            print("run %i" % j)
            shuffle(instances)
            for n, i in enumerate(instances):
                if n >= length:
                    break
                self.ifit(i)
        print(json.dumps(self._output_json()))
        for n, i in enumerate(instances):
            concept = self._labyrinth_categorize(i)
            if len(concept.children) == 0:
                concept = concept.parent
            clusters.append(concept.concept_name)
        return clusters

    def predictions(self, filename, length):
        n = 1 
        runs = []
        for i in range(0,n):
            print("run %i" % i)
            t = Labyrinth()
            runs.append(t.sequential_prediction(filename,
                                               length))
            print(json.dumps(t._output_json()))
            #runs.append(t.sequential_prediction("really_small.json", 10))

        #print(runs)
        print("MEAN")
        for i in range(0,len(runs[0])):
            a = []
            for r in runs:
                a.append(r[i])
            print("%0.2f" % (Labyrinth()._mean(a)))
            #print("mean: %0.2f, std: %0.2f" % (Labyrinth()._mean(a),
            #                                   Labyrinth()._std(a)))
        print()
        print("STD")
        for i in range(0,len(runs[0])):
            a = []
            for r in runs:
                a.append(r[i])
            print("%0.2f" % (Labyrinth()._std(a)))



if __name__ == "__main__":

    #t.train_from_json("labyrinth_test.json")
    #t.train_from_json("towers_small_trestle.json")
    #print(Labyrinth().cluster("towers_small_trestle.json", 15))
    print(Labyrinth().predictions("towers_small_trestle.json", 15))


    #print(t)

    #test = {}
    #test['success'] = '1'
    #print(t.predict(test))



