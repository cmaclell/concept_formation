import re
import itertools
import hungarianNative
import numpy
import json
import copy
import pickle
from itertools import combinations
from random import normalvariate
from random import choice
from random import random
from random import shuffle
from random import normalvariate
from cobweb3 import Cobweb3
from cobweb3 import ContinuousValue

class Trestle(Cobweb3):

    def flatten(self, instance):
        """
        Takes a hierarchical instance and flattens it. It represents
        hierarchy with periods in variable names. It also converts the 
        relations into tuples with values.
        """
        temp = {}
        for attr in instance:
            if isinstance(instance[attr], dict):
                subobject = self.flatten(instance[attr])
                for so_attr in subobject:
                    if isinstance(subobject[so_attr], list):
                        relation = []
                        for idx, val in enumerate(subobject[so_attr]):
                            if idx == 0:
                                relation.append(val)
                            else:
                                relation.append(attr + "." + val)
                        temp[tuple(relation)] = True
                        #temp[attr + "." + so_attr] = relation
                    else:
                        # annotate attributes to ignore
                        if so_attr[0] == "_":
                            temp["_" + attr + "." + so_attr] = instance[attr][so_attr]
                        else:
                            temp[attr + "." + so_attr] = instance[attr][so_attr]
            else:
                if isinstance(instance[attr], list):
                    temp[tuple(instance[attr])] = True
                else:
                    temp[attr] = instance[attr]
        return temp

    def align(self, instance):
        """
        NOT used, but interesting idea about how to shift focus of attention to
        maximize the match of continuous values... right?

        Given a matched instance an a concept, iterate through each numeric
        attribute (across structured objects and apply a shift to bring all
        matches into closer alignment... If attributes are linked (such as left
        and right) then they SHOULD match up relatively closely... however they
        may not.
        """
        # collect shiftable attributes
        errors = {}
        for attr in instance:
            if isinstance(instance[attr], float):
                name = attr.split(".")[-1]
                if name not in errors:
                    errors[name] = []
                if attr in self.av_counts:
                    errors[name].append(self.av_counts[attr].mean -
                                        instance[attr])
                else:
                    errors[name].append(0.0)
                #if self.av_counts[attr]

                #attributes.add(attr.split(".")[-1])
        corrections = {attr: sum(errors[attr]) / len(errors[attr]) for attr in
                       errors}
        
        for attr in corrections:
            for attr2 in instance:
                if (isinstance(instance[attr2], float) and attr2.split(".")[-1]
                    == attr):
                    instance[attr2] += corrections[attr]

        print(corrections)
        

        # compute error of each attribute.
        #for attr in instance:
        #    if isinstance(instance[attr], float): 


        #print(attributes)

        return instance

    def trestle(self, instance):
        """
        Flattens the representation, performs structure mapping, 
        and then calls cobweb on the result.
        """
        temp_instance = self.flatten(instance)

        temp_instance = self.match(temp_instance)

        #temp_instance = self.align(temp_instance)

        ret = self.cobweb(temp_instance)

        return ret

    #TODO write for flat representation
    def rename(self, instance, mapping):
        """
        Given a mapping (type = dict) rename the 
        components and relations and return the renamed
        instance.
        """
        # Ensure it is a complete mapping
        # Might be troublesome if there is a name collision
        for attr in instance:
            if attr[0] == "_":
                continue
            if isinstance(attr, tuple):
                continue
            for name in attr.split('.')[:-1]:
                if name not in mapping:
                    mapping[name] = name

        temp_instance = {}
        relations = []

        # rename all attribute values
        for attr in instance:
            if attr[0] == "_":
                temp_instance[attr] = instance[attr]
            if isinstance(attr, tuple):
                relations.append(attr)
            elif attr.split('.')[:-1]:
                new_attr = []
                for name in attr.split('.')[:-1]:
                    new_attr.append(mapping[name])
                new_attr.append(attr.split('.')[-1])
                temp_instance[".".join(new_attr)] = instance[attr]
            else:
                temp_instance[attr] = instance[attr]

        #rename relations and add them to instance
        for relation in relations:
            temp = []
            for idx, val in enumerate(relation):
                if idx == 0:
                    temp.append(val)
                else:
                    new_attr = []
                    for name in val.split("."):
                        new_attr.append(mapping[name])
                    temp.append(".".join(new_attr))
            temp_instance[tuple(temp)] = instance[relation]

        #print(instance)
        #print(mapping)
        #print(temp_instance)
        #print(relations)
        #print()

        return temp_instance

    def get_component_names(self, obj):
        """
        Given a flat representation of an instance or concept.av_counts
        return a list of all of the component names.
        """
        names = set()
        for attr in obj:
            if isinstance(attr, tuple):
                continue

            for name in attr.split(".")[:-1]:
                if name[0] != "_":
                    names.add(name)
        return list(names)

    def get_attr_names(self, component_name, obj):
        """
        Given a component name and an object return a list of attributes
        containing that component name.
        """
        #attributes = []
        for attr in obj:
            if isinstance(attr, tuple):
                continue
            
            name = component_name + "."
            if name in attr:
                yield attr
                #attributes.append(attr)
        #return attributes

    def find_replace(self, old, new, attribute):
        """
        Finds the old value in the list and replaces it with the new value in a
        complex attribute name.
        """
        return ".".join([re.sub("^" + old + "$", new, val) for val in
                         attribute.split(".")])

    def hungarian_match(self, instance):
        """
        Compute the greedy match between the instance and the current concept.
        This algorithm is O(n^3) and will return the optimal match if there are
        no relations. However, when there are relations, then it is only
        computing an approximately good match (no longer optimal, not even sure
        if it is the best greedy match possible). 

        Also!!! This current algorithm only matches objects with at most 1 level
        deep structure... 
        """
        from_names = self.get_component_names(instance)
        to_names = self.get_component_names(self.av_counts)

        if(len(from_names) == 0 or
           len(to_names) == 0):
            return {}
        
        length = max(len(from_names), len(to_names))

        # some reasonably large constant when dealing with really small
        # probabilities + bonuses for relations, which mean a given match may
        # be greater than 1.0.
        max_cost = 1000.0
        
        cost_matrix = []
        for row_index in range(length):
            if row_index >= len(from_names):
                cost_matrix.append([max_cost] * length)
                continue 

            row = []
            for col_index in range(length):
                if col_index >= len(to_names):
                    row.append(max_cost)
                    continue

                reward = 0.0

                from_name = from_names[row_index]
                to_name = to_names[col_index]
                
                for from_attr in self.get_attr_names(from_name, instance):
                    to_attr = from_attr.replace(from_name + ".", to_name + ".")
                    value = instance[from_attr]
                    prob = self.get_probability(to_attr, value)
                    reward += prob * prob

                # Additional bonus for part of a relational match
                for i_relation in instance:
                    if not isinstance(i_relation, tuple):
                        continue
                    for c_relation in self.av_counts:
                        if not isinstance(c_relation, tuple):
                            continue
                        if len(i_relation) != len(c_relation):
                            continue
                        if i_relation[0] != c_relation[0]:
                            continue
                        if (instance[i_relation] not in
                            self.av_counts[c_relation]):
                            continue
                        for i in range(1, len(i_relation)):
                            if (from_name in i_relation[i].split('.') and
                                (self.find_replace(from_name, to_name,
                                                   i_relation[i]) ==
                                 c_relation[i])):

                                reward += ((1.0 *
                                            self.av_counts[c_relation][instance[i_relation]]
                                            / self.count) * (1.0 /
                                                             len(c_relation)))
                row.append(max_cost - (reward))
                    
            cost_matrix.append(row)

        # Note: "a" is modified by hungarian.
        a = numpy.array(cost_matrix)
        assignment = hungarianNative.hungarian(a)
        #print(assignment)

        mapping = {}
        for index in assignment:
            if index >= len(from_names):
                continue
            elif assignment[index] >= len(to_names):
                  mapping[from_names[index]] = "component" + self.gensym()
            else:
                mapping[from_names[index]] = to_names[assignment[index]]

        #print(mapping)
        return mapping 

    #TODO rewrite for flat representation
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

    def trestle_categorize(self, instance):
        """
        The Trestle categorize function, this Trestle categorizes all the
        sub-components before categorizing itself.
        """
        temp_instance = self.flatten(instance)

        temp_instance = self.match(temp_instance)

        ret = self.cobweb_categorize(temp_instance)

        return ret

    def ifit(self, instance):
        """
        A modification of ifit to call Trestle instead.
        """
        return self.trestle(instance)

    def concept_attr_value(self, instance, attr, val):
        """
        A modification to call Trestle categorize instead of cobweb
        categorize.
        """
        concept = self.trestle_categorize(instance)
        return concept.get_probability(attr, val)

    def specific_prediction(self, instance, attr, guessing=False):
        """
        Uses the TRESTLE algorithm to make a prediction about the given
        attribute. 
        """
        concept = self.trestle_categorize(instance)
        return concept.get_probability(attr, instance[attr])

    def flexible_prediction(self, instance, guessing=False):
        """
        A modified version of the flexible predictor that removes relations
        that are related to attributes that have been removed. 
        """
        instance = self.flatten(instance)

        attrs = []
        probs = []
        for attr in instance:
            if attr[0] == "_":
                continue
            attrs.append(attr)
            temp = {}
            for attr2 in instance:
                if attr2[0] == "_":
                    continue
                if attr == attr2:
                    continue
                if (isinstance(attr2, list) and attr in attr2.split(".")[:-1]):
                    continue
                temp[attr2] = instance[attr2]

            mapping = self.hungarian_match(temp)

            # Ensure it is a complete mapping
            # Might be troublesome if there is a name collision
            for attr in instance:
                if attr[0] == "_":
                    continue
                if isinstance(attr, tuple):
                    continue
                for name in attr.split('.')[:-1]:
                    if name not in mapping:
                        mapping[name] = name
            
            if isinstance(attr, tuple):
                new_attr = []
                for idx, val in enumerate(attr):
                    if idx == 0:
                        new_attr.append(val)
                    else:
                        new_attr.append(mapping[val])
                new_attr = tuple(new_attr)
            else:
                new_attr = []
                for name in attr.split('.')[:-1]:
                    new_attr.append(mapping[name])
                new_attr.append(attr.split('.')[-1])
                new_attr = ".".join(new_attr)

            temp = self.match(temp)
            #print(attr)
            #print(new_attr)
            if guessing:
                probs.append(self.get_probability(new_attr, instance[attr]))
            else:
                probs.append(self.concept_attr_value(temp, new_attr, instance[attr]))
                #print(new_attr, instance[attr], probs[-1])
        #print(attrs)
        #print(probs)
        return sum(probs) / len(probs)

    #TODO modify to work with flat representation
    #def flexible_prediction(self, instance, guessing=False):
    #    """
    #    A modification of flexible prediction to handle component values.
    #    The flexible prediction task is called on all subcomponents. To compute
    #    the accuracy for each subcomponent.

    #    Guessing is the basecase that just returns the root probability
    #    """
    #    
    #    probs = []
    #    for attr in instance:
    #        #TODO add support for relational attribute values 
    #        if isinstance(instance[attr], list):
    #            continue
    #        if isinstance(instance[attr], dict):
    #            probs.append(self.flexible_prediction(instance[attr], guessing))
    #            continue

    #        # construct an object with missing attribute
    #        temp = {}
    #        for attr2 in instance:
    #            if attr == attr2:
    #                continue
    #            temp[attr2] = instance[attr2]

    #        if guessing:
    #            probs.append(self.get_probability(attr, instance[attr]))
    #        else:
    #            probs.append(self.concept_attr_value(temp, attr, instance[attr]))

    #    if len(probs) == 0:
    #        print(instance)
    #        return -1 
    #    return sum(probs) / len(probs)

    def verify_counts(self):
        """
        Checks the property that the counts of the children sum to the same
        count as the parent. This is/was useful when debugging. If you are
        doing some kind of matching at each step in the categorization (i.e.,
        renaming such as with Labyrinth) then this will start throwing errors.
        """
        if len(self.children) == 0:
            return 

        temp = {}
        temp_count = self.count
        for attr in self.av_counts:
            if isinstance(self.av_counts[attr], ContinuousValue):
                temp[attr] = self.av_counts[attr].num
            else:
                if attr not in temp:
                    temp[attr] = {}
                for val in self.av_counts[attr]:
                    temp[attr][val] = self.av_counts[attr][val]

        for child in self.children:
            temp_count -= child.count
            for attr in child.av_counts:
                assert attr in temp
                if isinstance(child.av_counts[attr], ContinuousValue):
                    temp[attr] -= child.av_counts[attr].num
                else:
                    for val in child.av_counts[attr]:
                        if val not in temp[attr]:
                            print(val.concept_name)
                            print(attr)
                            print(self)
                        assert val in temp[attr]
                        temp[attr][val] -= child.av_counts[attr][val]

        #if temp_count != 0:
        #    print(self.count)
        #    for child in self.children:
        #        print(child.count)
        assert temp_count == 0

        for attr in temp:
            if isinstance(temp[attr], int):
                assert temp[attr] == 0.0
            else:
                for val in temp[attr]:
                    #if temp[attr][val] != 0.0:
                    #    print(self)

                    assert temp[attr][val] == 0.0

        for child in self.children:
            child.verify_counts()

    # TODO modify to work with a flat representation
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

        concept = self.trestle_categorize_leaf(prediction)
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

            float_num = 0.0
            float_mean = 0.0
            float_std = 0.0
            
            if isinstance(concept.av_counts[attr], ContinuousValue):
                float_num = concept.av_counts[attr].num
                float_mean = concept.av_counts[attr].mean
                float_std = concept.av_counts[attr].std

            else:
                for val in concept.av_counts[attr]:
                    if isinstance(val, Trestle):
                        component_values += [val] * concept.av_counts[attr][val] 
                    else:
                        nominal_values += [val] * concept.av_counts[attr][val]

            rand = random()

            if rand < ((len(nominal_values) * 1.0) / (len(nominal_values) +
                                                      len(component_values) +
                                                      float_num)):
                prediction[attr] = choice(nominal_values)
            elif rand < ((len(nominal_values) + len(component_values) * 1.0) /
                         (len(nominal_values) + len(component_values) +
                          float_num)):
                prediction[attr] = choice(component_values).predict({})
            else:
                prediction[attr] = normalvariate(float_mean,
                                                 float_std)

        return prediction

    def flatten_instance(self, instance):
        duplicate = copy.deepcopy(instance)
        for attr in duplicate:
            if isinstance(duplicate[attr], dict):
                duplicate[attr] = self.flatten_instance(duplicate[attr])
        return repr(sorted(duplicate.items()))

    #TODO test this out more...
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
                        L.append(l.pop(0))
                        
                lists = [l for l in lists if l]
            return L

    def verify_parent_pointers(self):
        for c in self.children:
            assert c.parent == self
            c.verify_parent_pointers()     

    def get_root(self):
        """
        Gets the root of the categorization tree.
        """
        if self.parent == None:
            return self
        else:
            return self.parent.get_root()

    def cluster(self, instances, depth=1):
        """
        Used to cluster examples incrementally and return the cluster labels.
        The final cluster labels are at a depth of 'depth' from the root. This
        defaults to 1, which takes the first split, but it might need to be 2
        or greater in cases where more distinction is needed.
        """

        #print(len(instances))
        temp_clusters = []
        for idx, instance in enumerate(instances):
            #print("Categorizing: ", idx)
            temp_clusters.append(self.ifit(instance))
        #temp_clusters = [self.ifit(instance) for instance in instances]

        clusters = []
        for i,c in enumerate(temp_clusters):
            #print("Labeling: ", i)
            temp = c

            while (temp.parent and temp not in temp.parent.children):
                temp = temp.parent

            if temp.depth() < depth:
                #print("recategorizing: ", i)
                temp = self.trestle_categorize(instances[i])

            while temp.depth() > depth and temp.parent:
                temp = temp.parent

            clusters.append(temp.concept_name)

        with open('visualize/output.json', 'w') as f:
            f.write(json.dumps(self.output_json()))

        return clusters

    def kc_label(self, filename, length):
        """
        Used to provide a clustering of a set of examples provided in a JSON
        file. It starts by incorporating the examples into the categorization
        tree multiple times. After incorporating the instances it then
        categorizes each example (without updating the tree) and returns the
        concept it was assoicated with.
        """
        json_data = open(filename, "r")
        instances = json.load(json_data)
        print("%i Instances." % len(instances))
        shuffle(instances)
        instances = instances[0:length]
        o_instances = copy.deepcopy(instances)
        for instance in instances:
            #if "success" in instance:
            #    del instance['success']
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

        # categorize to add guids
        mapping = {}
        for idx, inst in enumerate(o_instances):
            print("categorizing instance: %i" % idx)

            if inst['guid'] in mapping:
                # ignore duplicate states.
                print("skipping duplicate guid")
                continue

            instance = copy.deepcopy(inst)
            
            #print(instance)

            # we want the KCS for only the correct productions.
            instance['Outcome'] = 'CORRECT'
            # for now just make the action correct, but leave the action.. so
            # closest action that is correct.
            #del instance['action']
            #if 'destination' in instance:
            #    del instance['destination']

            #del instance['r1']
            #del instance['r2']

            #if "success" in instance:
            #    del instance['success']
            if "guid" in instance:
                del instance['guid']
            g_instances[inst['guid']] = instance

            #print(instance)
            
            #print()
            mapping[inst['guid']] = self.trestle_categorize(instance)
            #print(mapping[inst['guid']].concept_name)

        # add guids
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
            if cluster.parent:
                cluster = cluster.parent
            clusters[g] = cluster.concept_name

        with open('visualize/output.json', 'w') as f:
            f.write(json.dumps(self.output_json()))

        # Output data for datashop KC labeling
        guidKcs = []
        for g in mapping:
            kcs = []
            temp = mapping[g]
            kcs.append(temp.concept_name)
            while temp.parent:
                temp = temp.parent
                kcs.append(temp.concept_name)
            kcs.append(g) 
            kcs.reverse()
            guidKcs.append(kcs)

        with open('kc-labels.csv', 'w') as f:
            max_len = 0
            for kc in guidKcs:
                if len(kc) > max_len:
                    max_len = len(kc)

            output = []
            for kc in guidKcs:
                for i in range(max_len - len(kc)):
                    kc.append(kc[-1])
                output.append(",".join(kc))

            f.write("\n".join(output))

        #print(json.dumps(self.output_json()))

        return clusters

    def save(self):
        pickle.dump(self, open('trestle.pickle', 'wb'))

    def load():
        return pickle.load(open('trestle.pickle', 'rb'))

def random_data():
    from random import uniform
    tree = Trestle()

    instances = [] 
    for i in range(100):
        instance = {}
        instance['v1'] = choice(['a','b','c','d','e','f','g'])
        instance['v2'] = uniform(0, 100)
        instance['v3'] = uniform(0, 100)
        instance['v4'] = uniform(0, 100)

        subinstance = {}
        subinstance['v1'] = uniform(0,100)
        subinstance['v2'] = uniform(0,100)
        subinstance['v3'] = uniform(0,100)
        subinstance['v4'] = uniform(0,100)

        instance['component1'] = subinstance

        instances.append(instance) 
    tree.predictions(instances, 30, 5, False)
    tree.predictions(instances, 30, 5, True)
    tree.cluster(instances)
    

def noise_experiments():
    from random import uniform
    from sklearn import metrics
    import math
    
    samples = 10
    iterations = 10

    print("continuous noise\tnominal noise\taccuracy\tstd")
    for cm in range(11):
        continuous_noise = cm/10.0
        for nm in range(11):
            nominal_noise = nm/10.0
            ari = []
            for xxx in range(iterations):
                tree = Trestle()

                #Clustering
                data1 = [{'key': 'c1', 'cf1': normalvariate(1,0.001), 'nf1':
                          'one'} for i in range(samples)]
                data2 = [{'key': 'c2', 'cf1': normalvariate(2,0.001), 'nf1':
                          'two'} for i in range(samples)]
                data3 = [{'key': 'c3', 'cf1': normalvariate(3,0.001), 'nf1':
                          'three'} for i in range(samples)]
                data4 = [{'key': 'c4', 'cf1': normalvariate(4,0.001), 'nf1':
                          'four'} for i in range(samples)]

                data = data1 + data2 + data3 + data4
                shuffle(data)

                labels_true = [d[a] for d in data for a in d if a == 'key']
                data = [{a: d[a] for a in d} for d in data]
                
                noisy = []
                for d in data:
                    temp = d

                    nv = set()
                    nv.add('one')
                    nv.add('two')
                    nv.add('three')
                    nv.add('four')

                    cv = set()
                    cv.add(1)
                    cv.add(2)
                    cv.add(3)
                    cv.add(4)

                    # nominal noise
                    if random() < nominal_noise:
                        temp['nf1'] = choice(list(nv))

                    # continuous value noise
                    if random() < continuous_noise:
                        temp['cf1'] = uniform(0,1000)
                        #temp['cf1'] = normalvariate(choice(list(cv - s)),0.001)
                              
                    noisy.append(temp)

                clusters = tree.cluster(noisy)

                ari.append(metrics.adjusted_rand_score(labels_true, clusters))

            mean = sum(ari)/(1.0 * len(ari))
            std = math.sqrt((float(sum([(v - mean) * (v - mean) for v in ari])) /
                             (len(ari) - 1.0)))
            print("%0.4f\t%0.4f\t%0.4f\t%0.4f" % (continuous_noise, nominal_noise, mean, std))


def cluster_data(json_file):
    tree = Trestle()
    with open(json_file, "r") as json_data:
        instances = json.load(json_data)

    #remove the relations
    new_instances = []
    for instance in instances:
        new_instance = {}
        for attr in instance:
            #if attr == "success":
            #    continue
            if not isinstance(instance[attr], list):
                new_instance[attr] = instance[attr]
        new_instances.append(new_instance)

    print(set(tree.cluster(new_instances)))


if __name__ == "__main__":

    # KC labeling
    #random_data()
    #noise_experiments()
    #cluster_data('data_files/rb_s_07_continuous.json')
    cluster_data('data_files/rb_com_11_noCheck.json')

    #with open('data_files/rb_s_07_continuous.json', "r") as json_data:
    #    instances = json.load(json_data)

    ##remove the relations
    #new_instances = []
    #for instance in instances:
    #    new_instance = {}
    #    for attr in instance:
    #        if attr == "success":
    #            continue
    #        if not isinstance(instance[attr], list):
    #            new_instance[attr] = instance[attr]
    #    new_instances.append(new_instance)

    #print(set(tree.cluster(new_instances)))
    #tree.predictions(new_instances, 30, 5, False)

    #labels = tree.kc_label("data_files/instant-test-processed.json", 16000)
    #pickle.dump(labels, open('clustering.pickle', 'wb'))



