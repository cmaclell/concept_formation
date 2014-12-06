import re
import json
import utils
from random import choice
from random import shuffle

class CobwebTree:

    def __init__(self):
        """
        Initialize the tree with a CobwebNode
        """
        self.root = CobwebNode()

    def __str__(self):
        return str(self.root)

    def ifit(self, instance):
        """
        Given an instance incrementally update the categorization tree.
        """
        return self.cobweb(instance)

    def fit(self, list_of_instances):
        """
        Call incremental fit on each element in a list of instances.
        """
        # TODO rewrite this to get the optimal fit by continually reclustering
        # until no change.
        for i, instance in enumerate(list_of_instances):
            #print("instance %i of %i" % (i, len(list_of_instances)))
            self.ifit(instance)

    def cobweb(self, instance):
        """
        Incrementally integrates an instance into the categorization tree.
        This function operates iteratively to integrate this instance and uses
        category utility as the heuristic to make decisions.
        """
        current = self.root

        while current:
            if (not current.children and current.cu_for_fringe_split(instance)
                <= 0.0):
                current.increment_counts(instance)
                return current 

            elif not current.children:
                new = current.__class__(current)
                current.parent = new
                new.children.append(current)

                if new.parent:
                    new.parent.children.remove(current)
                    new.parent.children.append(new)
                else:
                    self.root = new

                new.increment_counts(instance)
                return new.create_new_child(instance)
            else:
                best1, best2 = current.two_best_children(instance)
                action_cu, best_action = current.get_best_operation(instance,
                                                                    best1,
                                                                    best2)

                if best1:
                    best1_cu, best1 = best1
                if best2:
                    best2_cu, best2 = best2

                if best_action == 'best':
                    current.increment_counts(instance)
                    current = best1
                elif best_action == 'new':
                    current.increment_counts(instance)
                    return current.create_new_child(instance)
                elif best_action == 'merge':
                    current.increment_counts(instance)
                    new_child = current.merge(best1, best2)
                    current = new_child
                elif best_action == 'split':
                    current.split(best1)
                else:
                    raise Exception("Should never get here.")

    def cobweb_categorize_leaf(self, instance):
        """
        Sorts an instance in the categorization tree defined at the current
        node without modifying the counts of the tree.

        This version always goes to a leaf.
        """
        current = self.root
        while current:
            if not current.children:
                return current
            
            best1, best2 = current.two_best_children(instance)

            if best1:
                best1_cu, best1 = best1
                current = best1
            else:
                return current

    def cobweb_categorize(self, instance):
        """
        Sorts an instance in the categorization tree defined at the current
        node without modifying the counts of the tree.

        Uses the new and best operations; when new is the best operation it
        returns the current node otherwise it iterates on the best node. 
        """
        current = self.root
        while current:
            if not current.children:
                return current

            best1, best2 = current.two_best_children(instance)
            action_cu, best_action = current.get_best_operation(instance,
                                                                 best1, best2,
                                                                 ["best",
                                                                  "new"]) 
            if best1:
                best1_cu, best1 = best1
            else:
                return current

            if best_action == "new":
                return current
            elif best_action == "best":
                current = best1

    def predict_attribute(self, instance, attribute):
        if attribute in instance:
            raise ValueError("The attribute is present in the instance!")
        concept = self.cobweb_categorize(instance)
        current = concept

        while attribute not in concept.av_counts and current.parent:
            current = current.parent
        
        value, count = max(current.av_counts[attribute].items(), key=lambda x:
                           x[1])
        return value

    def predict(self, instance):
        """
        Given an instance predict any missing attribute values without
        modifying the tree.
        """
        prediction = {}

        # make a copy of the instance
        for attr in instance:
            prediction[attr] = instance[attr]

        concept = self.cobweb_categorize(instance)
        
        for attr in concept.av_counts:
            if attr in prediction:
                continue
            
            values = []
            for val in concept.av_counts[attr]:
                values += [val] * concept.av_counts[attr][val]

            prediction[attr] = choice(values)

        return prediction

    def concept_attr_value(self, instance, attr, val):
        """
        Gets the probability of a particular attribute value for the concept
        associated with a given instance.
        """
        concept = self.cobweb_categorize(instance)
        return concept.get_probability(attr, val)

    def flexible_prediction(self, instance, guessing=False):
        """
        Fisher's flexible prediction task. It computes the accuracy of
        correctly predicting each attribute value (removing it from the
        instance first). It then returns the average accuracy. 
        """
        probs = []
        for attr in instance:
            temp = {}
            for attr2 in instance:
                if attr == attr2:
                    continue
                temp[attr2] = instance[attr2]
            if guessing:
                probs.append(self.get_probability(attr, instance[attr]))
            else:
                probs.append(self.concept_attr_value(temp, attr, instance[attr]))
        return sum(probs) / len(probs)

    def specific_prediction(self, instance, attr, guessing=False):
        """
        Similar to flexible prediction, but for a single attribute.
        """
        temp = {}
        for attr2 in instance:
            if attr == attr2:
                continue
            temp[attr2] = instance[attr2]
        if guessing:
            prob = self.get_probability(attr, instance[attr])
        else:
            prob = self.concept_attr_value(temp, attr, instance[attr])
        return prob

    def train_from_json(self, filename, length=None):
        """
        Build the concept tree from a set of examples in a provided json file.
        """
        json_data = open(filename, "r")
        instances = json.load(json_data)
        if length:
            shuffle(instances)
            instances = instances[:length]
        self.fit(instances)
        json_data.close()

    def sequential_prediction(self, filename, length, guessing=False, attr=None):
        """
        Given a json file, perform an incremental sequential prediction task. 
        Try to flexibly predict each instance before incorporating it into the 
        tree. This will give a type of cross validated result.
        """
        json_data = open(filename, "r")
        instances = json.load(json_data)
        #shuffle(instances)
        #instances = instances[0:length]

        accuracy = []
        nodes = []
        for j in range(1):
            shuffle(instances)
            for n, i in enumerate(instances):
                if n >= length:
                    break
                if not attr:
                    accuracy.append(self.flexible_prediction(i, guessing))
                else:
                    accuracy.append(self.specific_prediction(i, attr, guessing))
                nodes.append(self.root.num_concepts())
                self.ifit(i)
        json_data.close()
        return accuracy, nodes

    def cluster_from_json(self, filename, ordered=True, depth=1, length=-1):
        """
        Cluster a set of examples in a provided json file
        """
        json_data = open(filename,"r")
        instances = json.load(json_data)
        if not ordered:
            shuffle(instances)
        if length > 0 and length < len(instances) :
            instances = instances[:length]
        clustering = self.cluster(instances,depth)
        json_data.close()
        return zip(instances,clustering)


    def kc_model_from_datashop(self, filename, stateField='Input', depth=1):
        """
        Reads in a datashop export file and generates a KC model based on the concept tree
        """
        datashop = csv.reader(open(filename,'r'),delimiter='\t')
        h = None
        instances = []
        mapping = {}

        for row in datashop :
            if not h:
                h = {}
                for i,v in enumerate(row):
                    h[v] = i
                continue

            instance = json.load(row[h[stateField]])
            concept = self.ifit(instance)
            mapping[row[h['Transaction Id']]] = concept

        datashop.close()

        #categorize with state action pairs
        #drop the actions 
        #group all of them by the same state
        #clusters of the ones that were successful
        #those are the positive actions that belong in that cluster
        
        # ensure that the final concept is actually in the tree and not a merged/split node
        for id in mapping :
            while mapping[id] not in mapping[id].parent.children:
                mapping[id] = parent
        


        #wanted the correct actions on each problem

        #{
        #   final state...    
        #   step1:{
        #       sel:""
        #       act:""
        #       inp:""
        #   }
        #   step2:{...}
        #   (ordered step1 step2):T
        #}
        #
        #

    def generate_d3_visualization(self):
        """
        Generates the .js file that is used by index.html to generate the d3 tree.
        """
        #with open('visualize/output.json', 'w') as f:
        #    f.write(json.dumps(self.root.output_json()))
        with open('visualize/output.js', 'w') as f:
            f.write("var output = '"+re.sub("'", '',
                                            json.dumps(self.root.output_json()))+"';")

    def cluster(self, instances, minsplit=1, maxsplit=1):
        """
        Returns a list of clusterings. the size of the list will be (1 +
        maxsplit - minsplit). The first clustering will be if the root was
        split, then each subsequent clustering will be if the least coupled
        cluster (in terms of category utility) is split. This might stop early
        if you reach the case where there are no more clusters to split (each
        instance in its own cluster). 
        """
        temp_clusters = [self.ifit(instance) for instance in instances]

        clusterings = []
        for nth_split in range(minsplit, maxsplit+1):
            #print(len(set([c.concept_id for c in temp_clusters])))

            clusters = []
            for i,c in enumerate(temp_clusters):
                while (c.parent and c.parent.parent):
                    c = c.parent
                clusters.append("Concept" + c.concept_id)
            clusterings.append(clusters)

            split_cus = sorted([(self.root.cu_for_split(c) -
                                 self.root.category_utility(), i, c) for i,c in
                                enumerate(self.root.children) if c.children])

            # Exit early, we don't need to re-reun the following part for the
            # last time through
            if nth_split == maxsplit or not split_cus:
                break

            #print(self.root.category_utility())
            #print(split_cus)
            self.root.split(split_cus[-1][2])

        self.generate_d3_visualization()

        return clusterings

    def baseline_guesser(self, filename, length, iterations):
        """
        Equivalent of predictions, but just makes predictions from the root of
        the concept tree. This is the equivalent of guessing the distribution
        of all attribute values. 
        """
        n = iterations
        runs = []
        nodes = []

        for i in range(0,n):
            print("run %i" % i)
            t = self.__class__()
            accuracy, num = t.sequential_prediction(filename, length, True)
            runs.append(accuracy)
            nodes.append(num)
            #print(json.dumps(t.output_json()))

        #print(runs)
        print("MEAN Accuracy")
        for i in range(0,len(runs[0])):
            a = []
            for r in runs:
                a.append(r[i])
            print("%0.2f" % (utils.mean(a)))

        print()
        print("STD Accuracy")
        for i in range(0,len(runs[0])):
            a = []
            for r in runs:
                a.append(r[i])
            print("%0.2f" % (utils.std(a)))

        print()
        print("MEAN Concepts")
        for i in range(0,len(runs[0])):
            a = []
            for r in nodes:
                a.append(r[i])
            print("%0.2f" % (utils.mean(a)))

        print()
        print("STD Concepts")
        for i in range(0,len(runs[0])):
            a = []
            for r in nodes:
                a.append(r[i])
            print("%0.2f" % (utils.std(a)))

    def predictions(self, filename, length, iterations, attr=None):
        """
        Perform the sequential prediction task many times and compute the mean
        and std of all flexible predictions.
        """
        n = iterations 
        runs = []
        nodes = []
        for i in range(0,n):
            print("run %i" % i)
            t = self.__class__()
            accuracy, num = t.sequential_prediction(filename, length, attr=attr)
            runs.append(accuracy)
            nodes.append(num)
            #print(json.dumps(t.output_json()))

        #print(runs)
        print("MEAN Accuracy")
        for i in range(0,len(runs[0])):
            a = []
            for r in runs:
                a.append(r[i])
            print("%0.2f" % (utils.mean(a)))

        print()
        print("STD Accuracy")
        for i in range(0,len(runs[0])):
            a = []
            for r in runs:
                a.append(r[i])
            print("%0.2f" % (utils.std(a)))

        print()
        print("MEAN Concepts")
        for i in range(0,len(runs[0])):
            a = []
            for r in nodes:
                a.append(r[i])
            print("%0.2f" % (utils.mean(a)))

        print()
        print("STD Concepts")
        for i in range(0,len(runs[0])):
            a = []
            for r in nodes:
                a.append(r[i])
            print("%0.2f" % (utils.std(a)))

class CobwebNode:

    counter = 0

    def __init__(self, otherNode=None):
        """
        The constructor creates a cobweb node with default values. It can also
        be used as a copy constructor to "deepcopy" a node.
        """
        self.concept_id = self.gensym()
        #self.concept_name = "Concept" + self.gensym()
        self.cached_guess_count = None
        self.count = 0
        self.av_counts = {}
        self.children = []
        self.parent = None

        # check if the constructor is being used as a copy constructor
        if otherNode:
            self.update_counts_from_node(otherNode)
            self.parent = otherNode.parent
            self.cached_guess_count = otherNode.expected_correct_guesses()

            for child in otherNode.children:
                self.children.append(self.__class__(child))

    def shallow_copy(self):
        """
        Creates a copy of the current node and its children (but not their
        children)
        """
        temp = self.__class__()
        temp.update_counts_from_node(self)
        temp.cached_guess_count = self.expected_correct_guesses()

        #for child in self.children:
        #    temp_child = self.__class__()
        #    temp_child.update_counts_from_node(child)
        #    temp.children.append(temp_child)

        return temp

    def increment_counts(self, instance):
        """
        Increment the counts at the current node according to the specified
        instance.

        input:
            instance: {a1: v1, a2: v2, ...} - a hashtable of attr and values. 
        """
        self.cached_guess_count = None
        self.count += 1 
        for attr in instance:
            self.av_counts[attr] = self.av_counts.setdefault(attr,{})
            self.av_counts[attr][instance[attr]] = (self.av_counts[attr].get(
                instance[attr], 0) + 1)
    
    def update_counts_from_node(self, node):
        """
        Increments the counts of the current node by the amount in the specified
        node.
        """
        self.cached_guess_count = None
        self.count += node.count
        for attr in node.av_counts:
            for val in node.av_counts[attr]:
                self.av_counts[attr] = self.av_counts.setdefault(attr,{})
                self.av_counts[attr][val] = (self.av_counts[attr].get(val,0) +
                                     node.av_counts[attr][val])

    def attr_val_guess_gain(self, attr, val):
        """
        Returns the gain in number of correct guesses if a particular attr/val
        was added to a concept.
        """
        if attr[0] == "_":
            return 0.0
        if attr not in self.av_counts:
            return 0.0
        if val not in self.av_counts[attr]:
            return 0.0

        before_prob = (self.av_counts[attr][val] / (self.count + 1.0))
        after_prob = (self.av_counts[attr][val] + 1) / (self.count + 1.0)

        return (after_prob * after_prob) - (before_prob * before_prob)

    def expected_correct_guesses(self):
        """
        Returns the number of correct guesses that are expected from the given
        concept. This is the sum of the probability of each attribute value
        squared. 
        """
        if self.cached_guess_count:
            return self.cached_guess_count

        correct_guesses = 0.0

        for attr in self.av_counts:
            if attr[0] == "_":
                continue
            for val in self.av_counts[attr]:
                prob = (self.av_counts[attr][val] / (1.0 * self.count))
                correct_guesses += (prob * prob)

        #if self.cached_guess_count:
        #    assert self.cached_guess_count == correct_guesses

        self.cached_guess_count = correct_guesses
        return correct_guesses

    def category_utility(self):
        """
        Returns the category utility of a particular division of a concept into
        its children. This is used as the heuristic to guide the concept
        formation.
        """
        if len(self.children) == 0:
            return 0.0

        child_correct_guesses = 0.0

        for child in self.children:
            p_of_child = child.count / (1.0 * self.count)
            child_correct_guesses += p_of_child * child.expected_correct_guesses()

        return ((child_correct_guesses - self.expected_correct_guesses()) /
                (1.0 * len(self.children)))

    def get_best_operation(self, instance, best1, best2, 
                            possible_ops=["best", "new", "merge", "split"]):
        """
        Given a set of possible operations, find the best and return its cu and
        the action name.
        """
        # If there is no best, then create a new child.
        if not best1:
            raise ValueError("Need at least one best child.")
            #return (self.cu_for_new_child(instance), 'new')

        if best1:
            best1_cu, best1 = best1
        if best2:
            best2_cu, best2 = best2
        operations = []

        if "best" in possible_ops:
            operations.append((best1_cu, 1, "best"))
        if "new" in possible_ops: 
            operations.append((self.cu_for_new_child(instance), 3, 'new'))
        if "merge" in possible_ops and len(self.children) > 2 and best2:
            operations.append((self.cu_for_merge(best1, best2, instance), 2,'merge'))
        if "split" in possible_ops and len(best1.children) > 0:
            operations.append((self.cu_for_split(best1),4, 'split'))

        operations.sort(reverse=True)
        #print(operations)
        best_op = (operations[0][0], operations[0][2])
        #print(best_op)
        return best_op

    def two_best_children(self, instance):
        """
        Returns the two best children to incorporate the instance
        into in terms of category utility.

        input:
            instance: {a1: v1, a2: v2,...} - a hashtable of attr. and values. 
        output:
            (0.2,2),(0.1,3) - the category utility and indices for the two best
            children (the second tuple will be None if there is only 1 child).
        """
        if len(self.children) == 0:
            raise Exception("No children!")

        # Sort childrent by CU, then by Size, then by Recency. 
        children_cu = [(self.cu_for_insert(child, instance), child.count, i, child) for i,
                       child in enumerate(self.children)]
        children_cu.sort(reverse=True)

        #if children_cu[0][0] == children_cu[1][0]:
        #    print(children_cu)

        if len(children_cu) == 0:
            return None, None
        if len(children_cu) == 1:
            return (children_cu[0][0], children_cu[0][3]), None 

        return ((children_cu[0][0], children_cu[0][3]), (children_cu[1][0],
                                                         children_cu[1][3]))

    def cu_for_insert(self, child, instance):
        """
        Computer the category utility of adding the instance to the specified
        child.
        """
        temp = self.shallow_copy()
        temp.increment_counts(instance)

        for c in self.children:
            temp_child = c.shallow_copy()
            temp.children.append(temp_child)
            if c == child:
                temp_child.increment_counts(instance)
        return temp.category_utility()

    def create_new_child(self, instance):
        """
        Creates a new child (to the current node) with the counts initialized by
        the given instance. 
        """
        new_child = self.__class__()
        new_child.parent = self
        new_child.increment_counts(instance)
        self.children.append(new_child)
        return new_child

    def create_child_with_current_counts(self):
        """
        Creates a new child (to the current node) with the counts initialized by
        the current node's counts.
        """
        if self.count > 0:
            new = self.__class__(self)
            new.parent = self
            self.children.append(new)
            return new

    def cu_for_new_child(self, instance):
        """
        Returns the category utility for creating a new child using the
        particular instance.
        """
        temp = self.shallow_copy()
        for c in self.children:
            temp.children.append(c.shallow_copy())

        #temp = self.shallow_copy()
        
        temp.increment_counts(instance)
        temp.create_new_child(instance)
        return temp.category_utility()

    def merge(self, best1, best2):
        """
        Merge the two specified nodes.

        input:
            best1: the best child
            best2: the second best child
        output:
            The new child formed from the merge
        """
        new_child = self.__class__()
        new_child.parent = self
        new_child.update_counts_from_node(best1)
        new_child.update_counts_from_node(best2)
        best1.parent = new_child
        best2.parent = new_child
        new_child.children.append(best1)
        new_child.children.append(best2)
        self.children.remove(best1)
        self.children.remove(best2)
        self.children.append(new_child)

        return new_child

    def cu_for_merge(self, best1, best2, instance):
        """
        Returns the category utility for merging the two best children.

        input:
            best1: the best child in the children array.
            best2: the second best child in the children array.
        output:
            0.02 - the category utility for the merge of best1 and best2.
        """
        temp = self.shallow_copy()
        temp.increment_counts(instance)

        new_child = self.__class__()
        new_child.update_counts_from_node(best1)
        new_child.update_counts_from_node(best2)
        new_child.increment_counts(instance)
        temp.children.append(new_child)

        for c in self.children:
            if c == best1 or c == best2:
                continue
            temp_child = c.shallow_copy()
            temp.children.append(temp_child)

        return temp.category_utility()

    def split(self, best):
        """
        Split the best node and promote its children
        """
        self.children.remove(best)
        for child in best.children:
            child.parent = self
            self.children.append(child)

    def cu_for_fringe_split(self, instance):
        """
        Determine the category utility of performing a fringe split (i.e.,
        adding a leaf to a leaf). It turns out that this is useful for
        identifying unnecessary fringe splits, when the two leaves are
        essentially identical. It can be used to keep the tree from growing and
        to increase the tree's predictive accuracy.
        """
        temp = self.shallow_copy()
        
        temp.create_child_with_current_counts()
        temp.increment_counts(instance)
        temp.create_new_child(instance)

        return temp.category_utility()

    def cu_for_split(self, best):
        """
        Return the category utility for splitting the best child.
        
        input:
            best1: a child in the children array.
        output:
            0.03 - the category utility for the split of best1.
        """
        temp = self.shallow_copy()

        for c in self.children + best.children:
            if c == best:
                continue
            temp_child = c.shallow_copy()
            temp.children.append(temp_child)

        return temp.category_utility()

    def __hash__(self):
        """
        The basic hash function. This hashes the concept name, which is
        generated to be unique across concepts.
        """
        return hash("CobwebNode" + str(self.concept_id))

    def gensym(self):
        """
        Generates a unique id and increments the class counter. This is used to
        create a unique name for every concept. 
        """
        self.__class__.counter += 1
        return str(self.__class__.counter)

    def __str__(self):
        """
        Call pretty print
        """
        return self.pretty_print()

    def pretty_print(self, depth=0):
        """
        Prints the categorization tree.
        """
        ret = str(('\t' * depth) + "|-" + str(self.av_counts) + ":" +
                  str(self.count) + '\n')
        
        for c in self.children:
            ret += c.pretty_print(depth+1)

        return ret

    def depth(self):
        """
        Returns the depth of the current node in the tree.
        """
        if self.parent:
            return 1 + self.parent.depth()
        return 0

    def is_parent(self, other_concept):
        """
        Returns True if self is a parent of other concept.
        """
        temp = other_concept
        while temp != None:
            if temp == self:
                return True
            try:
                temp = temp.parent
            except:
                print(temp)
                assert False
        return False

    def num_concepts(self):
        """
        Return the number of concepts contained in the tree defined by the
        current node. 
        """
        children_count = 0
        for c in self.children:
           children_count += c.num_concepts() 
        return 1 + children_count 

    def output_json(self):
        """
        Outputs the categorization tree in JSON form so that it can be
        displayed, I usually visualize it with d3js in a web browser.
        """
        output = {}
        output['name'] = "Concept" + self.concept_id
        output['size'] = self.count
        output['children'] = []

        temp = {}
        for attr in self.av_counts:
            for value in self.av_counts[attr]:
                temp[attr + " = " + str(value)] = self.av_counts[attr][value]

        for child in self.children:
            output['children'].append(child.output_json())

        output['counts'] = temp

        return output

    def get_probability(self, attr, val):
        """
        Gets the probability of a particular attribute value at the given
        concept.

        """
        current = self
        while attr not in current.av_counts:
            if not current.parent:
                return 0.0
            current = current.parent

        if val not in current.av_counts[attr]:
            return 0.0

        return (1.0 * current.av_counts[attr][val]) / current.count

if __name__ == "__main__":
    #Cobweb().predictions("data_files/cobweb_test.json", 10, 100)
    #Cobweb().predictions("data_files/mushrooms.json", 30, 10)
    #Cobweb().baseline_guesser("data_files/cobweb_test.json", 10, 100)
    #print(Cobweb().cluster("cobweb_test.json", 10, 1))

    #t = Cobweb()
    #print(t.sequential_prediction("cobweb_test.json", 10))
    #t.verify_counts()

    #test = {}
    #print(t.predict(test))

    tree = CobwebTree()
    tree.predictions("data_files/mushrooms.json", 30, 10, attr="0")
    #tree.predictions("data_files/mushrooms.json", 30, 10)

    #tree.ifit({'a': 'v', 'b': 'v'})
    #tree.ifit({'a': 'v2'})
    #tree.ifit({'a': 'v'})
    #tree.ifit({'a': 'v'})
    #tree.ifit({'a': 'v'})
    #tree.ifit({'a': 'v'})
    #print(tree.cobweb_categorize({'a':'v'}))
    #print(tree)

    #print(tree.predict_attribute({'a': 'v'}, 'b'))

