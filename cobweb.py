import json
from random import shuffle
from random import random

class CobwebTree:

    def __init__(self):
        """
        Initialize the tree with a CobwebNode
        """
        self.root = CobwebNode()
        self.root.root = self.root

    def __str__(self):
        return str(self.root)

    def ifit(self, instance):
        """
        Given an instance incrementally update the categorization tree.
        """
        return self.cobweb(instance)

    def fit(self, instances, iterations=1, shuffle=True):
        """
        This is a batch ifit function that takes a collection of instances
        and categorizes all of them. This function does not return anything.

        instances -- a collection of instances
        iterations -- the number of iterations to perform
        shuffle -- whether or not to shuffle the list between iterations.

        Note that the first iteration is not shuffled.
        """
        instances = [i for i in instances]
        for x in range(iterations):
            for i in instances:
                self.ifit(i)
            if shuffle:
                shuffle(instances)

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
                    self.root.root = new

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

    def categorize(self, instance):
        """
        A categorize function that can be used polymorphicaly without 
        having to worry about the type of the underlying object.

        In Cobweb's case this calls cobweb_categorize()
        """
        return self.cobweb_categorize(instance)
    
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

class CobwebNode:

    counter = 0

    def __init__(self, otherNode=None):
        """
        The constructor creates a cobweb node with default values. It can also
        be used as a copy constructor to "deepcopy" a node.
        """
        self.concept_id = self.gensym()
        self.count = 0
        self.av_counts = {}
        self.children = []
        self.parent = None
        self.root = None

        if otherNode:
            self.update_counts_from_node(otherNode)
            self.parent = otherNode.parent
            self.root = otherNode.root

            for child in otherNode.children:
                self.children.append(self.__class__(child))

    def shallow_copy(self):
        """
        Creates a shallow copy of the current node (and not its children)
        """
        temp = self.__class__()
        temp.root = self.root
        temp.update_counts_from_node(self)
        return temp

    def increment_counts(self, instance):
        """
        Increment the counts at the current node according to the specified
        instance.

        input:
            instance: {a1: v1, a2: v2, ...} - a hashtable of attr and values. 
        """
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
        self.count += node.count
        for attr in node.av_counts:
            for val in node.av_counts[attr]:
                self.av_counts[attr] = self.av_counts.setdefault(attr,{})
                self.av_counts[attr][val] = (self.av_counts[attr].get(val,0) +
                                     node.av_counts[attr][val])

    def expected_correct_guesses(self, alpha=0.001):
        """
        Returns the number of correct guesses that are expected from the given
        concept. This is the sum of the probability of each attribute value
        squared. 
        """
        correct_guesses = 0.0

        for attr in self.root.av_counts:
            if attr[0] == "_":
                continue
            val_count = 0

            # the +1 is for the "missing" value
            n_values = len(self.root.av_counts[attr]) + 1

            for val in self.root.av_counts[attr]:
                if attr not in self.av_counts or val not in self.av_counts[attr]:
                    prob = 0
                    if alpha > 0:
                        prob = alpha / (alpha * n_values)
                else:
                    val_count += self.av_counts[attr][val]
                    prob = ((self.av_counts[attr][val] + alpha) / (1.0 * self.count
                                                              + alpha * n_values))
                correct_guesses += (prob * prob)

            #Factors in the probability mass of missing values
            prob = ((self.count - val_count + alpha) / (1.0*self.count + alpha * n_values))
            correct_guesses += (prob * prob)

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
        Given a set of possible operations, find the operator that produces the
        highest category utility, and then returns the cu and the action name
        for the best operation. In the case of ties, an operator is randomly
        chosen.
        """
        if not best1:
            raise ValueError("Need at least one best child.")

        if best1:
            best1_cu, best1 = best1
        if best2:
            best2_cu, best2 = best2
        operations = []

        if "best" in possible_ops:
            operations.append((best1_cu, random(), "best"))
        if "new" in possible_ops: 
            operations.append((self.cu_for_new_child(instance), random(), 'new'))
        if "merge" in possible_ops and len(self.children) > 2 and best2:
            operations.append((self.cu_for_merge(best1, best2, instance),
                               random(),'merge'))
        if "split" in possible_ops and len(best1.children) > 0:
            operations.append((self.cu_for_split(best1), random(), 'split'))

        operations.sort(reverse=True)
        #print(operations)
        best_op = (operations[0][0], operations[0][2])
        #print(best_op)
        return best_op

    def two_best_children(self, instance):
        """
        Returns the two best children to incorporate the instance
        into in terms of category utility. When selecting the best children it
        sorts first by CU, then by Size, then randomly. 

        input:
            instance: {a1: v1, a2: v2,...} - a hashtable of attr. and values. 
        output:
            (0.2,2),(0.1,3) - the category utility and indices for the two best
            children (the second tuple will be None if there is only 1 child).
        """
        if len(self.children) == 0:
            raise Exception("No children!")

        children_cu = [(self.cu_for_insert(child, instance), child.count,
                        random(), child) for child in self.children]
        children_cu.sort(reverse=True)

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
        new_child.root = self.root
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
            new.root = self.root
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
        new_child.root = self.root
        new_child.update_counts_from_node(best1)
        new_child.update_counts_from_node(best2)
        best1.parent = new_child
        best1.root = new_child.root
        best2.parent = new_child
        best2.root = new_child.root
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
        new_child.root = self.root
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
            child.root = self.root
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

    def get_probability(self, attr, val, alpha=0.001):
        """
        Gets the probability of a particular attribute value at the given
        concept. This takes into account the possibilities that an attribute
        can take any of the values available at the root, or be missing.
        Laplace smoothing is used to place a prior over these possibilites.
        Alpha determines the strength of this prior.
        """
        if attr not in self.av_counts:
            return 0.0

        if val not in self.av_counts[attr]:
            return 0.0

        n_values = len(self.root.av_counts[attr]) + 1

        return ((self.av_counts[attr][val] + alpha) / 
                (1.0 * self.count + alpha * n_values))

    def get_probability_missing(self, attr, alpha=0.001):
        """
        Gets the probability of a particular attribute value at the given
        concept. This takes into account the possibilities that an attribute
        can take any of the values available at the root, or be missing.
        Laplace smoothing is used to place a prior over these possibilites.
        Alpha determines the strength of this prior.
        """
        # the +1 is for the "missing" value
        if attr in self.root.av_counts:
            n_values = len(self.root.av_counts[attr]) + 1
        else:
            n_values = 1

        val_count = 0
        if attr in self.av_counts:
            for val in self.av_counts[attr]:
                val_count += self.av_counts[attr][val]

        return ((self.count - val_count + alpha) / (1.0*self.count + alpha * n_values))

