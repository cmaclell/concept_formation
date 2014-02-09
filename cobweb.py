import json
from random import choice
from random import shuffle

class CobwebTree:

    # static variable for hashing concepts
    counter = 0
    #root = None

    def __init__(self, tree=None):
        """
        The constructor.
        """
        #if self.__class__.root == None:
        #    self.__class__.root = self
        self.concept_name = "Concept" + self._gensym()
        self.count = 0
        self.av_counts = {}
        self.children = []
        self.parent = None

        # check if the constructor is being used as a copy constructor
        if tree:
            self._update_counts_from_node(tree)
            self.parent = tree.parent

            for child in tree.children:
                self.children.append(self.__class__(child))

    def __hash__(self):
        """
        The basic hash function.
        """
        return hash(self.concept_name)

    def _gensym(self):
        self.__class__.counter += 1
        return str(self.__class__.counter)

    def _shallow_copy(self):
        """
        Creates a copy of the current node and its children (but not their
        children)
        """
        temp = self.__class__()
        temp._update_counts_from_node(self)

        for child in self.children:
            temp_child = self.__class__()
            temp_child._update_counts_from_node(child)
            temp.children.append(temp_child)

        return temp

    def _increment_counts(self, instance):
        """
        Increment the counts at the current node according to the specified
        instance.

        input:
            instance: {a1: v1, a2: v2, ...} - a hashtable of attr and values. 
        """
        #TODO do counts need to be floats?
        self.count += 1 
        for attr in instance:
            self.av_counts[attr] = self.av_counts.setdefault(attr,{})
            self.av_counts[attr][instance[attr]] = (self.av_counts[attr].get(
                instance[attr], 0) + 1)
    
    def _update_counts_from_node(self, node):
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

    def _two_best_children(self, instance):
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

        children_cu = [(self._cu_for_insert(child, instance), child) for child
                       in self.children]
        children_cu.sort(key=lambda x: x[0],reverse=True)

        if len(children_cu) == 1:
            return children_cu[0], None 

        return children_cu[0], children_cu[1]

    def _cu_for_insert(self, child, instance):
        """
        Computer the category utility of adding the instance to the specified
        child.
        """
        temp = self.__class__()
        temp._update_counts_from_node(self)
        temp._increment_counts(instance)

        for c in self.children:
            temp_child = self.__class__()
            temp_child._update_counts_from_node(c)
            temp.children.append(temp_child)
            if c == child:
                temp_child._increment_counts(instance)
        return temp._category_utility()

    def _create_new_child(self, instance):
        """
        Creates a new child (to the current node) with the counts initialized by
        the given instance. 
        """
        new_child = self.__class__()
        new_child.parent = self
        new_child._increment_counts(instance)
        self.children.append(new_child)
        return new_child

    def _create_child_with_current_counts(self):
        """
        Creates a new child (to the current node) with the counts initialized by
        the current node's counts.
        """
        if self.count > 0:
            new = self.__class__(self)
            new.parent = self
            self.children.append(new)

    def _cu_for_new_child(self, instance):
        """
        Returns the category utility for creating a new child using the
        particular instance.
        """
        temp = self._shallow_copy()
        temp._increment_counts(instance)
        temp._create_new_child(instance)
        return temp._category_utility()

    def _merge(self, best1, best2, instance):
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
        new_child._update_counts_from_node(best1)
        new_child._update_counts_from_node(best2)
        best1.parent = new_child
        best2.parent = new_child
        new_child.children.append(best1)
        new_child.children.append(best2)
        self.children.remove(best1)
        self.children.remove(best2)
        self.children.append(new_child)

        return new_child

    def _cu_for_merge(self, best1, best2, instance):
        """
        Returns the category utility for merging the two best children.
        NOTE! - I decided that testing a merge does not incorporate the latest
        instance, but waits for a second call of cobweb on the root. The
        original paper says that I should incorporate the instance into the
        merged node, but since we don't do something like this for split I
        didn't do it here. This gives the option to merge multiple nodes before
        incorporating the instance. 

        input:
            best1: the best child in the children array.
            best2: the second best child in the children array.
        output:
            0.02 - the category utility for the merge of best1 and best2.
        """
        temp = self.__class__()
        temp._update_counts_from_node(self)
        temp._increment_counts(instance)

        new_child = self.__class__()
        new_child._update_counts_from_node(best1)
        new_child._update_counts_from_node(best2)
        new_child._increment_counts(instance)
        temp.children.append(new_child)

        for c in self.children:
            if c == best1 or c == best2:
                continue
            temp_child = self.__class__()
            temp_child._update_counts_from_node(c)
            temp.children.append(temp_child)

        return temp._category_utility()

    def _split(self, best):
        """
        Split the best node and promote its children
        """
        self.children.remove(best)
        for child in best.children:
            child.parent = self
            self.children.append(child)

    def _cu_for_fringe_split(self, instance):
        temp = self.__class__()
        temp._update_counts_from_node(self)
        
        temp._create_child_with_current_counts()
        temp._increment_counts(instance)
        temp._create_new_child(instance)

        return temp._category_utility()

    def _cu_for_split(self, best):
        """
        Return the category utility for splitting the best child.
        
        input:
            best1: a child in the children array.
        output:
            0.03 - the category utility for the split of best1.
        """
        temp = self.__class__()
        temp._update_counts_from_node(self)

        for c in self.children + best.children:
            if c == best:
                continue
            temp_child = self.__class__()
            temp_child._update_counts_from_node(c)
            temp.children.append(temp_child)

        return temp._category_utility()

    def verify_counts(self):
        """
        Checks the property that the counts of the children sum to the same
        count as the parent. This is/was useful when debugging.
        """
        if len(self.children) == 0:
            return 

        temp = {}
        temp_count = self.count
        for attr in self.av_counts:
            if attr not in temp:
                temp[attr] = {}
            for val in self.av_counts[attr]:
                temp[attr][val] = self.av_counts[attr][val]

        for child in self.children:
            temp_count -= child.count
            for attr in child.av_counts:
                assert attr in temp
                for val in child.av_counts[attr]:
                    assert val in temp[attr]
                    temp[attr][val] -= child.av_counts[attr][val]

        assert temp_count == 0

        for attr in temp:
            for val in temp[attr]:
                assert temp[attr][val] == 0.0

        for child in self.children:
            child.verify_counts()

    # DEPRECIATED
    #def _is_concept(self, instance):
    #    """
    #    Checks to see if the current node perfectly represents the instance (all
    #    of the attribute values the instance has are probability 1.0 and here
    #    are no extra attribute values).
    #    """
    #    for attribute in self.av_counts:
    #        for value in self.av_counts[attribute]:
    #            if (self.av_counts[attribute][value] / (1.0 * self.count)) != 1.0:
    #                return False
    #            if attribute not in instance:
    #                return False
    #            if instance[attribute] != value:
    #                return False
    #    
    #    for attribute in instance:
    #        if attribute not in self.av_counts:
    #            return False
    #        if instance[attribute] not in self.av_counts[attribute]:
    #            return False
    #        if ((self.av_counts[attribute][instance[attribute]] / 
    #             (1.0 * self.count)) != 1.0):
    #            return False
    #    
    #    return True

    def _cobweb(self, instance):
        """
        Incrementally integrates an instance into the categorization tree
        defined by the current node. This function operates recursively to
        integrate this instance and uses category utility as the heuristic to
        make decisions.
        """

        # instead of checking if the instance is the fringe concept, I
        # check to see if category utility is increased by fringe splitting.
        # this is more generally and will be used by the Labyrinth/Trestle
        # systems to achieve more complex fringe behavior. 
        if not self.children and self._cu_for_fringe_split(instance) == 0:
            self._increment_counts(instance)
            return self

        elif not self.children:
            self._create_child_with_current_counts()
            self._increment_counts(instance)
            return self._create_new_child(instance)
            
        else:
            #TODO is there a cleaner way to do this?
            best1, best2 = self._two_best_children(instance)
            action_cu, best_action = self._get_best_operation(instance, best1,
                                                              best2)
            best1_cu, best1 = best1
            if best2:
                best2_cu, best2 = best2

            if action_cu == 0.0 or best_action == 'best':
                self._increment_counts(instance)
                return best1._cobweb(instance)
            elif best_action == 'new':
                self._increment_counts(instance)
                return self._create_new_child(instance)
            elif best_action == 'merge':
                self._increment_counts(instance)
                new_child = self._merge(best1, best2, instance)
                return new_child._cobweb(instance)
            elif best_action == 'split':
                self._split(best1)
                return self._cobweb(instance)
            else:
                raise Exception("Should never get here.")

    def _get_best_operation(self, instance, best1, best2, 
                            possible_ops=["best", "new", "merge", "split"]):
        """
        Given a set of possible operations, find the best and return its cu and
        the action name.
        """
        best1_cu, best1 = best1
        if best2:
            best2_cu, best2 = best2
        operations = []

        if "best" in possible_ops:
            operations.append((best1_cu,"best"))
        if "new" in possible_ops: 
            operations.append((self._cu_for_new_child(instance),'new'))
        if "merge" in possible_ops and best2:
            operations.append((self._cu_for_merge(best1, best2,
                                                 instance),'merge'))
        if "split" in possible_ops and len(best1.children) > 0:
            operations.append((self._cu_for_split(best1),'split'))

        # pick the best operation
        operations.sort(reverse=True)

        return operations[0]
        
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
            return best1._cobweb_categorize(instance)

    def _expected_correct_guesses(self):
        correct_guesses = 0.0
        for attr in self.av_counts:
            for val in self.av_counts[attr]:
                prob = (self.av_counts[attr][val] / (1.0 * self.count))
                correct_guesses += (prob * prob)
        return correct_guesses

    def _category_utility(self):
        if len(self.children) == 0:
            return 0.0

        category_utility = 0.0

        for child in self.children:
            p_of_child = child.count / (1.0 * self.count)
            category_utility += (p_of_child *
                                 (child._expected_correct_guesses()
                                  - self._expected_correct_guesses()))
        return category_utility / (1.0 * len(self.children))

    def _num_concepts(self):
        """
        Return the number of concepts contained in the tree defined by the
        current node. 
        """
        children_count = 0
        for c in self.children:
           children_count += c._num_concepts() 
        return 1 + children_count 

    def _pretty_print(self, depth=0):
        """
        Prints the categorization tree.
        """
        ret = str(('\t' * depth) + "|-" + str(self.av_counts) + ":" +
                  str(self.count) + '\n')
        
        for c in self.children:
            ret += c._pretty_print(depth+1)

        return ret

    def _output_json(self):
        output = {}
        output['name'] = self.concept_name
        output['size'] = self.count
        output['children'] = []

        temp = {}
        for attr in self.av_counts:
            for value in self.av_counts[attr]:
                temp[attr + " = " + str(value)] = self.av_counts[attr][value]

        for child in self.children:
            output['children'].append(child._output_json())

        output['counts'] = temp

        return output
    
    def __str__(self):
        """
        Converts the categorization tree into a string for printing"
        """
        return self._pretty_print()

    def ifit(self, instance):
        """
        Given an instance incrementally update the categorization tree.
        """
        self._cobweb(instance)

    def fit(self, list_of_instances):
        """
        Call incremental fit on each element in a list of instances.
        """
        for i, instance in enumerate(list_of_instances):
            #print("instance %i of %i" % (i, len(list_of_instances)))
            self.ifit(instance)

    def predict(self, instance):
        """
        Given an instance predict any missing attribute values without
        modifying the tree.
        """
        prediction = {}

        # make a copy of the instance
        for attr in instance:
            prediction[attr] = instance[attr]

        concept = self._cobweb_categorize(instance)
        
        for attr in concept.av_counts:
            if attr in prediction:
                continue
            
            values = []
            for val in concept.av_counts[attr]:
                values += [val] * concept.av_counts[attr][val]

            prediction[attr] = choice(values)

        return prediction

    def _get_probability(self, attr, val):
        if attr not in self.av_counts or val not in self.av_counts[attr]:
            return 0.0
        return (1.0 * self.av_counts[attr][val]) / self.count

    def _prob_attr_value(self, instance, attr, val):
        concept = self._cobweb_categorize(instance)
        return concept._get_probability(attr, val)

    def _flexible_prediction(self, instance):
        probs = []
        for attr in instance:
            temp = {}
            for attr2 in instance:
                if attr == attr2:
                    continue
                temp[attr2] = instance[attr2]
            probs.append(self._prob_attr_value(temp, attr, instance[attr]))
        return sum(probs) / len(probs)

    #depreciated
    #def predict_all(self, instances):
    #    """
    #    Predicts missing attribute values of all instances in the give
    #    list.
    #    """
    #    predictions = [self.predict(instance) for instance in instances]
    #    return predictions

    def train_from_json(self, filename):
        json_data = open(filename, "r")
        instances = json.load(json_data)
        self.fit(instances)
        json_data.close()

    def sequential_prediction(self, filename, length):
        json_data = open(filename, "r")
        instances = json.load(json_data)
        instances = instances[0:length]
        accuracy = []
        for j in range(10):
            shuffle(instances)
            for n, i in enumerate(instances):
                if n >= length:
                    break
                accuracy.append(self._flexible_prediction(i))
                print(self._num_concepts())
                #print(self.root._num_concepts())
                #print(self.root)
                self.ifit(i)
        json_data.close()
        return accuracy

if __name__ == "__main__":
    t = CobwebTree()
    #t.train_from_json("cobweb_test.json")
    print(t.sequential_prediction("cobweb_test.json", 10))
    print(json.dumps(t._output_json()))
    t.verify_counts()
    #print(t)
    #print()

    #test = {}
    #test['HeartChamber'] = "four"
    #print(t.predict(test))


