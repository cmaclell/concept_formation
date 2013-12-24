import random

class CobwebTree:

    def __init__(self, tree=None):
        """
        The constructor.
        """
        self.count = 0.0
        self.av_counts = {}
        self.children = []

        # check if the constructor is being used as a copy constructor
        if tree:
            self._update_counts_from_node(tree)

            for child in tree.children:
                self.children.append(self.__class__(child))

    def _shallow_copy(self):
        """
        Creates a copy of the current node and its children (but not their
        children)
        """
        temp = self.__class__()
        temp._update_counts_from_node(self)

        # important to maintain order
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
        self.count += 1.0 
        for attr in instance:
            self.av_counts[attr] = self.av_counts.setdefault(attr,{})
            self.av_counts[attr][instance[attr]] = (self.av_counts[attr].get(
                instance[attr], 0) + 1.0)
    
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
        Returns the indices of the two best children to incorporate the instance
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
        new_child._increment_counts(instance)
        self.children.append(new_child)

    def _create_child_with_current_counts(self):
        """
        Creates a new child (to the current node) with the counts initialized by
        the current node's counts.
        """
        if self.count > 0:
            self.children.append(self.__class__(self))

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
        new_child._update_counts_from_node(best1)
        new_child._update_counts_from_node(best2)
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
            self.children.append(child)

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

    def _check_children_eq_parent(self):
        """
        Checks the property that the counts of the children sum to the same
        count as the parent. This is/was useful when debugging.
        """
        if len(self.children) == 0:
            return

        child_count = 0.0
        for child in self.children:
            child_count += child.count
        assert self.count == child_count

    def _is_concept(self, instance):
        """
        Checks to see if the current node perfectly represents the instance (all
        of the attribute values the instance has are probability 1.0 and here
        are no extra attribute values).
        """
        for attribute in self.av_counts:
            for value in self.av_counts[attribute]:
                if (self.av_counts[attribute][value] / self.count) != 1.0:
                    return False
                if attribute not in instance:
                    return False
                if instance[attribute] != value:
                    return False
        
        for attribute in instance:
            if attribute not in self.av_counts:
                return False
            if instance[attribute] not in self.av_counts[attribute]:
                return False
            if ((self.av_counts[attribute][instance[attribute]] / self.count) !=
                1.0):
                return False
        
        return True

    def _cobweb(self, instance):
        """
        Incrementally integrates an instance into the categorization tree
        defined by the current node. This function operates recursively to
        integrate this instance and uses category utility as the heuristic to
        make decisions.
        """
        if not self.children and self._is_concept(instance): 
            self._increment_counts(instance)

        elif not self.children:
            self._create_child_with_current_counts()
            self._increment_counts(instance)
            self._create_new_child(instance)
            
        else:
            best1, best2 = self._two_best_children(instance)
            best1_cu, best1 = best1
            if best2:
                best2_cu, best2 = best2

            operations = []
            operations.append((best1_cu,"best"))
            operations.append((self._cu_for_new_child(instance),'new'))
            if best2:
                operations.append((self._cu_for_merge(best1, best2,
                                                     instance),'merge'))
            if len(best1.children) > 0:
                operations.append((self._cu_for_split(best1),'split'))

            # pick the best operation
            operations.sort(reverse=True)
            action_cu, best_action = operations[0]

            #print(operations)

            if action_cu == 0.0 or best_action == 'best':
                self._increment_counts(instance)
                best1._cobweb(instance)
            elif best_action == 'new':
                self._increment_counts(instance)
                self._create_new_child(instance)
            elif best_action == 'merge':
                self._increment_counts(instance)
                new_child = self._merge(best1, best2, instance)
                new_child._cobweb(instance)
            elif best_action == 'split':
                self._split(best1)
                self._cobweb(instance)
            else:
                raise Exception("Should never get here.")

    def _category_utility_nominal(self, parent):
        category_utility = 0.0
        for attr in self.av_counts:
            for val in self.av_counts[attr]:
                category_utility += ((self.av_counts[attr][val] /
                                      self.count)**2 -
                                     (parent.av_counts[attr][val] /
                                      parent.count)**2)
        
        return category_utility

    def _category_utility(self):
        if len(self.children) == 0:
            return 0.0

        category_utility = 0.0

        for child in self.children:
            p_of_child = child.count / self.count
            category_utility += (p_of_child *
                                 child._category_utility_nominal(self))
        return category_utility / (1.0 * len(self.children))

    def _category_utility_old(self):
        """
        The category utility is a local heuristic calculation to determine if
        the split of instances across the children increases the ability to
        guess from the parent node. 
        """
        if len(self.children) == 0:
            return 0.0

        category_utility = 0.0

        exp_parent_guesses = self._expected_correct_guesses()

        for child in self.children:
            p_of_child = child.count / self.count
            exp_child_guesses = child._expected_correct_guesses()
            category_utility += p_of_child * (exp_child_guesses -
                                              exp_parent_guesses)

        return category_utility / (1.0 * len(self.children))

    def _expected_correct_guesses(self):
        """
        The number of attribute value guesses we would be expected to get
        correct using the current concept.
        """
        if self.count == 0:
            return 0.0

        exp_count = 0.0
        for attr in self.av_counts:
            for val in self.av_counts[attr]:
                if not isinstance(val, float):
                    exp_count += (self.av_counts[attr][val] / self.count)**2
        return exp_count

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
        print(('\t' * depth) + "|-" + str(self.av_counts) + ":" + str(self.count))
        
        for c in self.children:
            c._pretty_print(depth+1)

    def __str__(self):
        """
        Converts the categorization tree into a string.
        """
        ret = str(self.av_counts)
        for c in self.children:
            ret += "\n" + str(c)

        return ret

if __name__ == "__main__":
    t = CobwebTree()

    instances = []

    # concept 1 mammal
    for i in range(1):
        r = {}
        r['BodyCover'] = "hair"
        r['HeartChamber'] = "four"
        r['BodyTemp'] = "regulated"
        r['Fertilization'] = "internal"
        r['Name'] = "mammal"
        instances.append(r)

    # concept 2 bird 
    for i in range(1):
        r = {}
        r['BodyCover'] = "feathers"
        r['HeartChamber'] = "four"
        r['BodyTemp'] = "regulated"
        r['Fertilization'] = "internal"
        r['Name'] = "bird"
        instances.append(r)

    # concept 3 reptile 
    for i in range(1):
        r = {}
        r['BodyCover'] = "cornified-skin"
        r['HeartChamber'] = "imperfect-four"
        r['BodyTemp'] = "unregulated"
        r['Fertilization'] = "internal"
        r['Name'] = "reptile"
        instances.append(r)

    # concept 4 amphibian 
    for i in range(1):
        r = {}
        r['BodyCover'] = "moist-skin"
        r['HeartChamber'] = "three"
        r['BodyTemp'] = "unregulated"
        r['Fertilization'] = "external"
        r['Name'] = "amphibian"
        instances.append(r)

    # concept 5 fish
    for i in range(1):
        r = {}
        r['BodyCover'] = "scales"
        r['HeartChamber'] = "two"
        r['BodyTemp'] = "unregulated"
        r['Fertilization'] = "external"
        r['Name'] = "fish"
        instances.append(r)

    random.shuffle(instances)
    for i in instances:
        t._cobweb(i)

    t._pretty_print()
    print(t._category_utility())

