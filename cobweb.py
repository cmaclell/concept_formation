import copy
import random

class ConceptTree:

    def __init__(self, concept_tree=None):
        """
        The constructor.
        """
        # check if the constructor is being used as a copy constructor
        if concept_tree:
            self.count = concept_tree.count
            self.av_counts = copy.deepcopy(concept_tree.av_counts)
            self.children = copy.deepcopy(concept_tree.children)
        else:
            self.count = 0.0
            self.av_counts = {}
            self.children = []

    def increment_counts(self, instance):
        """
        Increment the counts at the current node according to the specified
        instance.

        input:
            instance: {a1: v1, a2: v2, ...} - a hashtable of attr and values. 
        """
        self.count += 1.0 
        for a in instance:
            self.av_counts[a] = self.av_counts.setdefault(a,{})
            self.av_counts[a][instance[a]] = (self.av_counts[a].get(instance[a],
                                                                    0) + 1.0)

    def decrement_counts(self, instance):
        """
        Decrement the counts at the current node according to the specified
        instance.
        
        input:
            instance: {a1: v1, a2: v2, ...} - a hashtable of attr and values. 
        """
        self.count -= 1.0 
        for a in instance:
            self.av_counts[a] = self.av_counts.setdefault(a,{})
            self.av_counts[a][instance[a]] = (self.av_counts[a].get(instance[a],
                                                                    0) - 1.0)
            # for clarity in printing we remove the values and attributes
            if self.av_counts[a][instance[a]] == 0:
                del self.av_counts[a][instance[a]]
            if self.av_counts[a] == {}:
                del self.av_counts[a]
    
    def update_counts_from_node(self, node):
        self.count += node.count
        for a in node.av_counts:
            for v in node.av_counts[a]:
                self.av_counts[a] = self.av_counts.setdefault(a,{})
                self.av_counts[a][v] = (self.av_counts[a].get(v,0) +
                                     node.av_counts[a][v])

    def create_new_child(self,instance):
        """
        Creates a new child (to the current node) with the counts initialized by
        the given instance. 
        """
        new_child = ConceptTree()
        new_child.increment_counts(instance)
        self.children.append(new_child)

    def create_child_with_current_counts(self):
        """
        Creates a new child (to the current node) with the counts initialized by
        the current node's counts.
        """
        self.children.append(ConceptTree(self))

    def two_best_children(self,instance):
        """
        Returns the indices of the two best children to incorporate the instance
        into in terms of category utility.

        input:
            instance: {a1: v1, a2: v2,...} - a hashtable of attr. and values. 
        output:
            (0.2,2),(0.1,3) - the category utility and indices for the two best
            children (the second tuple will be None if there is only 1 child).
        """
        #TODO - Run some simple check to ensure the structure has not
        #       been canged by this operation.
        if len(self.children) == 0:
            raise Exception("No children!")
        
        self.increment_counts(instance)
        children_cu = []
        for i in range(len(self.children)):
            self.children[i].increment_counts(instance)
            children_cu.append((self.category_utility(),i))
            self.children[i].decrement_counts(instance)
        self.decrement_counts(instance)
        children_cu.sort()

        if len(self.children) == 1:
            return children_cu[0], None 

        return children_cu[0], children_cu[1]

    def new_child(self,instance):
        """
        Updates root count and adds child -- permenant.
        """
        return self.cu_for_new_child(instance,False)

    def cu_for_new_child(self,instance, undo=True):
        """
        Returns the category utility for creating a new child using the
        particular instance.
        """
        self.increment_counts(instance)
        self.create_new_child(instance)
        cu = self.category_utility()
        if undo:
            self.children.pop()
            self.decrement_counts(instance)
        return cu

    def merge(self,best1,best2):
        """
        A version of merge that is permenant.
        """
        return self.cu_for_merge(best1,best2,False)

    def cu_for_merge(self, best1, best2, undo=True):
        """
        Returns the category utility for merging the two best children.
        NOTE! - I decided that testing a merge does not incorporate the latest
        instance, but waits for a second call of cobweb on the root. The
        original paper says that I should incorporate the instance into the
        merged node, but since we don't do something like this for split I
        didn't do it here. This gives the option to merge multiple nodes before
        incorporating the instance. 

        input:
            best1: 1 - an index for a child in the children array.
            best2: 2 - an index for a child in the children array.
        output:
            0.02 - the category utility for the merge of best1 and best2.
        """
        #TODO - Might want to consider adding the instance to the merged node.
        first = best1
        second = best2

        if second < first:
            temp = first 
            first = second 
            second = temp

        first_c = self.children[first]
        second_c = self.children[second]

        new_c = ConceptTree()
        new_c.update_counts_from_node(first_c)
        new_c.update_counts_from_node(second_c)

        self.children.pop(second)
        self.children.pop(first)
        self.children.append(new_c)

        cu = self.category_utility()

        if undo:
            self.children.pop()
            self.children.insert(first,first_c)
            self.children.insert(second,second_c)

        return cu

    def split(self,best):
        """
        Permemantly split the best.
        """
        return self.cu_for_split(best,False)

    def cu_for_split(self,best,undo=True):
        """
        Return the category utility for splitting the best child.
        
        input:
            best1: 0 - an index for a child in the children array.
        output:
            0.03 - the category utility for the split of best1.
        """
        best_c = self.children.pop(best)
        for child in best_c.children:
            self.children.append(child)
        cu = self.category_utility()

        if undo:
            for i in range(len(best_c.children)):
                self.children.pop()
            self.children.insert(best,best_c)

        return cu

    def check_children_eq_parent(self):
        if len(self.children) == 0:
            return

        child_count = 0.0
        for child in self.children:
            child_count += child.count
        assert self.count == child_count

    def is_instance(self,instance):
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

    def closest_matching_child(self,instance):
        best = 0
        smallest_diff = float('inf')
        for i in range(len(self.children)):
            child = self.children[i]
            sum_diff = 0.0
            count = 0.0
            for attribute in child.av_counts:
                for value in self.av_counts[attribute]:
                    count += 1
                    if attribute in instance and instance[attribute] == value:
                        sum_diff += 1.0 - (self.av_counts[attribute][value] /
                                       self.count)
                    else:
                        sum_diff += 1.0

            if count > 0:
                sum_diff /= count
            else:
                sum_diff = float('inf')

            if sum_diff < smallest_diff:
                best = i
                smallest_diff = sum_diff
        
        return best

    def cobweb(self, instance):
        """
        Incrementally integrates an instance into the categorization tree
        defined by the current node. This function operates recursively to
        integrate this instance and uses category utility as the heuristic to
        make decisions.
        """
        if not self.children and self.is_instance(instance): 
            self.increment_counts(instance)

        elif not self.children:
            #print self.av_counts, self.count, instance
            self.create_child_with_current_counts()
            self.increment_counts(instance)
            self.create_new_child(instance)
            
        else:
            best1, best2 = self.two_best_children(instance)
            operations = []
            operations.append((best1[0],"best"))
            operations.append((self.cu_for_new_child(instance),'new'))
            if best2:
                operations.append((self.cu_for_merge(best1[1],best2[1]),'merge'))
            if len(self.children[best1[1]].children):
                operations.append((self.cu_for_split(best1[1]),'split'))
            operations.sort(reverse=True)
            print operations

            best_action = operations[0][1]
            action_cu = operations[0][0]
            if action_cu == 0.0:
                self.increment_counts(instance)
                self.children[self.closest_matching_child(instance)].cobweb(instance)
            elif best_action == 'best':
                self.increment_counts(instance)
                self.children[best1[1]].cobweb(instance)
            elif best_action == 'new':
                self.new_child(instance)
            elif best_action == 'merge':
                self.merge(best1[1],best2[1])
                self.cobweb(instance)
            elif best_action == 'split':
                self.split(best1[1])
                self.cobweb(instance)
            else:
                raise Exception("Should never get here.")


    def category_utility(self):
        """
        The category utility is a local heuristic calculation to determine if
        the split of instances across the children increases the ability to
        guess from the parent node. 
        """
        if len(self.children) == 0:
            return 0.0

        # should be initialized to a small non-zero value so that we favor
        # merging like nodes.
        category_utility = 0.0

        exp_parent_guesses = self.expected_correct_guesses()

        for child in self.children:
            p_of_child = child.count / self.count
            exp_child_guesses = child.expected_correct_guesses()
            category_utility += p_of_child * (exp_child_guesses -
                                              exp_parent_guesses)
            #print (p_of_child, exp_child_guesses, exp_parent_guesses,
            #       category_utility)

        # return the category utility normalized by the number of children.
        print category_utility, category_utility / (1.0 * len(self.children)), len(self.children)

        return category_utility / (1.0 * len(self.children))

    def expected_correct_guesses(self):
        """
        The number of attribute value guesses we would be expected to get
        correct using the current concept.
        """
        exp_count = 0.0
        for attribute in self.av_counts:
            for value in self.av_counts[attribute]:
                exp_count += (self.av_counts[attribute][value] / self.count)**2
        return exp_count

    def num_concepts(self):
        """
        Return the number of concepts contained in the tree defined by the
        current node. 
        """
        children_count = 0
        for c in self.children:
           children_count += c.num_concepts() 
        return 1 + children_count 

    def pretty_print(self,depth=0):
        """
        Prints the categorization tree.
        """
        for i in range(depth):
            print "\t",
        print "|-" + str(self.av_counts) + ":" + str(self.count)
        
        for c in self.children:
            c.pretty_print(depth+1)

    def __str__(self):
        """
        Converts the categorization tree into a string.
        """
        ret = str(self.av_counts)
        for c in self.children:
            ret += "\n" + str(c)

        return ret

if __name__ == "__main__":
    t = ConceptTree()

    instances = []
    
    # concept 1 lizard
    for i in range(10):
        r = {}
        r['a1'] = 1 
        r['a3'] = 1
        instances.append(r)

    # concept 2 bird 
    for i in range(2):
        r = {}
        r['a1'] = 0 
        r['a3'] = 1
        instances.append(r)

    #random.shuffle(instances)
    for i in instances:
        t.cobweb(i)

    t.pretty_print()
    print t.category_utility()

