import copy
import random

class ClassificationTree:

    def __init__(self, root=None, classification_tree=None):
        """
        The constructor.
        """
        # keep track of the root of the tree
        if root:
            self.root = root
        else:
            self.root = self

        # check if the constructor is being used as a copy constructor
        if classification_tree:
            self.count = classification_tree.count
            self.av_counts = copy.deepcopy(classification_tree.counts)
            self.children = copy.deepcopy(classification_tree.children)
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
    
    def update_counts_from_node(self, node):
        self.count += node.count
        for a in node.counts:
            for v in node.counts[a]:
                self.av_counts[a] = self.av_counts.setdefault(a,{})
                self.av_counts[a][v] = (self.av_counts[a].get(v,0) +
                                     node.counts[a][v])

    def create_new_child(self,instance):
        """
        Creates a new child (to the current node) with the counts initialized by
        the given instance. 
        """
        new_child = ClassificationTree(self.root)
        new_child.increment_counts(instance)
        self.children.append(new_child)

    def create_child_with_current_counts(self):
        """
        Creates a new child (to the current node) with the counts initialized by
        the current node's counts.
        """
        self.children.append(ClassificationTree(self.root,self))


    def cobweb(self, instance):
        """
        Incrementally integrates an instance into the categorization tree
        defined by the current node. This function operates recursively to
        integrate this instance and uses category utility as the heuristic to
        make decisions.
        """
        if not self.children: 
            self.create_child_with_current_counts()
            self.increment_counts(instance)
            self.create_new_child(instance)

        else:
            self.increment_counts(instance)

            # calculate the category utility for adding to each child
            children_cu = []
            for child in self.children:
                child.increment_counts(instance)
                children_cu.append((self.category_utility(),child))
                child.decrement_counts(instance)
            
            # sort the children by their cu
            children_cu.sort()

            best_cu = float('-inf') 
            action = None 

            # calc cu for creating a new category
            
            # calc cu for merge of two best

            # calc cu for split of best

            # take the best action and permenantly update the tree

    def category_utility(self):
        """
        The category utility is a local heuristic calculation to determine if
        the split of instances across the children increases the ability to
        guess from the parent node. 
        """
        category_utility = 0.0

        exp_parent_guesses = self.expected_correct_guesses()

        for child in self.children:
            p_of_child = child.count / self.count
            exp_child_guesses = child.expected_correct_guesses()
            category_utility += p_of_child * (exp_child_guesses -
                                              exp_parent_guesses)

        # return the category utility normalized by the number of children.
        return category_utility / (1.0 * len(self.children))

    def expected_correct_guesses(self):
        """
        The number of attribute value guesses we would be expected to get
        correct using the current concept.
        """
        exp_count = 0.0
        for attribute in self.av_counts:
            for value in self.av_counts[attribute]:
                exp_count == (self.av_counts[attribute][value] / self.count)**2
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
        print "|-" + str(self.av_counts)
        
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
    t = ClassificationTree()

    
    for i in range(40):
        r = {}
        r['n'] = random.randint(0,5) 
        t.cobweb(r)

    t.pretty_print()
    print t.category_utility()

