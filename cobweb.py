import copy
import random

class ClassificationTree:

    def __init__(self, root=None, classification_tree=None):
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

    def increment_counts(self, record):
        self.count += 1.0 
        for a in record:
            self.av_counts[a] = self.av_counts.setdefault(a,{})
            self.av_counts[a][record[a]] = self.av_counts[a].get(record[a],0) + 1.0

    def decrement_counts(self, record):
        self.count -= 1.0 
        for a in record:
            self.av_counts[a] = self.av_counts.setdefault(a,{})
            self.av_counts[a][record[a]] = self.av_counts[a].get(record[a],0) - 1.0
    
    def update_counts_from_node(self, node):
        self.count += node.count
        for a in node.counts:
            for v in node.counts[a]:
                self.av_counts[a] = self.av_counts.setdefault(a,{})
                self.av_counts[a][v] = (self.av_counts[a].get(v,0) +
                                     node.counts[a][v])

    def cobweb(self, record):

        if not self.children: 
            # make a copy of the current node and make it a child 
            self.children.append(ClassificationTree(self.root, self))

            # increment count at the current node
            self.increment_counts(record)
            
            # create a new child for the record
            new_child = ClassificationTree(self.root)
            new_child.increment_counts(record)
            self.children.append(new_child)
        else:
            self.increment_counts(record)

            # calculate the category utility for adding to each child
            children_cu = []
            for i in range(len(self.children)):
                old_c = self.children[i]
                self.children[i] = ClassificationTree(self.root,self.children[i])
                
                # just update the counts at the child


                # don't recursively add yet... 
                # self.children[i].cobweb(record)
                children_cu.append((i,self.category_utility()))          
                self.children[i] = old_c
            
            # sort the children by their cu
            children_cu.sort(key=lambda x: x[1])

            best_cu = float('-inf') 
            action = None 

            # calc cu for creating a new category
            new_child = ClassificationTree(self.root)
            new_child.increment_counts(record)
            self.children.append(new_child)
            cu = self.category_utility()
            self.children.pop()
            if cu >= best_cu:
                best_cu = cu
                action = 'new'
            
            # calc cu for merge of two best
            if len(self.children) > 1:
                index1 = children_cu[0][0]
                index2 = children_cu[1][0]
                if index2 < index1:
                    temp = index1
                    index1 = index2
                    index2 = temp
                old1 = self.children[index1]
                old2 = self.children[index2]
                new_c = ClassificationTree(self.root)
                new_c.increment_counts_from_node(old1)
                new_c.update_counts_from_node(old2)
                new_c.children.append(ClassificationTree(self.root,old1))
                new_c.children.append(ClassificationTree(self.root,old2))
                self.children.pop(index2)
                self.children.pop(index1)
                self.children.append(new_c)
                cu = self.category_utility()
                if cu >= best_cu:
                    best_cu = cu
                    action = 'merge'
                self.children.pop()
                self.children.insert(index1,old1)
                self.children.insert(index2,old2)

            # calc cu for split of best
            old_index = children_cu[0][0]
            old = self.children.pop(old_index)
            for c in old.children:
                self.children.append(ClassificationTree(self.root,c))
            cu = self.category_utility()
            if cu >= best_cu:
                best_cu = cu
                action = 'split'
            for i in range(len(old.children)):
                self.children.pop()
            self.children.insert(old_index,old)

            # take the best action and permenantly update the tree
            if action == 'new':
                new_child = ClassificationTree(self.root)
                new_child.increment_counts(record)
                self.children.append(new_child)
            elif action == 'merge':
                index1 = children_cu[0][0]
                index2 = children_cu[1][0]
                if index2 < index1:
                    temp = index1
                    index1 = index2
                    index2 = temp
                old1 = self.children[index1]
                old2 = self.children[index2]
                new_c = ClassificationTree(self.root)
                new_c.update_counts_from_node(old1)
                new_c.update_counts_from_node(old2)
                new_c.children.append(old1)
                new_c.children.append(old2)
                self.children.pop(index2)
                self.children.pop(index1)
                self.children.append(new_c)
            elif action == 'split':
                old = self.children.pop(children_cu[0][0])
                for c in old.children:
                    self.children.append(c)
            else:
                self.children[children_cu[0][0]].cobweb(record)


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
        children_count = 0
        for c in self.children:
           children_count += c.num_concepts() 
        return 1 + children_count 

    def pretty_print(self,depth=0):
        for i in range(depth):
            print "\t",
        print "|-" + str(self.av_counts)
        
        for c in self.children:
            c.pretty_print(depth+1)

    def __str__(self):
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

