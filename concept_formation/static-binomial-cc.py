from concept_formation.cobweb import CobwebNode
from time import time

from tqdm import tqdm

from concept_formation.cobweb import CobwebTree
from visualize import visualize
from preprocess_text import _load_text

def get_path(node):
    path = [node]
    while path[-1].parent:
        path.append(path[-1].parent)
    return path

class ContextualCobwebTree(CobwebTree):

    def __init__(self, window):
        """
        Note window only specifies how much context to add to each side,
        doesn't include the anchor word.

        E.g., to get a window with 2 before and 2 words after the anchor, then
        set the window=2
        """
        super().__init__()
        self.root = ContextualCobwebNode()
        self.root.tree = self

        self.n_concepts = 1
        self.window = window
        self.instance = None
        self.prune_threshold = 0.01

    def ifit(self, instance):
        self.instance = instance
        return super().ifit(instance)
    
    def fit_to_text(self, text):
        context_nodes = []

        for anchor_idx in tqdm(range(len(text))):
            while ((len(context_nodes) < anchor_idx + self.window + 1) and
                   len(context_nodes) < len(text)):
                context_nodes.append(self.categorize({'anchor': text[len(context_nodes)]}))

            for _ in range(2):
                for i in range(2*self.window + 1):
                    idx = anchor_idx - self.window + i
                    if idx < 0 or idx >= len(text): 
                        continue
                    instance = self.create_instance(idx, self.window, text, context_nodes)
                    context_nodes[idx] = self.categorize(instance)

            instance = self.create_instance(anchor_idx, self.window, text, context_nodes)
            instance['_idx'] = str(anchor_idx)

            context_nodes[anchor_idx] = self.ifit(instance)

    def create_instance(self, anchor_idx, window, text, context_nodes):
        context = context_nodes[max(0, anchor_idx-self.window): anchor_idx]
        context += context_nodes[anchor_idx+1:anchor_idx+self.window+1]

        instance = {}
        for n in context:
            for c in get_path(n):
                instance[c] = instance.get(c, 0) + 1

        instance['anchor'] = text[anchor_idx]
        return instance

    def cobweb(self, instance):
        """
        Updated version of cobweb algorithm that updates cross-concept references
        """
        current = self.root

        while current:
            # the current.count == 0 here is for the initially empty tree.
            if not current.children and (current.is_exact_match(instance) or
                                         current.count == 0):
                # print("leaf match")
                current.increment_counts(instance)
                break

            elif not current.children:
                # print("fringe split")
                new = current.__class__(current)
                current.parent = new
                new.children.append(current)

                if new.parent:
                    new.parent.children.remove(current)
                    new.parent.children.append(new)
                else:
                    self.root = new

                # REGISTER UPDATE
                current.register_new_parent(new)

                new.increment_counts(instance)
                current = new.create_new_child(instance)
                self.n_concepts += 2
                break

            else:
                best1_cu, best1, best2 = current.two_best_children(instance)
                _, best_action = current.get_best_operation(instance, best1,
                                                            best2, best1_cu)

                # print(best_action)
                if best_action == 'best':
                    current.increment_counts(instance)
                    current = best1
                elif best_action == 'new':
                    current.increment_counts(instance)
                    current = current.create_new_child(instance)
                    self.n_concepts += 1
                    break
                elif best_action == 'merge':
                    current.increment_counts(instance)
                    new_child = current.merge(best1, best2)

                    # REGISTER UPDATE
                    best1.register_new_parent(new_child)
                    best2.register_new_parent(new_child)

                    current = new_child
                    self.n_concepts += 1

                elif best_action == 'split':
                    # REGISTER UPDATE
                    best1.register_delete()
                    current.split(best1)
                    self.n_concepts -= 1
                else:
                    raise Exception('Best action choice "' + best_action +
                                    '" not a recognized option. This should be'
                                    ' impossible...')

        return current


class ContextualCobwebNode(CobwebNode):

    def __init__(self, otherNode=None):
        self.n_context_elements = 0
        self.registered = set()

        super().__init__(otherNode)

    def get_concepts(self):
        s = set([self.concept_id])
        for c in self.children:
            s.update(c.get_concepts())
        return s

    def test_valid(self, valid_concepts):
        for attr in self.av_counts:
            if isinstance(attr, ContextualCobwebNode):
                assert attr.concept_id in valid_concepts
        for c in self.children:
            c.test_valid(valid_concepts)

    def increment_counts(self, instance):
        """
        Adds binomial distribution for estimating concept counts
        """
        self.count += 1

        for attr in instance:
            self.av_counts[attr] = self.av_counts.setdefault(attr, {})

            if isinstance(attr, ContextualCobwebNode):
                self.av_counts[attr]['count'] = (self.av_counts[attr].get('count', 0) +
                        instance[attr])

                # only count if it is a terminal, don't count nonterminals
                if not attr.children:
                    self.n_context_elements += instance[attr]

                attr.register(self)

            else:
                prior_count = self.av_counts[attr].get(instance[attr], 0)
                self.av_counts[attr][instance[attr]] = prior_count + 1

        # self.prune_low_probability()

    def prune_low_probability(self):
        del_nodes = []
        del_av = []

        for attr in self.attrs():
            if isinstance(attr, ContextualCobwebNode):
                if (self.av_counts[attr]['count'] / self.n_context_elements) < self.tree.prune_threshold:
                    del_nodes.append(attr)
            else:
                for val in self.av_counts[attr]:
                    if (self.av_counts[attr][val] / self.count) < self.tree.prune_threshold:
                        del_av.append((attr, val))
            
        # del_nodes = [attr for attr in self.attrs(lambda x: isinstance(x, ContextualCobweb))
        #         if (self.av_counts[attr]['count'] / self.n_context_elements <
        #             self.tree.prune_threshold))]

        for n in del_nodes:
            n.unregister(self)
            del self.av_counts[n]

        for a,v in del_av:
            del self.av_counts[a][v]
            if len(self.av_counts[a]) == 0:
                del self.av_counts[a]

        # del_av = [attr, val for attr in self.attrs(lambda x: not isinstance(x,  for val in self.av_counts[attr] if isinstance

    def update_counts_from_node(self, node):
        """
        Adds binomial distribution for estimating concept counts
        """
        self.count += node.count
        self.n_context_elements += node.n_context_elements
        
        for attr in node.attrs('all'):
            self.av_counts[attr] = self.av_counts.setdefault(attr, {})
            if isinstance(attr, ContextualCobwebNode):
                self.av_counts[attr]['count'] = (self.av_counts[attr].get('count', 0) +
                        node.av_counts[attr]['count'])
                attr.register(self)

            else:
                for val in node.av_counts[attr]:
                    self.av_counts[attr][val] = (self.av_counts[attr].get(val,
                        0) + node.av_counts[attr][val])

        # self.prune_low_probability()

    def register_delete(self):
        for attr in self.av_counts:
            if isinstance(attr, ContextualCobwebNode):
                attr.unregister(self)

        for c in self.registered:
            del c.av_counts[self]

        if (self.tree.instance is not None and self in self.tree.instance):
            del self.tree.instance[self]

    def register_new_parent(self, parent):
        for c in self.registered:
            parent.register(c)
            if parent not in c.av_counts:
                c.av_counts[parent] = {}
            for val in c.av_counts[self]:
                c.av_counts[parent]['count'] = (c.av_counts[parent].get('count', 0) +
                        c.av_counts[self]['count'])

        if (self.tree.instance is not None and self in self.tree.instance):
            self.tree.instance[parent] = (self.tree.instance.get(parent, 0) +
                    self.tree.instance[self])


    def __hash__(self):
        return hash(self.concept_id)

    def __str__(self):
        return "Concept-{}".format(self.concept_id)

    def __getitem__(self, indices):
        return True

    def register(self, other):
        self.registered.add(other)

    def unregister(self, other):
        if other in self.registered:
            self.registered.remove(other)

    def expected_correct_guesses(self):
        """
        Modified to handle attribute that are concepts to tally and compute
        correct guesses over all concepts in path.
        """
        correct_guesses = 0.0
        attr_count = 0
        # concept_attr_count = 0
        concept_vals = {}
        n_concepts = self.tree.n_concepts # len(self.tree.root.get_concepts())
        eval_concepts = 0

        # # TEST
        # concept_counts = {}
        # new_concept_counts = {}
        
        for attr in self.attrs():

            if isinstance(attr, ContextualCobwebNode):
                eval_concepts += 1
                prob = self.av_counts[attr]['count'] / self.n_context_elements
                correct_guesses += (prob * prob) / n_concepts
                correct_guesses += ((1-prob) * (1-prob)) / n_concepts
                
                # # TEST
                # new_concept_counts[attr] = (new_concept_counts.get(attr, 0) + 
                #         self.av_counts[attr]['count'])
                # if not attr.children:
                #     curr = attr
                #     concept_counts[attr] = (concept_counts.get(attr, 0) +
                #             self.av_counts[attr]['count'])
                #     
                #     while curr.parent:
                #         curr = curr.parent
                #         concept_counts[curr] = (concept_counts.get(curr, 0) +
                #                 self.av_counts[attr]['count'])

            else:
                attr_count += 1
                for val in self.av_counts[attr]:
                    prob = (self.av_counts[attr][val]) / self.count
                    correct_guesses += (prob * prob)

        # # TEST
        # for attr in set(concept_counts).union(set(new_concept_counts)):
        #     assert attr in concept_counts
        #     assert attr in new_concept_counts
        #     assert new_concept_counts[attr] == concept_counts[attr]

        correct_guesses += (n_concepts - eval_concepts) / n_concepts

        # count the concept nodes as a single attr
        # this is basically the weighting factor between anchor and context
        attr_count += 1

        return correct_guesses / attr_count
    
    def output_json(self):
        """
        Outputs the categorization tree in JSON form.
        This is a modification of the :meth:`CobwebNode.output_json
        <concept_formation.cobweb.CobwebNode.output_json>` to handle numeric
        values.
        :return: an object that contains all of the structural information of
            the node and its children
        :rtype: obj
        """
        output = {}
        output["name"] = "Concept" + str(self.concept_id)
        output["size"] = self.count
        output["children"] = []

        temp = {}

        for attr in self.attrs('all'):

            if isinstance(attr, ContextualCobwebNode):
                temp[str(attr)] = {'count': self.av_counts[attr]['count'],
                        'n': self.n_context_elements,
                        'p': self.av_counts[attr]['count'] / self.n_context_elements}

            else:
                temp[str(attr)] = {}
                for val in self.av_counts[attr]:
                    temp[str(attr)][str(val)] = self.av_counts[attr][val]

        for child in self.children:
            output["children"].append(child.output_json())

        output["counts"] = temp

        return output


if __name__ == "__main__":

    tree = ContextualCobwebTree(window=2)

    stop_words = {*"i me my myself we our ours ourselves you your yours yourself "
                   "yourselves he him his himself she her hers herself it its "
                   "itself they them their theirs themselves what which who whom "
                   "this that these those am is are was were be been being have "
                   "has had having do does did doing a an the and but if or "
                   "because as until while of at by for with about against "
                   "between into through during before after above below to from "
                   "up down in out on off over under again further then once here "
                   "there when where why how all any both each few more most "
                   "other some such no nor not only own same so than too very s t "
                   "can will just don't should now th".split(' ')}

    for text_num in range(1):
        text = [word for word in _load_text(text_num) if word not in
                stop_words][:100]

        print('iterations needed', len(text))
        start = time()
        tree.fit_to_text(text)
        print(time()-start)
        print(text_num)
    visualize(tree)

    valid_concepts = tree.root.get_concepts()
    tree.root.test_valid(valid_concepts)







