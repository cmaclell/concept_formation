from functools import partial
from collections import Counter
# from multiprocess import Pool
from tqdm import tqdm
import timeit

from concept_formation.cobweb import CobwebNode
from concept_formation.cobweb import CobwebTree
from visualize import visualize
from preprocess_text import load_text, stop_words


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
        self.prune_threshold = 0.1
        self.anchor_weight = 1
        self.context_weight = 2

        self.log_times = False

    def ifit(self, instance):
        self.instance = instance
        return super().ifit(instance)

    def fit_to_text(self, text):
        context_nodes = []

        for anchor_idx, anchor_wd in enumerate(tqdm(text)):
            if self.log_times:
                start = timeit.default_timer()
            while ((len(context_nodes) < anchor_idx + self.window + 1) and
                   len(context_nodes) < len(text)):
                context_nodes.append(self.categorize({'anchor': text[len(context_nodes)]}))

            for _ in range(2):
                # for i in range(2*self.window + 1):
                #     idx = anchor_idx - self.window + i
                for i in range(self.window + 1):
                    idx = anchor_idx + i
                    if idx < 0 or idx >= len(text):
                        continue
                    instance = self.create_instance(idx, anchor_wd, context_nodes)
                    context_nodes[idx] = self.categorize(instance)
                # print([str(c) for c in context_nodes[anchor_idx - self.window: anchor_idx + self.window + 1]])

            instance = self.create_instance(anchor_idx, anchor_wd, context_nodes)

            context_nodes[anchor_idx] = self.ifit(instance)

            if self.log_times:
                stop = timeit.default_timer()
                with open('out.csv', 'a') as fout:
                    fout.write("{},{:.8f}\n".format(anchor_idx, stop - start))

    def create_instance(self, anchor_idx, word, context_nodes):
        context = context_nodes[max(0, anchor_idx-self.window): anchor_idx]
        context += context_nodes[anchor_idx+1:anchor_idx+self.window+1]

        instance = {}
        for n in context:
            for c in get_path(n):
                instance[c] = instance.get(c, 0) + 1

        instance['anchor'] = word
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
                current.increment_counts(instance, track=True)
                break

            elif not current.children:
                # print("fringe split")
                new = current.__class__(current, track=True)
                current.parent = new
                new.children.append(current)

                if new.parent:
                    new.parent.children.remove(current)
                    new.parent.children.append(new)
                else:
                    self.root = new

                # REGISTER UPDATE
                current.register_new_parent(new)

                new.increment_counts(instance, track=True)
                new.prune()
                current = new.create_new_child(instance)
                self.n_concepts += 2
                break

            else:
                best1_cu, best1, best2 = current.two_best_children(instance)
                _, best_action = current.get_best_operation(instance, best1,
                                                            best2, best1_cu)

                # print(best_action)
                if best_action == 'best':
                    current.increment_counts(instance, track=True)
                    current.prune()
                    current = best1
                elif best_action == 'new':
                    current.increment_counts(instance, track=True)
                    current.prune()
                    current = current.create_new_child(instance, track=True)
                    self.n_concepts += 1
                    break
                elif best_action == 'merge':
                    current.increment_counts(instance, track=True)
                    current.prune()
                    new_child = current.merge(best1, best2, track=True)

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

    def create_categorization_instance(self, anchor_idx, word, context_nodes):
        context = context_nodes[max(0, anchor_idx-self.window): anchor_idx]
        context += context_nodes[anchor_idx+1:anchor_idx+self.window+1]

        instance = {}
        for n in filter(None, context):
            for c in get_path(n):
                instance[c] = instance.get(c, 0) + 1

        instance['anchor'] = word
        return instance

    def guess_missing(self, text, options, options_needed):
        """
        None used to represent missing words
        """
        assert len(options) >= options_needed
        missing_idx = text.index(None)
        context_nodes = []

        for anchor_idx, anchor_wd in tqdm(enumerate(text)):
            while ((len(context_nodes) < anchor_idx + self.window + 1) and
                    len(context_nodes) < len(text)):
                if len(context_nodes) == missing_idx:
                    context_nodes.append(None)
                    continue
                context_nodes.append(self.categorize({'anchor': text[len(context_nodes)]}))

            if anchor_idx == missing_idx:
                continue

            for _ in range(2):
                for i in range(self.window + 1):
                    idx = anchor_idx + i
                    if idx < 0 or idx >= len(text) or idx == missing_idx:
                        continue
                    instance = self.create_categorization_instance(idx, anchor_wd, context_nodes)
                    context_nodes[idx] = self.categorize(instance)

            instance = self.create_categorization_instance(anchor_idx, anchor_wd, context_nodes)
            context_nodes[anchor_idx] = self.categorize(instance)

        missing_instance = self.create_instance(missing_idx, anchor_wd, context_nodes)
        del missing_instance['anchor']

        concept = self.categorize(missing_instance)
        while sum([(option in self.__get_anchor_counts(concept)) for option in options]) < options_needed:
            concept = concept.parent

        return max(options, key=lambda opt: self.__get_anchor_counts(concept)[opt])

    def __get_anchor_counts(self, node):
        if not node.children:
            return Counter(node.av_counts['anchor'])
        return sum([self.__get_anchor_counts(child) for child in node.children], start=Counter())


class ContextualCobwebNode(CobwebNode):
    def __init__(self, otherNode=None, track=False):
        self.n_context_elements = 0
        self.registered = set()

        self.concept_id = self.gensym()
        self.count = 0.0
        self.av_counts = {}
        self.children = []
        self.parent = None
        self.tree = None

        if otherNode:
            self.tree = otherNode.tree
            self.parent = otherNode.parent
            self.update_counts_from_node(otherNode, track)

            for child in otherNode.children:
                self.children.append(self.__class__(child))

    def create_new_child(self, instance, track=False):
        """
        Create a new child (to the current node) with the counts initialized by
        the *given instance*.

        This is the operation used for creating a new child to a node and
        adding the instance to it.

        :param instance: The instance currently being categorized
        :type instance: :ref:`Instance<instance-rep>`
        :return: The new child
        :rtype: CobwebNode
        """
        new_child = self.__class__()
        new_child.parent = self
        new_child.tree = self.tree
        new_child.increment_counts(instance, track)
        self.children.append(new_child)
        return new_child

    def merge(self, best1, best2, track=False):
        """
        Merge the two specified nodes.

        A merge operation introduces a new node to be the merger of the the two
        given nodes. This new node becomes a child of the current node and the
        two given nodes become children of the new node.

        :param best1: The child of the current node with the best category
            utility
        :type best1: CobwebNode
        :param best2: The child of the current node with the second best
            category utility
        :type best2: CobwebNode
        :return: The new child node that was created by the merge
        :rtype: CobwebNode
        """
        new_child = self.__class__()
        new_child.parent = self
        new_child.tree = self.tree

        new_child.update_counts_from_node(best1, track)
        new_child.update_counts_from_node(best2, track)
        best1.parent = new_child
        # best1.tree = new_child.tree
        best2.parent = new_child
        # best2.tree = new_child.tree
        new_child.children.append(best1)
        new_child.children.append(best2)
        self.children.remove(best1)
        self.children.remove(best2)
        self.children.append(new_child)

        return new_child

    def get_concepts(self):
        s = set([self])
        for c in self.children:
            s.update(c.get_concepts())
        return s

    def test_valid(self, valid_concepts):
        for attr in self.av_counts:
            if isinstance(attr, ContextualCobwebNode):
                assert attr.concept_id in valid_concepts
        for c in self.children:
            c.test_valid(valid_concepts)

    def increment_counts(self, instance, track=False):
        """
        Adds binomial distribution for estimating concept counts
        """
        self.count += 1

        for attr in instance:
            self.av_counts.setdefault(attr, {})

            if isinstance(attr, ContextualCobwebNode):
                self.av_counts[attr]['count'] = (self.av_counts[attr].get('count', 0) +
                        instance[attr])

                # only count if it is a terminal, don't count nonterminals
                if not attr.children:
                    self.n_context_elements += instance[attr]

                if track:
                    attr.register(self)

            else:
                prior_count = self.av_counts[attr].get(instance[attr], 0)
                self.av_counts[attr][instance[attr]] = prior_count + 1

    def update_counts_from_node(self, node, track=False):
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

                if track:
                    attr.register(self)

            else:
                for val in node.av_counts[attr]:
                    self.av_counts[attr][val] = (self.av_counts[attr].get(val,
                        0) + node.av_counts[attr][val])

        # self.prune_low_probability()

    def prune(self):
        del_nodes = []
        del_av = []

        for attr in self.attrs('all'):
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
            # if len(self.av_counts[a]) == 0:
            #     del self.av_counts[a]

        # del_av = [attr, val for attr in self.attrs(lambda x: not isinstance(x,  for val in self.av_counts[attr] if isinstance

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
        context_guesses = 0.0
        n_concepts = self.tree.n_concepts

        for attr in self.attrs():

            if isinstance(attr, ContextualCobwebNode):
                prob = self.av_counts[attr]['count'] / self.n_context_elements
                # context_guesses += (prob * prob)
                # context_guesses += ((1-prob) * (1-prob))
                context_guesses -= 2 * prob * (1-prob)
                # Should add 1 after this, but it's wrapped up into the end processing

            else:
                for val in self.av_counts[attr]:
                    prob = (self.av_counts[attr][val]) / self.count
                    correct_guesses += (prob * prob) * self.tree.anchor_weight

        context_guesses += n_concepts
        context_guesses *= self.tree.context_weight / n_concepts

        return correct_guesses + context_guesses

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

        temp['aa-context-aa'] = {}

        for attr in self.attrs('all'):

            if isinstance(attr, ContextualCobwebNode):
                temp['aa-context-aa'][str(attr)] = (self.av_counts[attr]['count'] /
                        self.n_context_elements * self.count)
                # temp[str(attr)] = {'count': self.av_counts[attr]['count'],
                #         'n': self.n_context_elements,
                #         'p': self.av_counts[attr]['count'] / self.n_context_elements}

            else:
                temp[str(attr)] = {}
                for val in self.av_counts[attr]:
                    temp[str(attr)][str(val)] = self.av_counts[attr][val]

        for child in self.children:
            output["children"].append(child.output_json())

        output["counts"] = temp

        # from pprint import pprint
        # pprint(output)

        return output


if __name__ == "__main__":

    tree = ContextualCobwebTree(window=2)

    for text_num in range(1):
        text = [word for word in load_text(text_num) if word not in
                stop_words][:500]
        tree.fit_to_text(text)
    visualize(tree)

    # valid_concepts = tree.root.get_concepts()
    # tree.root.test_valid(valid_concepts)

    # print(tree.guess_missing(['correct', 'view', 'probably', None, 'two', 'extremes'], ['lies', 'admittedly', 'religious', 'opinions', 'worship'], 1))
