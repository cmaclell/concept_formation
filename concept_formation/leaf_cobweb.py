from cProfile import run
from collections import Counter
# from multiprocess import Pool
from tqdm import tqdm
import pickle
from os.path import dirname, join
from sys import setrecursionlimit
from os import listdir
import resource
from time import time
import random
from itertools import chain

from concept_formation.utils import random_tiebreaker
from concept_formation.cobweb import CobwebNode
from concept_formation.cobweb import CobwebTree
from concept_formation.utils import skip_slice
from visualize import visualize
from preprocess_text import load_text, stop_words, load_microsoft_qa

TREE_RECURSION = 0x10000

# May segfault without this line. 0x100 is a guess at the size of stack frame.
try:
    resource.setrlimit(resource.RLIMIT_STACK,
                       [0x100 * TREE_RECURSION, resource.RLIM_INFINITY])
except ValueError:
    print(Warning("Warning: Saving this model may result in a segfault"))
setrecursionlimit(TREE_RECURSION)

SAVE = False
LOAD = False
MODELS_PATH = join(dirname(__file__), 'saved_models')
MODEL_SAVE_LOCATION = join(MODELS_PATH, 'saved_model_%s' % time())
if LOAD:
    for option_num, s in enumerate(listdir(MODELS_PATH)):
        print('%s:' % option_num, s)
    index = int(input('Which model would you like to load? '))
    MODEL_LOAD_LOCATION = join(MODELS_PATH, listdir(MODELS_PATH)[index])
print(listdir(MODELS_PATH)[0])
run

random.seed(16)
minor_key = '#MinorCtxt#'
major_key = '#MajorCtxt#'
anchor_key = 'anchor'

# Debugging output:
word_to_leaf = {}


def get_path(node):
    while node:
        yield node
        node = node.parent


class ContextualCobwebTree(CobwebTree):

    def __init__(self, minor_window, major_window):
        """
        Note window only specifies how much context to add to each side,
        doesn't include the anchor word.

        E.g., to get a window with 2 before and 2 words after the anchor, then
        set the window=2
        """
        super().__init__()
        self.root = ContextualCobwebNode()
        self.root.tree = self

        self.minor_window = minor_window
        self.major_window = major_window
        self.anchor_weight = 16
        self.minor_weight = 0
        self.major_weight = 2

    def _sanity_check_instance(self, instance):
        for attr in instance:
            try:
                hash(attr)
                attr[0]
            except Exception:
                raise ValueError('Invalid attribute: '+str(attr) +
                                 ' of type: '+str(type(attr)) +
                                 ' in instance: '+str(instance) +
                                 ',\n'+type(self).__name__ +
                                 ' only works with hashable ' +
                                 'and subscriptable attributes' +
                                 ' (e.g., strings).')
            try:
                if not isinstance(instance[attr], str) and attr[0] != '_':
                    map(hash, instance[attr])
            except Exception:
                raise ValueError('Invalid value: '+str(instance[attr]) +
                                 ' of type: '+str(type(instance[attr])) +
                                 ' in instance: '+str(instance) +
                                 ',\n'+type(self).__name__ +
                                 ' only works with hashable values.')
            if instance[attr] is None:
                raise ValueError("Attributes with value None should"
                                 " be manually removed.")

    def fit_to_text_wo_stopwords(self, text):
        """filters stop words here"""
        output_nodes = []
        text = [word for word in text if word not in stop_words]

        for anchor_idx, anchor_wd in enumerate(tqdm(text)):
            output_nodes.append(self.ifit(
                self.create_instance(anchor_idx, anchor_wd, text)))
            word_to_leaf.setdefault(anchor_wd, [])
            word_to_leaf[anchor_wd].append(output_nodes[-1])

        return output_nodes

    def create_instance(self, anchor_idx, anchor_word, text,
                        filter_stop_for_minor=False):
        major_context = self.categorize_window(
            text, anchor_idx, self.major_window)
        if filter_stop_for_minor:
            raise NotImplementedError
        else:
            minor_context = self.categorize_window(
                text, anchor_idx, self.minor_window)

        return {minor_key: minor_context,
                major_key: major_context,
                anchor_key: anchor_word,
                '_idx': anchor_idx}

    def categorize_window(self, text, anchor_idx, window_size):
        return list(map(self.similarity_categorize, map(
            self.anchor_only_inst, self.surrounding(
                text, anchor_idx, window_size))))

    def similarity_categorize(self, instance):
        current = self.root

        def similarity(node):
            """Average probability"""
            correct_guesses = 0
            if anchor_key in instance:
                correct_guesses += node.av_counts[anchor_key].get(
                    instance[anchor_key], 0) * self.anchor_weight
            if minor_key in instance:
                correct_guesses += sum(
                    node.av_counts[minor_key].get(ctxt, 0)
                    for ctxt in instance[minor_key]) * self.minor_weight
            if major_key in instance:
                correct_guesses += sum(
                    node.av_counts[major_key].get(ctxt, 0)
                    for ctxt in instance[major_key]) * self.major_weight

            return correct_guesses  # / node.count

        while current:
            if not current.children:
                return current

            best = max(current.children, key=similarity)
            current = best
        return current

    def anchor_only_inst(self, word):
        return {anchor_key: word}

    def surrounding(self, sequence, center, dist):
        return list(
            # Observe that we use trailing window since ahead nodes don't have leaves yet
            skip_slice(sequence, max(0, center-dist-dist), center+1, center))  # TODO

    def guess_missing(self, text, options, options_needed=1,
                      filter_stop_for_minor=False):
        text = [word for word in text if word not in stop_words]
        index = text.index(None)
        major_context = self.categorize_window(text, index, self.major_window)
        if filter_stop_for_minor:
            raise NotImplementedError
        else:
            minor_context = self.categorize_window(
                text, index, self.minor_window)

        concept = self.similarity_categorize(
            {minor_key: minor_context, major_key: major_context})
        path = [concept]
        while sum([(option in concept.av_counts[anchor_key])
                   for option in options]) < options_needed:
            concept = concept.parent
            path.append(concept)
            if concept is None:
                print('Words not seen')
                return random.choice(options)
                raise ValueError('None of the options have been seen')

        return max(options,
                   key=lambda opt: concept.av_counts[anchor_key].get(opt, 0)), path, minor_context


class ContextualCobwebNode(CobwebNode):
    def increment_counts(self, instance):
        """
        Increment the counts at the current node according to the specified
        instance.
        :param instance: A new instances to incorporate into the node.
        :type instance: :ref:`Instance<instance-rep>`
        """
        self.count += 1
        for attr in instance:
            if attr == minor_key or attr == major_key:
                self.av_counts.setdefault(
                    attr, Counter()).update(instance[attr])
                continue

            self.av_counts.setdefault(attr, {})
            self.av_counts[attr][instance[attr]] = (
                self.av_counts[attr].get(instance[attr], 0) + 1)

    def update_counts_from_node(self, node):
        """
        Adds binomial distribution for estimating concept counts
        """
        self.count += node.count

        for attr in node.attrs('all'):
            if attr == minor_key or attr == major_key:
                self.av_counts.setdefault(
                    attr, Counter()).update(node.av_counts[attr])
                continue

            counts = self.av_counts.setdefault(attr, {})
            for val in node.av_counts[attr]:
                counts[val] = counts.get(val, 0)+node.av_counts[attr][val]

    def set_counts_from_node(self, node):
        raise NotImplementedError()

    def __str__(self):
        return "Concept-{}".format(self.concept_id)

    def __repr__(self):
        return str(self)

    def expected_correct_guesses(self, instance=None):
        """
        Returns the number of correct guesses that are expected from the given
        concept.
        This is the sum of the probability of each attribute value squared.
        This function is used in calculating category utility.
        :return: the number of correct guesses that are expected from the given
                 concept.
        :rtype: float
        """
        correct_guesses = 0.0
        if instance is None:
            attrs = self.attrs()
        else:
            attrs = self.attrs(attr_filter=lambda x: x in instance)
        for attr in attrs:
            if attr[0] == '_':
                continue
            temp = 0
            counts = self.av_counts[attr]
            for val in counts:
                prob = counts[val] / self.count
                temp += (prob * prob)

            if attr == major_key:
                correct_guesses += temp * self.tree.major_weight
            elif attr == minor_key:
                correct_guesses += temp * self.tree.minor_weight
            elif attr == anchor_key:
                correct_guesses += temp * self.tree.anchor_weight
            else:
                assert False

        return correct_guesses

    def category_utility(self, instance=None):
        """
        Return the category utility of a particular division of a concept into
        its children.
        Category utility is always calculated in reference to a parent node and
        its own children. This is used as the heuristic to guide the concept
        formation process. Category Utility is calculated as:
        .. math::
            CU(\\{C_1, C_2, \\cdots, C_n\\}) = \\frac{1}{n} \\sum_{k=1}^n
            P(C_k) \\left[ \\sum_i \\sum_j P(A_i = V_{ij} | C_k)^2 \\right] -
            \\sum_i \\sum_j P(A_i = V_{ij})^2
        where :math:`n` is the numer of children concepts to the current node,
        :math:`P(C_k)` is the probability of a concept given the current node,
        :math:`P(A_i = V_{ij} | C_k)` is the probability of a particular
        attribute value given the concept :math:`C_k`, and :math:`P(A_i =
        V_{ij})` is the probability of a particular attribute value given the
        current node.
        In general this is used as an internal function of the cobweb algorithm
        but there may be times when it would be useful to call outside of the
        algorithm itself.
        :return: The category utility of the current node with respect to its
                 children.
        :rtype: float
        """
        if len(self.children) == 0:
            return 0.0

        child_correct_guesses = 0.0

        for child in self.children:
            p_of_child = child.count / self.count
            child_correct_guesses += (p_of_child *
                                      child.expected_correct_guesses(instance))

        return ((child_correct_guesses - self.expected_correct_guesses(instance))
                / len(self.children))

    def get_best_operation(self, instance, best1, best2, best1_cu,
                           possible_ops=("best", "new", "merge", "split")):
        """
        Given an instance, the two best children based on category utility and
        a set of possible operations, find the operation that produces the
        highest category utility, and then return the category utility and name
        for the best operation. In the case of ties, an operator is randomly
        chosen.
        Given the following starting tree the results of the 4 standard Cobweb
        operations are shown below:
        .. image:: images/Original.png
            :width: 200px
            :align: center
        * **Best** - Categorize the instance to child with the best category
          utility. This results in a recurisve call to :meth:`cobweb
          <concept_formation.cobweb.CobwebTree.cobweb>`.
            .. image:: images/Best.png
                :width: 200px
                :align: center
        * **New** - Create a new child node to the current node and add the
          instance there. See: :meth:`create_new_child
          <concept_formation.cobweb.CobwebNode.create_new_child>`.
            .. image:: images/New.png
                :width: 200px
                :align: center
        * **Merge** - Take the two best children, create a new node as their
          mutual parent and add the instance there. See: :meth:`merge
          <concept_formation.cobweb.CobwebNode.merge>`.
            .. image:: images/Merge.png
                    :width: 200px
                    :align: center
        * **Split** - Take the best node and promote its children to be
          children of the current node and recurse on the current node. See:
          :meth:`split <concept_formation.cobweb.CobwebNode.split>`
            .. image:: images/Split.png
                :width: 200px
                :align: center
        Each operation is entertained and the resultant category utility is
        used to pick which operation to perform. The list of operations to
        entertain can be controlled with the possible_ops parameter. For
        example, when performing categorization without modifying knoweldge
        only the best and new operators are used.
        :param instance: The instance currently being categorized
        :type instance: :ref:`Instance<instance-rep>`
        :param best1: A tuple containing the relative cu of the best child and
            the child itself, as determined by
            :meth:`CobwebNode.two_best_children`.
        :type best1: (float, CobwebNode)
        :param best2: A tuple containing the relative cu of the second best
            child and the child itself, as determined by
            :meth:`CobwebNode.two_best_children`.
        :type best2: (float, CobwebNode)
        :param possible_ops: A list of operations from ["best", "new", "merge",
            "split"] to entertain.
        :type possible_ops: ["best", "new", "merge", "split"]
        :return: A tuple of the category utility of the best operation and the
            name of the best operation.
        :rtype: (cu_bestOp, name_bestOp)
        """
        if not best1:
            raise ValueError("Need at least one best child.")

        operations = []

        if "best" in possible_ops:
            operations.append((best1_cu, "best"))
        if "new" in possible_ops:
            operations.append((self.cu_for_new_child(instance), 'new'))
        if "merge" in possible_ops and len(self.children) > 2 and best2:
            operations.append((self.cu_for_merge(best1, best2, instance),
                               'merge'))
        if "split" in possible_ops and len(best1.children) > 0:
            operations.append((self.cu_for_split(best1, instance), 'split'))

        operations.sort(reverse=True)

        return random_tiebreaker(operations, key=lambda x: x[0])

    def compute_relative_CU_const(self, instance):
        """
        Computes the constant value that is used to convert between CU and
        relative CU scores. The constant value is basically the category
        utility that results from adding the instance to the root, but none of
        the children. It can be computed directly as:
        .. math::
            const = \\frac{1}{n} \\sum_{k=1}^{n} \\left[
            \\frac{C_k.count}{count + 1} \\sum_i \\sum_j P(A_i = V_{ij} |
            C_k)^2 \\right] - \\sum_i \\sum_j P(A_i = V_{ij} | UpdatedRoot)^2
        where :math:`n` is the number of children of the root, :math:`C_k` is
        child :math:`k`,  :math:`C_k.count` is the number of instances stored
        in child :math:`C_k`, :math:`count` is the number of instances stored
        in the root. Finally, :math:`UpdatedRoot` is a copy of the root that
        has been updated with the counts of the instance.
        :param instance: The instance currently being categorized
        :type instance: :ref:`Instance<instance-rep>`
        :return: The value of the constant used to relativize the CU.
        :rtype: float
        """
        temp = self.shallow_copy()
        temp.increment_counts(instance)
        ec_root_u = temp.expected_correct_guesses(instance)

        const = 0
        for c in self.children:
            const += c.count * c.expected_correct_guesses(instance)

        const /= self.count + 1  # Turns counts into probabilities
        const -= ec_root_u
        const /= len(self.children)
        return const

    def relative_cu_for_insert(self, child, instance):
        """
        Computes a relative CU score for each insert operation. The relative CU
        score is more efficient to calculate for each insert operation and is
        guranteed to have the same rank ordering as the CU score so it can be
        used to determine which insert operation is best. The relative CU can
        be computed from the CU using the following transformation.
        .. math::
            relative\\_cu(cu) = (cu - const) * n * (count + 1)
        where :math:`const` is the one returned by
        :meth:`CobwebNode.compute_relative_CU_const`, :math:`n` is the number
        of children of the current node, and :math:`count` is the number of
        instances stored in the current node (the root).
        The particular :math:`const` value was chosen to make the calculation
        of the relative cu scores for each insert operation efficient. When
        computing the CU for inserting the instance into a particular child,
        the terms in the formula above can be expanded and many of the
        intermediate calculations cancel out. After these cancelations,
        computing the relative CU for inserting into a particular child
        :math:`C_i` reduces to:
        .. math::
            relative\\_cu\\_for\\_insert(C_i) = (C_i.count + 1) * \\sum_i
            \\sum_j P(A_i = V_{ij}| UpdatedC_i)^2 - (C_i.count) * \\sum_i
            \\sum_j P(A_i = V_{ij}| C_i)^2
        where :math:`UpdatedC_i` is a copy of :math:`C_i` that has been updated
        with the counts from the given instance.
        By computing relative_CU scores instead of CU scores for each insert
        operation, the time complexity of the underlying Cobweb algorithm is
        reduced from :math:`O(B^2 \\times log_B(n) \\times AV)` to
        :math:`O(B \\times log_B(n) \\times AV)` where :math:`B` is the average
        branching factor of the tree, :math:`n` is the number of instances
        being categorized, :math:`A` is the average number of attributes per
        instance, and :math:`V` is the average number of values per attribute.
        :param child: a child of the current node
        :type child: CobwebNode
        :param instance: The instance currently being categorized
        :type instance: :ref:`Instance<instance-rep>`
        :return: the category utility of adding the instance to the given node
        :rtype: float
        """
        temp = child.shallow_copy()
        temp.increment_counts(instance)
        return ((child.count + 1) * temp.expected_correct_guesses(instance) -
                child.count * child.expected_correct_guesses(instance))

    def cu_for_insert(self, child, instance):
        """
        Compute the category utility of adding the instance to the specified
        child.
        This operation does not actually insert the instance into the child it
        only calculates what the result of the insertion would be. For the
        actual insertion function see: :meth:`CobwebNode.increment_counts` This
        is the function used to determine the best children for each of the
        other operations.
        :param child: a child of the current node
        :type child: CobwebNode
        :param instance: The instance currently being categorized
        :type instance: :ref:`Instance<instance-rep>`
        :return: the category utility of adding the instance to the given node
        :rtype: float
        .. seealso:: :meth:`CobwebNode.two_best_children` and
            :meth:`CobwebNode.get_best_operation`
        """
        temp = self.shallow_copy()
        temp.increment_counts(instance)

        for c in self.children:
            temp_child = c.shallow_copy()
            temp.children.append(temp_child)
            temp_child.parent = temp
            if c == child:
                temp_child.increment_counts(instance)
        return temp.category_utility(instance)

    def cu_for_new_child(self, instance):
        """
        Return the category utility for creating a new child using the
        particular instance.
        This operation does not actually create the child it only calculates
        what the result of creating it would be. For the actual new function
        see: :meth:`CobwebNode.create_new_child`.
        :param instance: The instance currently being categorized
        :type instance: :ref:`Instance<instance-rep>`
        :return: the category utility of adding the instance to a new child.
        :rtype: float
        .. seealso:: :meth:`CobwebNode.get_best_operation`
        """
        temp = self.shallow_copy()
        for c in self.children:
            temp.children.append(c.shallow_copy())

        # temp = self.shallow_copy()

        temp.increment_counts(instance)
        temp.create_new_child(instance)
        return temp.category_utility(instance)

    def cu_for_merge(self, best1, best2, instance):
        """
        Return the category utility for merging the two best children.
        This does not actually merge the two children it only calculates what
        the result of the merge would be. For the actual merge operation see:
        :meth:`CobwebNode.merge`
        :param best1: The child of the current node with the best category
            utility
        :type best1: CobwebNode
        :param best2: The child of the current node with the second best
            category utility
        :type best2: CobwebNode
        :param instance: The instance currently being categorized
        :type instance: :ref:`Instance<instance-rep>`
        :return: The category utility that would result from merging best1 and
            best2.
        :rtype: float
        .. seealso:: :meth:`CobwebNode.get_best_operation`
        """
        temp = self.shallow_copy()
        temp.increment_counts(instance)

        new_child = self.__class__()
        new_child.tree = self.tree
        new_child.parent = temp
        new_child.update_counts_from_node(best1)
        new_child.update_counts_from_node(best2)
        new_child.increment_counts(instance)
        temp.children.append(new_child)

        for c in self.children:
            if c == best1 or c == best2:
                continue
            temp_child = c.shallow_copy()
            temp.children.append(temp_child)

        return temp.category_utility(instance)

    def cu_for_split(self, best, instance):
        """
        Return the category utility for splitting the best child.
        This does not actually split the child it only calculates what the
        result of the split would be. For the actual split operation see:
        :meth:`CobwebNode.split`. Unlike the category utility calculations for
        the other operations split does not need the instance because splits
        trigger a recursive call on the current node.
        :param best: The child of the current node with the best category
            utility
        :type best: CobwebNode
        :return: The category utility that would result from splitting best
        :rtype: float
        .. seealso:: :meth:`CobwebNode.get_best_operation`
        """
        temp = self.shallow_copy()

        for c in chain(self.children, best.children):
            if c == best:
                continue
            temp_child = c.shallow_copy()
            temp.children.append(temp_child)

        return temp.category_utility(instance)

    def is_exact_match(self, instance):
        """
        Returns true if the concept exactly matches the instance.
        :param instance: the instance currently being categorized
        :type instance: :ref:`Instance<instance-rep>`
        :return: whether the instance perfectly matches the concept
        :rtype: boolean
        .. seealso:: :meth:`CobwebNode.get_best_operation`
        """
        anchors = self.av_counts.get(anchor_key, ())
        if len(anchors) == 1 and instance[anchor_key] in anchors:
            global overlaps
            overlaps += 1
        return len(anchors) == 1 and instance[anchor_key] in anchors


overlaps = 0


def create_questions(text, question_length, nimposters, n):
    questions = []
    for _ in range(n):
        pos = random.randint(0, len(text)-question_length-1)
        blank = random.randint(2, question_length-3)
        question = text[pos:pos+question_length]
        answer = question[blank]
        question[blank] = None
        questions.append((question,
                         [answer, *(random.choice(text)
                          for _ in range(nimposters))]))
    return questions


def test_microsoft(model):
    correct = 0
    for total, (question, answers, answer) in enumerate(load_microsoft_qa()):
        if model.guess_missing(question, answers, 1) == answers[answer]:
            correct += 1
    total += 1
    return correct / total


if __name__ == "__main__":

    if LOAD:
        tree = pickle.load(open(MODEL_LOAD_LOCATION, mode='rb'))
    else:
        tree = ContextualCobwebTree(1, 4)

    for text_num in range(1):
        text = list(load_text(text_num))[:5000]

        tree.fit_to_text_wo_stopwords(text)
        text = [word for word in text[:5000] if word not in stop_words]
        text_counts = Counter(text)

        print(overlaps)
        print('total overlaps',
              len(list(text_counts.elements()))-len(text_counts))

        questions = create_questions(text, 10, 4, 200)

        correct = 0
        answers_needed = 1
        for question in questions:
            guess, path, minctxt = tree.guess_missing(
                *question, answers_needed)
            answer = question[1][0]
            # print(question)
            if guess == answer:
                correct += 1
                ...  # print('correct')
            else:
                '''print()
                print('question', question[0])
                print('minor ctxt', minctxt)
                ctxt_words = [next(iter(concept.av_counts[anchor_key])) for concept in minctxt]
                print('ctxt_words', ctxt_words)
                print('ctxt_counts', [text_counts[word] for word in ctxt_words])
                for word in ctxt_words:
                    for leaf in word_to_leaf[word]:
                        print(word, list(get_path(leaf)),)
                print('-'*90)
                print('path', [concept.concept_id for concept in path])
                print('initial counts', path[0].av_counts)
                print('id', path[-1].concept_id)
                print('counts', [(answer, path[-1].av_counts[anchor_key].get(answer, 0)) for answer in question[1]])
                print('answer', answer)
                print()'''
                ...  # print('incorrect. guessed "{}" when "{}" was correct'.format(guess, answer))
        print(correct/len(questions), 'answers needed: ', answers_needed)
    visualize(tree)

    # valid_concepts = tree.root.get_concepts()
    # tree.root.test_valid(valid_concepts)

    if SAVE:
        setrecursionlimit(TREE_RECURSION)
        pickle.dump(tree, open(MODEL_SAVE_LOCATION, mode='xb'))
