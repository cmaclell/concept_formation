"""
The file contains the code for training contextual cobweb from large datasets
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
from os.path import dirname
from os.path import join
import re
import copy
import random
from tqdm import tqdm
from csv import reader
from concept_formation.preprocess_text import stop_words, _preprocess
from concept_formation.leaf_cobweb import ContextualCobwebTree as LeafTree
from concept_formation.static_binomial_cc import ContextualCobwebTree as StaticTree
# from concept_formation.literal_cobweb import ContextualCobwebTree as LiteralTree
# from concept_formation.word_model import ContextualCobwebTree as WordTree
from visualize import visualize
from itertools import chain, product, islice
from collections import Counter
import multiprocessing

from sklearn.metrics import adjusted_rand_score
from concept_formation.cluster import CU

# NUM_SENTENCES = 515
# NUM_SENTENCES = 1040
NUM_SENTENCES = 20 

random.seed()


module_path = dirname(__file__)


def load_microsoft_qa():
    let_to_num = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
    with open(join(module_path, 'data_files',
                   'MSC_testing_data.csv'), newline='') as questions:
        with open(join(module_path, 'data_files',
                       'MSC_test_answer.csv'), newline='') as answers:
            data = zip(reader(questions), reader(answers))
            # Remove top row
            data.__next__()
            for quest, ans in data:
                yield (list(_preprocess(quest[1], True)), quest[2:],
                       let_to_num[ans[1]])


def test_microsoft(model):
    correct = 0
    for total, (question, answers, answer) in enumerate(load_microsoft_qa()):
        if model.guess_missing(question, answers, 1) == answers[answer]:
            correct += 1
    total += 1
    return correct / total


def full_ms_sentences():
    return list(map(question_to_sentence, load_microsoft_qa()))


def question_to_sentence(question):
    sent, quest, ans = question
    sent[sent.index(None)] = quest[ans]
    return sent


def generate_ms_sentence_variant_synonyms(nsynonyms=2, ncopies=5, nms_sentences=1500):
    """
    args:
        nsynonyms (int): number of possible synonyms
        ncopies (int): number of times each sentence appears"""
    sentences = list(full_ms_sentences())[:nms_sentences]
    for _ in range(ncopies):
        for sentence in sentences:
            yield synonymize(sentence, nsynonyms)


def synonymize(sentence, nsynonyms=2):
    return [(word+'-%s' % random.randint(1, nsynonyms) if word else None) for word in sentence]


def synonymize_question(question, nsynonyms=2):
    return (synonymize(question[0], nsynonyms), synonymize(question[1], nsynonyms), question[2])


def generate_ms_sentence_variant_homographs(homographs=[], nsenses=2, ncopies=5, nms_sentences=1500):
    """
    homographs (Seq): List of words to be turned into homographs
    nsenses (int): Number of senses for each homographs"""
    sentences = list(full_ms_sentences())[:nms_sentences]
    for _ in range(ncopies):
        for sentence in sentences:
            for i in range(nsenses):
                yield [('#%s#' % word if word in homographs else word+'-%s' % i) for word in sentence]


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


def word_to_hidden_homograph_attr(word, homo_type):
    return word[1:-1] + '-' + homo_type


def get_path(node):
    while node:
        yield node
        node = node.parent


def nearby_leaf_pairs(leaves, cutoff):
    for leaf_1, leaf_2 in product(leaves, repeat=2):
        first_path = list(get_path(leaf_1))[:cutoff]
        if any(node in first_path for node in islice(get_path(leaf_2), 0, cutoff)):
            yield (leaf_1, leaf_2)


def synonym_similarity(cutoff, word_to_leaf):
    """cutoff: maximum number of levels above before leaves are considered to far apart"""
    # The base of a word to the variants of that base
    base_to_variants = {}
    # The base of a word to the number of occurances
    base_to_counts = Counter()
    for variant, leaves in word_to_leaf.items():
        base = word_to_base(variant)
        base_to_variants.setdefault(base, set()).add(variant)
        base_to_counts[base] += sum(leaf.count for leaf in leaves)
    # Pick a base-word uniformly at random, then pick a leaf pair (could be two of the same leaf)
    # Expected number of pairs whose common ancestor is within cutoff nodes of both.
    result = 0
    for base, variants in base_to_variants.items():
        for node_1, node_2 in nearby_leaf_pairs(chain.from_iterable(word_to_leaf[variant] for variant in variants), cutoff):
            result += node_1.count * node_2.count / (base_to_counts[base] * base_to_counts[base])
    return result / len(base_to_variants)


def leaf_to_homograph_counts(leaf, anch_key, maj_key):
    # result = Counter(word_to_suffix(next(iter(ctxt_leaf.av_counts[anch_key]))) for ctxt_leaf in leaf.av_counts[maj_key].elements())
    # del result[None]
    return Counter(leaf.av_counts['_homograph'])


def homograph_difference(homographs, cutoff, word_to_leaf, anch_key, maj_key):
    # Given two of the same word, expected proportion
    # that have their common ancestor above cutoff generations above them or are the same.
    total = 0
    for homograph in homographs:
        temp = 0
        leaves = list(word_to_leaf[homograph])
        homograph_counts = {leaf: leaf_to_homograph_counts(leaf, anch_key, maj_key) for leaf in leaves}
        for leaf_1, leaf_2 in nearby_leaf_pairs(leaves, cutoff):
            temp += sum(homograph_counts[leaf_1].values()) * sum(homograph_counts[leaf_2].values()) - sum(count * homograph_counts[leaf_2][sense] for sense, count in homograph_counts[leaf_1].items())
        temp /= sum(sum(counts.values()) for counts in homograph_counts.values()) ** 2
        total += temp
        # Σ

    return 1 - total / len(homographs)


# For Paper
def top_n(data, n):
    # print(Counter(chain.from_iterable([sent for sent in data])))
    # return Counter(chain.from_iterable([sent[4:-4] for sent in data])).most_common(n)
    return Counter(chain.from_iterable(data)).most_common(n)


def word_to_base(word):
    return word.split('-')[0]


def word_to_suffix(word):
    if len(word.split('-')) == 1:
        return None
    return word.split('-')[1]


def word_to_variant(word, num):
    return word+'-%s' % num


def list_path(node):
    path = []
    while node:
        path.append(node)
        node = node.parent
    return path


def distance(node_1, node_2):
    if node_1 == node_2:
        return 0
    path_1 = list_path(node_1)
    path_2 = list_path(node_2)
    for in_common, (n1, n2) in enumerate(zip(reversed(path_1), reversed(path_2))):
        if n1 != n2:
            break
    return max(len(path_1), len(path_2)) - in_common

def withhold(data, banned_words, window=4):

    training_sent = []
    test_sent = []

    for sent in data:
        for word in banned_words:
            if word in sent:
                test_sent.append(sent)
                break
        training_sent.append(sent)

    return training_sent, test_sent

def peter_withhold(data, banned_words, window=4):
    new_data = []
    withheld_instances = []
    for sent in data:
        new_sent = []
        for idx, word in enumerate(sent):
            if word in banned_words:
                withheld_instances.append((sent[max(0, idx-window):idx+1+window], min(window, idx)))
                new_sent.append(None)
            else:
                new_sent.append(word)
        new_data.append(new_sent)

    from pprint import pprint
    pprint(new_data)

    return (new_data, withheld_instances)

def withhold_word_by_number(data, banned_words, banned_number, window=4):
    training_sent = []
    testing_sent = []

    for sent in data:
        endings = map(word_to_suffix, sent)
        if str(banned_number) in endings:
            for word in banned_words:
                if word in sent:
                    testing_sent.append(sent)
                    break
            else:
                print('banned number in training set')
                print(sent)
                training_sent.append(sent)
        else:
            training_sent.append(sent)

    # print("TRAINING")
    # for sent in training_sent:
    #     print(sent)

    # print()
    # print("TESTING")
    # for sent in testing_sent:
    #     print(sent)
    return training_sent, testing_sent

def peter_withhold_word_by_number(data, banned_words, banned_number, window=4):
    """Withholds banned words if one or more of the sentence's words is a variant in banned_numbers"""
    new_data = []
    withheld_instances = []
    for sent in data:
        endings = map(word_to_suffix, sent)
        if str(banned_number) in endings:
            new_sent = []
            for idx, word in enumerate(sent):
                if word in banned_words:
                    withheld_instances.append((sent[max(0, idx-window):idx+1+window], min(window, idx)))
                    new_sent.append(None)
                else:
                    new_sent.append(word)
            new_data.append(new_sent)
        else:
            new_data.append(sent)

    return (new_data, withheld_instances)


def filter_stop_words(data):
    for sent in data:
        yield [word for word in sent if word not in stop_words]


def length_filter(data, min_length):
    return (sent for sent in data if len(sent) >= min_length)


def synonymize(synonyms=[], nsynonyms=2, ncopies=2, data=[]):
    for _ in range(ncopies):
        for sent in data:
            for i in range(nsynonyms):
                yield [(word_to_variant(word, i) if word in synonyms else word)
                       for word in sent]


def homonymize(homonyms=[], nhomonyms=2, ncopies=2, data=[]):
    for _ in range(ncopies):
        for sent in data:
            # for i in range(nhomonyms):
            #     for h in homonyms:
            #         yield [(word_to_variant(word, i) if word != h else word)
            #                for word in sent]
            for i in range(nhomonyms):
                yield [(word_to_variant(word, i) if word not in homonyms else word)
                       for word in sent]

def cluster_iter(tree, temp_clusters, heuristic=CU, minsplit=1, maxsplit=100000, mod=True, labels=True):
    """
    This is the core clustering process that splits the tree according to a
    given heuristic.
    """
    if minsplit < 1:
        raise ValueError("minsplit must be >= 1")
    if minsplit > maxsplit:
        raise ValueError("maxsplit must be >= minsplit")
    if isinstance(heuristic, str):
        if heuristic.upper() == 'CU':
            heuristic = CU
        elif heuristic.upper() == 'AIC':
            heuristic = AIC
        elif heuristic.upper() == 'BIC':
            heuristic = BIC
        elif heuristic.upper() == 'AICC':
            heuristic = AICc
        else:
            raise ValueError('unkown heuristic provided as str', heuristic)

    for nth_split in range(1, maxsplit+1):

        cluster_assign = []
        child_cluster_assign = []
        if nth_split >= minsplit:
            clusters = []
            for i, c in enumerate(temp_clusters):
                child = None
                while (c.parent and c.parent.parent):
                    child = c
                    c = c.parent
                if labels:
                    clusters.append("Concept" + str(c.concept_id))
                else:
                    clusters.append(c)
                cluster_assign.append(c)
                child_cluster_assign.append(child)
            yield clusters, heuristic(cluster_assign, temp_clusters)

        split_cus = []

        for i, target in enumerate(set(cluster_assign)):
            if len(target.children) == 0:
                continue
            c_labels = [label if label != target else child_cluster_assign[j]
                        for j, label in enumerate(cluster_assign)]
            split_cus.append((heuristic(c_labels, temp_clusters), i, target))

        split_cus = sorted(split_cus)

        # Exit early, we don't need to re-run the following part for the
        # last time through
        if not split_cus:
            break

        # Split the least cohesive cluster
        tree.root.split(split_cus[0][2])

        nth_split += 1


def get_leaves(concept):

    for child in concept.children:
        if len(child.children) == 0:
            yield child
        else:
            for c in get_leaves(child):
                yield c

def run_synonym_ari(CobwebTree):
    WINDOW_SIZE = 4
    FOLDS = 5
    NSYNONYMS = FOLDS

    text = full_ms_sentences()
    # subset = filter_stop_words(text)
    subset = list(length_filter(filter_stop_words(text), 10))
    random.shuffle(subset)
    subset = subset[:NUM_SENTENCES]
    SYNONYMS = [word for word, _ in top_n(subset, 5)]
    print(SYNONYMS)

    data = list(synonymize(SYNONYMS, NSYNONYMS, ncopies=1,
                           data=subset))
    # for sent in data:
    #     print(sent)

    random.shuffle(data)

    result = []

    tree = CobwebTree(WINDOW_SIZE)
    for sent in tqdm(data):
        tree.ifit_text(sent)

    # visualize(tree)

    synonyms = [word_to_variant(word, fold_num) for word in SYNONYMS for fold_num in range(FOLDS)]

    leaves = [l for l in get_leaves(tree.root) if len(l.av_counts['anchor']) == 1 and list(l.av_counts['anchor'])[0] in synonyms]
    # print('leaves', len(leaves))

    labels = [list(l.av_counts['anchor'])[0].split('-')[0] for l in leaves]
    # print('labels', labels)

    best = float('-inf')
    for i, (clust, _) in enumerate(cluster_iter(tree, leaves, minsplit=1, maxsplit=50, labels=False)):
        pred = [str(c) for c in clust]
        ari = adjusted_rand_score(labels, pred)
        if ari > best:
            best = ari
        # print('split {}: {}'.format(i, ari))
        # print(labels)
        # print(pred)
        # print()
    return best
    print('BEST ARI: {}'.format(best))

def run_homonym_ari(CobwebTree):
    WINDOW_SIZE = 4
    FOLDS = 5
    NHOMONYMS = FOLDS

    text = full_ms_sentences()
    # subset = filter_stop_words(text)
    subset = list(length_filter(filter_stop_words(text), 10))
    random.shuffle(subset)
    subset = subset[:NUM_SENTENCES]
    
    HOMONYMS = [word for word, _ in top_n(subset, 5)]
    data = list(homonymize(HOMONYMS, NHOMONYMS, ncopies=1,
                           data=subset))

    print(HOMONYMS)

    random.shuffle(data)
    result = []

    original_tree = CobwebTree(WINDOW_SIZE)
    for sent in tqdm(data):
        original_tree.ifit_text(sent)

    # tree = copy.deepcopy(original_tree)
    tree = original_tree

    visualize(tree)

    leaves = [l for l in get_leaves(tree.root) if len(l.av_counts['anchor']) == 1 and list(l.av_counts['anchor'])[0] in HOMONYMS]
    # print('leaves', len(leaves))

    # print(leaves[0].av_counts)
    # print(leaves[0].av_counts['context'])
    labels = ["{}-{}".format(list(l.av_counts['anchor'])[0], get_homonym_id(l)) for l in leaves]
    # print('labels', labels)

    best = float('-inf')
    for i, (clust, _) in enumerate(cluster_iter(tree, leaves, minsplit=1, maxsplit=50, labels=False)):
        pred = [str(c) for c in clust]
        ari = adjusted_rand_score(labels, pred)
        print('split {}: {}'.format(i, ari))
        if ari > best:
            best = ari
        # print(labels)
        # print(pred)
        # print()

    print('BEST ARI: {}'.format(best))
    return best


def get_homonym_id(c):
    for e in c.av_counts['context']:
        if isinstance(e, str):
            if '-' in e:
                return e.split('-')[1]
        else:
            anchor = list(e.av_counts['anchor'])[0]
            if '-' in anchor:
                return anchor.split('-')[1]


def run_test_1(CobwebTree):
    WINDOW_SIZE = 4
    FOLDS = 5
    NSYNONYMS = FOLDS
    SYNONYMS = [word for word, _ in top_n(5)]

    text = full_ms_sentences()
    subset = list(length_filter(filter_stop_words(text), 10))
    data = list(homonymize(HOMONYMS, NHOMONYMS, ncopies=1,
                           data=length_filter(filter_stop_words(text), 10)))
    data = list(synonymize(SYNONYMS, NSYNONYMS, ncopies=1,
                           data=subset))

    print(len(text))
    print(len(subset))
    print(len(data))

    random.shuffle(data)
    result = []
    fold_total = 0
    for fold_num in tqdm(range(FOLDS)):
        tree = CobwebTree(WINDOW_SIZE)
        # print('fold %s' % fold_num)
        withheld_synonyms = [word_to_variant(word, fold_num) for word in SYNONYMS]
        fold_data, withheld_instances = withhold(data, withheld_synonyms)
        for sent in fold_data:
            tree.ifit_text(sent)
        total_distance = 0
        for sent in withheld_instances:
            for index, word in enumerate(sent):
                if word not in withheld_synonyms:
                    continue
                # concept = tree.categorize_text(sent)[index]
                concept = tree.categorize_text(sent)[index]
                base = word_to_base(sent[index])
                synonym_nodes = list(chain.from_iterable(
                    tree.word_to_leaf.get(word_to_variant(base, var_num), ())
                    for var_num in range(FOLDS)))
                total_distance += min(distance(concept, node) for node in synonym_nodes)
                result.append(min(distance(concept, node) for node in synonym_nodes))

        # for (instance, index) in withheld_instances:
        #     # print(instance)
        #     concept = tree.categorize_text(instance)[index]
        #     base = word_to_base(instance[index])
        #     synonym_nodes = list(chain.from_iterable(
        #         tree.word_to_leaf.get(word_to_variant(base, var_num), ())
        #         for var_num in range(FOLDS)))
        #     total_distance += min(distance(concept, node) for node in synonym_nodes)
        #     result.append(min(distance(concept, node) for node in synonym_nodes))
        # fold_total += total_distance / len(withheld_instances)
    return result


def run_test_2(CobwebTree):
    WINDOW_SIZE = 4
    FOLDS = 5
    NHOMONYMS = FOLDS
    HOMONYMS = [word for word, _ in top_n(5)]

    text = full_ms_sentences()
    data = list(homonymize(HOMONYMS, NHOMONYMS, ncopies=1,
                           data=length_filter(filter_stop_words(text), 10)))
    random.shuffle(data)
    result = []
    fold_total = 0
    for fold_num in tqdm(range(FOLDS)):
        tree = CobwebTree(WINDOW_SIZE)
        # print('fold %s' % fold_num)
        fold_data, withheld_instances = withhold_word_by_number(data, HOMONYMS, fold_num)
        assert withheld_instances
        if LeafTree == CobwebTree:
            fold_data = tqdm(fold_data)
        for sent in fold_data:
            tree.ifit_text(sent)

        visualize(tree)

        for sent in withheld_instances:
            for index, word in enumerate(sent):
                if word not in HOMONYMS:
                    continue
                concept = tree.categorize_text(sent)[index]
                to_add = min(
                    distance(concept, node)
                    for node in tree.word_to_leaf[sent[index]])

                result.append(to_add)

                print(sent)
                print(concept)
                print(to_add)
                input('beep')


        # visualize(tree)

        # total_distance = 0
        # for (instance, index) in withheld_instances:
        #     # ctx_nodes = tree.categorize_text(instance)
        #     # concept = tree.term_cat(instance, ctx_nodes, index)
        #     concept = tree.categorize_text(instance)[index]

        #     to_add = min(
        #         distance(concept, node)
        #         for node in tree.word_to_leaf[instance[index]])

        #     # print(instance)
        #     # print(concept)
        #     # print(to_add)
        #     # input('press key')

        #     total_distance += to_add
        #     result.append(to_add)
        # fold_total += total_distance / len(withheld_instances)

    return result


def run_test_3(CobwebTree):
    WINDOW_SIZE = 4
    FOLDS = 5
    NSYNONYMS = FOLDS
    SYNONYMS = [word for word, _ in top_n(50)]
    SYNONYMS.remove("upon")

    text = full_ms_sentences()
    data = list(synonymize(SYNONYMS, NSYNONYMS, ncopies=1,
                           data=length_filter(filter_stop_words(text), 10)))
    print(len(text))
    print(len(list(length_filter(filter_stop_words(text), 10))))
    print(len(data))

    random.shuffle(data)
    tree = CobwebTree(WINDOW_SIZE)
    track = {word: [] for word in SYNONYMS}
    for sent in tqdm(data):
        tree.ifit_text(sent, track=track)

    return track


def run(tup):
    s, f, t = tup
    res = f(t)
    print(s, res)
    return res


if __name__ == "__main__":
    # (WordTree, LeafTree, StaticTree, LiteralTree)
    print(run_homonym_ari(StaticTree))
    raise Exception('blerg')

    with multiprocessing.Pool(6) as p:
        list(p.map(run,
              (
                ('Static Cobweb Homo. Test:', run_homonym_ari, StaticTree),
                ('Static Cobweb Homo. Test:', run_homonym_ari, StaticTree),
                ('Static Cobweb Homo. Test:', run_homonym_ari, StaticTree),
                ('Static Cobweb Homo. Test:', run_homonym_ari, StaticTree),
                ('Static Cobweb Homo. Test:', run_homonym_ari, StaticTree),
                ('Static Cobweb Homo. Test:', run_homonym_ari, StaticTree),
               # ('Word Cobweb Syno. Test:', run_test_1, WordTree),
               # # ('Literal Cobweb Syno. Test:', run_test_1, LiteralTree),
               # ('Leaf Cobweb Syno. Test:', run_test_1, LeafTree),
               # ('Static Cobweb Syno. Test:', run_test_1, StaticTree),
               # ('Word Cobweb Homo. Test:', run_test_2, WordTree),
               # ('Literal Cobweb Homo. Test:', run_test_2, LiteralTree),
               # ('Leaf Cobweb Homo. Test:', run_test_2, LeafTree),
               # ('Static Cobweb Homo. Test:', run_test_2, StaticTree),
               # ('Word Cobweb Syno. Prediction Test:', run_test_3, WordTree),
               # ('Literal Cobweb Syno. Prediction Test:', run_test_3, LiteralTree),
               # ('Leaf Cobweb Syno. Prediction Test:', run_test_3, LeafTree),
               # ('Static Cobweb Syno. Prediction Test:', run_test_3, StaticTree)
               )))
