from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division

from concept_formation.contextual_cobweb import ContextualCobwebTree
# from cProfile import run
# import random
from time import time


def word_to_obj(word):
    return {'Anchor': word}


def test_words(data, ctxt_size=2, ctxt_weight=4, show_intermediates=True):
    tree = ContextualCobwebTree(ctxt_weight=ctxt_weight)

    for i, sent in enumerate(data):
        nodes = tree.contextual_ifit(
            tuple(map(word_to_obj, sent.split(' '))), ctxt_size)
        print([node.concept_id for node in nodes])
        if show_intermediates:
            print('-'*40+' (Iteration %s, sentence: "%s")' % (i, sent))
            print(str(tree).replace('Node', 'N'))
    print('-'*40+' (Final)')
    print(str(tree).replace('Node', 'N'))


def test_words_1():
    sens = ("one fish two fish red fish blue fish",
            "black fish blue fish old fish new fish",
            "this one has a little car",
            "this one has a little star",
            "say what a lot of fish there are",
            "yes some are red and some are blue",
            "some are old and some are new",
            "some are sad and some are glad",
            "and some are very very bad",
            "why are they sad and glad and bad",
            "i do not know go ask your dad",
            "some are thin and some are fat",
            "the fat one has a yellow hat",
            "from there to here",
            "from here to there",
            "funny things are everywhere",
            "here are some who like to run",
            "they run for fun in the hot hot sun",
            "oh me oh my oh me oh my",
            "what a lot of funny things go by",
            "some have two feet and some have four",
            "some have six feet and some have more",
            "where do they come from i can't say",
            "but i bet they have come a long long way",
            "we see them come we see them go",
            "some are fast some are slow",
            "some are high some are low",
            "not one of them is like another",
            "don't ask us why go ask your mother")
    test_words(sens, ctxt_size=2, ctxt_weight=4, show_intermediates=False)


def test_words_2():
    sens = ("cat tries to chase dog",
            "dog chases cat in field",
            "cat climbs tree in field",
            "dog barks at tree in field",
            "cat climbs down tree cat scratches dog",
            "dog flees cat and tree",
            "dog meets owner in park",
            "owner pets dog in park",
            "cat chases owner and dog",
            "owner and dog flee cat",
            "owner climbs stairs in house",
            "dog follows owner up stairs",
            "owner looks at tree in field",
            "dog looks at owner in house",
            "bug looks at dog in house",
            "dog and cat are pets",
            "most pets are with owner",
            "bug and owner are not pets",
            "dog scratches itself",
            "most bug are in field",
            "bug in house flees owner",
            "owner follows bug down stairs",
            "owner chases bug into field")

    test_words(sens, ctxt_size=2, ctxt_weight=4, show_intermediates=False)


def test_words_3():
    sens = ('deer ingests leaves in barn',
            'deer nibbles grass in field',
            'horse eats food in pasture',
            'deer ingests hay in field',
            'deer finds grass in pasture',
            'deer finds food in barn',
            'cow finds bark in barn',
            'cow ingests hay in field',
            'cow eats hay in barn',
            'pig eats grass in woods',
            'cow eats hay in barn',
            'animal eats food in pasture',
            'cow ingests leaves in field',
            'duck nibbles grass in barn',
            'duck eats bark in barn',
            'cow nibbles bark in field',
            'cow ingests food in barn',
            'animal eats grass in barn',
            'horse ingests food in pasture',
            'animal eats grass in field')
    test_words(sens, ctxt_size=1, ctxt_weight=4)


def test_small():
    sens = ("cat chases dog",
            "bird sees dog",
            "dog flees cat",
            "dog chases cat",
            "cat flees dog",
            "cat chases bird",
            "bird flees cat")

    test_words(sens, ctxt_size=1, ctxt_weight=2)


def test_words_homonym():
    # See "saw." Needs large (>3) ctxt_weight
    sens = ("carpenter uses saw",
            "carpenter makes cat",
            "cat saw dog",
            "carpenter makes dog",
            "saw and wood",
            "carpenter buys wood",
            "dog saw cat",
            "dog uses saw",
            "cat and dog",
            "carpenter saw saw")

    test_words(sens, ctxt_size=1, ctxt_weight=4)


if __name__ == "__main__":
    start = time()
    test_words_homonym()
    print('-'*70)
    print('Finished in %ss' % round(time() - start, 3))
