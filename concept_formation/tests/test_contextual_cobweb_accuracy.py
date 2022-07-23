from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division

from concept_formation.contextual_cobweb import ContextualCobwebTree
from concept_formation.contextual_cobweb import ca_key
from cProfile import run
import re
# import random
from time import time


find_context = re.compile(r"'%s': {.*?}" % ca_key)


def ellide_context(string):
    return re.sub(find_context, "'%s': {...}" % ca_key, string)


def word_to_obj(word):
    return {'Anchor': word}


def print_tree(tree, ctxt=True):
    if ctxt:
        print(str(tree).replace('Node', 'N'))
    else:
        print(ellide_context(str(tree)))


def test_words(data, ctxt_size=2, ctxt_weight=4, show_context=False):
    tree = ContextualCobwebTree(ctxt_weight=ctxt_weight)

    nodes = tree.contextual_ifit(
        tuple(map(word_to_obj, data.split(' '))), context_size=ctxt_size)
    print([node.concept_id for node in nodes])
    print('-'*40+' (Final)')
    print_tree(tree, show_context)
    return tree


def test_words_1():
    sens = ("one fish two fish red fish blue fish "
            "black fish blue fish old fish new fish "
            "this one has a little car "
            "this one has a little star "
            "say what a lot of fish there are "
            "yes some are red and some are blue "
            "some are old and some are new "
            "some are sad and some are glad "
            "and some are very very bad "
            "why are they sad and glad and bad "
            "i do not know go ask your dad "
            "some are thin and some are fat "
            "the fat one has a yellow hat "
            "from there to here "
            "from here to there "
            "funny things are everywhere "
            "here are some who like to run "
            "they run for fun in the hot hot sun "
            "oh me oh my oh me oh my "
            "what a lot of funny things go by "
            "some have two feet and some have four "
            "some have six feet and some have more "
            "where do they come from i can't say "
            "but i bet they have come a long long way "
            "we see them come we see them go "
            "some are fast some are slow "
            "some are high some are low "
            "not one of them is like another "
            "don't ask us why go ask your mother")
    test_words(sens, ctxt_size=2, ctxt_weight=4)


def test_words_2():
    sens = ("cat tries to chase dog "
            "dog chases cat in field "
            "cat climbs tree in field "
            "dog barks at tree in field "
            "cat climbs down tree cat scratches dog "
            "dog flees cat and tree "
            "dog meets owner in park "
            "owner pets dog in park "
            "cat chases owner and dog "
            "owner and dog flee cat "
            "owner climbs stairs in house "
            "dog follows owner up stairs "
            "owner looks at tree in field "
            "dog looks at owner in house "
            "bug looks at dog in house "
            "dog and cat are pets "
            "most pets are with owner "
            "bug and owner are not pets "
            "dog scratches itself "
            "most bug are in field "
            "bug in house flees owner "
            "owner follows bug down stairs "
            "owner chases bug into field")

    test_words(sens, ctxt_size=2, ctxt_weight=4)


def test_words_3():
    sens = ('deer ingests leaves in barn '
            'deer nibbles grass in field '
            'horse eats food in pasture '
            'deer ingests hay in field '
            'deer finds grass in pasture '
            'deer finds food in barn '
            'cow finds bark in barn '
            'cow ingests hay in field '
            'cow eats hay in barn '
            'pig eats grass in woods '
            'cow eats hay in barn '
            'animal eats food in pasture '
            'cow ingests leaves in field '
            'duck nibbles grass in barn '
            'duck eats bark in barn '
            'cow nibbles bark in field '
            'cow ingests food in barn '
            'animal eats grass in barn '
            'horse ingests food in pasture '
            'animal eats grass in field')
    test_words(sens, ctxt_size=1, ctxt_weight=4)


def test_small():
    sens = ("cat chases dog "
            "bird sees dog "
            "dog flees cat "
            "dog chases cat "
            "cat flees dog "
            "cat chases bird "
            "bird flees cat")

    test_words(sens, ctxt_size=1, ctxt_weight=3)


def test_words_homonym():
    # See "saw." Needs large (>3) ctxt_weight
    sens = ("carpenter uses saw "
            "carpenter makes cat "
            "cat saw dog "
            "carpenter makes dog "
            "saw and wood "
            "carpenter buys wood "
            "dog saw cat "
            "dog uses saw "
            "cat and dog "
            "carpenter saw saw ")

    tree = test_words(sens, ctxt_size=1, ctxt_weight=5)
    tree.context_weight = 3  # 3 leads to good categorization, 4 to bad
    print()
    tree.contextual_ifit([word_to_obj("wood"), word_to_obj("buys")],
                         context_size=2)
    print_tree(tree)


if __name__ == "__main__":
    start = time()
    # test_small()
    run('test_words_2()')
    print('-'*70)
    print('Finished in %ss' % round(time() - start, 3))
