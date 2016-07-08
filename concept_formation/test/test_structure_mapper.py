from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

from concept_formation.trestle import TrestleTree
from random import normalvariate
from concept_formation.structure_mapper import StructureMappingProblem
from concept_formation.structure_mapper import get_component_names
from concept_formation.structure_mapper import compute_rewards
from concept_formation.structure_mapper import StructureMapper
from concept_formation.preprocessor import Pipeline
from concept_formation.preprocessor import Tuplizer
from concept_formation.preprocessor import NameStandardizer
from concept_formation.preprocessor import SubComponentProcessor
from concept_formation.preprocessor import Flattener
from py_search.search import compare_searches
from py_search.search import widening_beam_search
from py_search.search import best_first_search
from py_search.search import Problem
from py_search.search import Node
from pprint import pprint
import random

class NoHeuristic(StructureMappingProblem):

    def node_value(self, node):
        return node.cost()

class OptimizationProblem(Problem):

    def node_value(self, node):
        return node.cost()

    def reward(self, mapping, rewards):
        reward = 0
        mapped = frozenset(mapping.keys())
        for iattr in rewards:
            if frozenset(iattr).issubset(mapped):
                bindings = tuple(mapping[o] for o in iattr)
                if bindings in rewards[iattr]:
                    reward += rewards[iattr][bindings]
        return reward

    def successor(self, node):
        mapping, others = node.state
        rewards = node.extra
        mapping = dict(mapping)

        for o1 in mapping:
            for o2 in mapping:
                if o1 == o2:
                    continue
                new_mapping = {a:mapping[a] for a in mapping}
                new_mapping[o1] = mapping[o2]
                new_mapping[o2] = mapping[o1]
                path_cost = -1 * self.reward(new_mapping, rewards)
                yield Node((frozenset(mapping.items()), others), node, 
                           ('swap', o1, o2), path_cost, node.extra)

        for o1 in mapping:
            for o2 in others:
                new_mapping = {a:mapping[a] for a in mapping}
                new_mapping[o1] = o2
                new_others = (others -
                              frozenset([o2])).union(frozenset([mapping[o1]]))
                path_cost = -1 * self.reward(new_mapping, rewards)
                yield Node((frozenset(mapping.items()), new_others), node,
                            ('swap', o1, 'others', o2), path_cost, node.extra)

    def goal_test(self, node):
        min_adjacent = min([n.cost() for n in self.successor(node)])
        print(node.cost(), min_adjacent)
        if node.cost() <= min_adjacent:
            return True
        return False

def random_instance(num_objects=10, num_sub_objects=0, num_attributes=2):
    i = {}
    for o in range(num_objects):
        obj = '?rand_obj' + str(o)
        i[obj] = {}

        for a in range(num_attributes):
            attr = 'a' + str(a)
            i[obj][attr] = normalvariate(a,1)
            #i[obj][attr] = random.choice(['v1', 'v2', 'v3', 'v4'])

        for so in range(num_sub_objects):
            sobj = '?rand_sobj' + str(so)
            i[obj][sobj] = {}

            for a in range(num_attributes):
                attr = 'a' + str(a)
                i[obj][sobj][attr] = normalvariate(a,1)
                #i[obj][attr] = random.choice(['v1', 'v2', 'v3', 'v4'])

    return i

def random_concept(num_instances=3):
    tree = TrestleTree()
    for i in range(num_instances):
        print("Training concept with instance", i+1)
        inst = random_instance()
        pprint(inst)
        tree.ifit(inst)
    return tree.root

instance = random_instance()
#x = {'?w1': {'x': 75, 'y': 25, 'value': '5000'},
#     '?w2': {'x': 19, 'y': 25, 'value': '+'}}
#x = {'?widget58': {'width': 75, 'value': '5000', 'height': 25, 'type': 'input', 'left': 127.21875, 'top': 59}, '?widget59': {'width': 19, 'value': '+', 'height': 25, 'type': 'Label', 'left': 100, 'top': 59}, '?widget55': {'width': 75, 'value': '', 'height': 25, 'type': 'input', 'left': 226.03125, 'top': 59}, '?widget56': {'width': 75, 'value': 'Button', 'height': 25, 'type': 'button', 'left': 115.09375, 'top': 253}, '?widget57': {'width': 75, 'value': '1012', 'height': 25, 'type': 'input', 'left': 17.21875, 'top': 59}, '?widget60': {'width': 10, 'value': '=', 'height': 24, 'type': 'Label', 'left': 209, 'top': 60}}
#instance = pipeline.transform(x)
##pprint(instance)
#t = TrestleTree()
#t.ifit(x)
#concept = t.root
#pprint(t.root.av_counts)
concept = random_concept()
subconcept = concept.children[0]
print(concept.av_counts)
print("EC:")
print(concept.expected_correct_guesses())

pl = Pipeline(Tuplizer(), SubComponentProcessor(), Flattener())
sm = StructureMapper(concept, concept.tree.gensym)

i = sm.transform(pl.transform(subconcept.av_counts))
print("STRUCTURE MAPPED INSTANCE")
print(i)


pipeline = Pipeline(Tuplizer(), NameStandardizer(concept.tree.gensym),
                                 SubComponentProcessor(), Flattener())
ns = NameStandardizer(concept.tree.gensym)

pprint(subconcept.av_counts)

instance = ns.transform(subconcept.av_counts)
pprint(instance)

inames = frozenset(get_component_names(instance))
cnames = frozenset(get_component_names(concept.av_counts, True))

rewards = compute_rewards(cnames, instance, concept)
pprint(rewards)

print(inames)
print(cnames)
problem = StructureMappingProblem((frozenset(), inames, cnames),
                                  extra=rewards)
#noproblem = NoHeuristic((frozenset(), inames, cnames),
#                                  extra=rewards)
#
#values = frozenset(next(permutations(inames.union(cnames), len(inames))))
#remaining = inames.union(cnames) - values
#initial = frozenset(zip(inames, values))
#print(initial)
#op_problem = OptimizationProblem((initial, remaining), extra=rewards)
#
#
#sol = next(best_first_search(problem))
#pprint(sol.path())
#mapping, unnamed, availableNames = sol.state
#print({a:v for a,v in mapping})
#
def beam_1(problem):
    return widening_beam_search(problem, initial_beam_width=1)

def beam_2(problem):
    return widening_beam_search(problem, initial_beam_width=2)

def beam_3(problem):
    return widening_beam_search(problem, initial_beam_width=3)

compare_searches([problem], [beam_1, beam_2, beam_3])#, best_first_search])
