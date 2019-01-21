from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

from random import normalvariate
from pprint import pprint
from random import shuffle
from random import choice

from munkres import Munkres
# from scipy.optimize import linear_sum_assignment

from concept_formation.trestle import TrestleTree
from concept_formation.structure_mapper import StructureMappingOptProblem
from concept_formation.structure_mapper import mapping_cost
from concept_formation.structure_mapper import hungarian_mapping
from concept_formation.structure_mapper import get_component_names
from concept_formation.preprocessor import Pipeline
from concept_formation.preprocessor import Tuplizer
from concept_formation.preprocessor import NameStandardizer
from concept_formation.preprocessor import SubComponentProcessor
from concept_formation.preprocessor import Flattener
from py_search.utils import compare_searches
from py_search.optimization import local_beam_search
from py_search.optimization import hill_climbing
from py_search.optimization import simulated_annealing


def unmapped_mapping(inames):
    mapping = []
    for ot in inames:
        mapping.append((ot, ot))
    return frozenset(mapping)


def random_mapping(inames, cnames):
    available = set(cnames)
    available.add(None)
    base = list(inames.union(cnames))
    shuffle(base)

    mapping = []
    for ot in inames:
        bt = choice(list(available))
        if bt is None:
            mapping.append((ot, ot))
        else:
            mapping.append((ot, bt))
        available.remove(bt)

    return frozenset(mapping)


def random_instance(num_objects=10, num_sub_objects=0, num_attributes=1):
    i = {}
    for o in range(num_objects):
        obj = '?rand_obj' + str(o)
        i[obj] = {}

        for a in range(num_attributes):
            attr = 'a' + str(a)
            i[obj][attr] = normalvariate(a, 1)
            # i[obj][attr] = random.choice(['v1', 'v2', 'v3', 'v4'])

        for so in range(num_sub_objects):
            sobj = '?rand_sobj' + str(so)
            i[obj][sobj] = {}

            for a in range(num_attributes):
                attr = 'a' + str(a)
                i[obj][sobj][attr] = normalvariate(a, 1)
                # i[obj][attr] = random.choice(['v1', 'v2', 'v3', 'v4'])

    return i


def random_concept(num_instances=1, num_objects=10):
    tree = TrestleTree()
    for i in range(num_instances):
        # print("Training concept with instance", i+1)
        inst = random_instance(num_objects)
        # pprint(inst)
        tree.ifit(inst)
    return tree.root


def gen_cost_matrix(inames, cnames, target, base):
    cost_matrix = []
    for o in inames:
        row = []
        for c in cnames:
            nm = {}
            nm[o] = c
            cost = mapping_cost({o: c}, target, base)
            row.append(cost)
            # r = eval_obj_mapping(o, nm, target, base, index, partial=partial)
            # row.append(-r)
        unmapped_cost = mapping_cost({}, target, base)
        for other_o in inames:
            if other_o == o:
                row.append(unmapped_cost)
            else:
                row.append(float('inf'))

        cost_matrix.append(row)
    return cost_matrix


if __name__ == "__main__":

    # import timeit
    # for i in range(1,26,2):
    #    setup = "from __main__ import random_concept\n"
    #    setup += "from __main__ import random_instance\n"
    #    setup += "from __main__ import test\n"
    #    setup += "c = random_concept(1, %i)\n" % i
    #    setup += "i = random_instance(%i)\n" % i
    #
    #    for j in range(10):
    #        print("%i\t%0.3f" % (i, timeit.timeit("test(c,i)", setup=setup,
    #                                         number=10)))

    num_c_inst = 1
    num_objs = 20

    concept = random_concept(num_instances=num_c_inst, num_objects=num_objs)
    instance = random_instance(num_objects=num_objs)

    pl = Pipeline(Tuplizer(), SubComponentProcessor(), Flattener())

    # i = sm.transform(pl.transform(subconcept.av_counts))
    # print("STRUCTURE MAPPED INSTANCE")
    # print(i)

    pipeline = Pipeline(Tuplizer(), NameStandardizer(concept.tree.gensym),
                        SubComponentProcessor(), Flattener())
    # ns = NameStandardizer(concept.tree.gensym)

    # pprint(subconcept.av_counts)

    # instance = ns.transform(subconcept.av_counts)
    instance = pipeline.transform(random_instance(num_objects=num_objs))

    inames = frozenset(get_component_names(instance))
    cnames = frozenset(get_component_names(concept.av_counts, True))

    print("INAMES:")
    print(inames)
    print("CNAMES:")
    print(cnames)

    print("########################")
    print("MUNKRES OPTIMIZATION")
    print("########################")
    targetlist = list(inames)
    baselist = list(cnames) + list(inames)
    cost_matrix = gen_cost_matrix(targetlist, baselist, instance, concept)
    print(len(cost_matrix), len(cost_matrix[0]))

    for row in cost_matrix:
        print("\t".join(["%0.2f" % v for v in row]))

    mun = Munkres()
    s = mun.compute(cost_matrix)
    print(s)
    # s = linear_sum_assignment(cost_matrix)
    # mun_sol = {targetlist[ti]: baselist[s[1][i]] for i, ti in
    # enumerate(s[0])}
    mun_sol = {targetlist[row]: baselist[len(cnames)+row] if col > len(cnames)
               else baselist[col] for row, col in s}
    print("Munkres solution:")
    pprint(mun_sol)
    print("Munkres cost:")

    print(mapping_cost(frozenset(mun_sol.items()), instance, concept))

    # munkres_sol =

    # print("########################")
    # print("TREE SEARCH OPTIMIZATION")
    # print("########################")
    #
    # problem = StructureMappingProblem((frozenset(), inames, cnames),
    #                                  extra=rewards)
    # noproblem = NoHeuristic((frozenset(), inames, cnames),
    #                                  extra=rewards)
    #
    # values = frozenset(next(permutations(inames.union(cnames), len(inames))))
    # remaining = inames.union(cnames) - values
    # initial = frozenset(zip(inames, values))
    # print(initial)
    # op_problem = OptimizationProblem((initial, remaining), extra=rewards)
    #
    #

    # mapping, unnamed, availableNames = sol.state
    # print({a:v for a,v in mapping})
    #
    # def beam_1(problem):
    #    return widening_beam_search(problem, initial_beam_width=1)
    #
    # def beam_2(problem):
    #    return widening_beam_search(problem, initial_beam_width=2)
    #
    # def beam_3(problem):
    #    return widening_beam_search(problem, initial_beam_width=3)
    #
    # compare_searches([problem], [beam_1])#, beam_2, beam_3])#,
    # best_first_search])

    # print("BFS solution")
    # sol = next(best_first_search(problem))
    # pprint(dict(sol.state[0]))

    print()
    print("#########################")
    print("Local search optimization")
    print("#########################")
    print()
    rm = random_mapping(inames, cnames)
    um = unmapped_mapping(inames)
    # gm = greedy_best_mapping(inames, cnames, index, instance, concept)
    gm = hungarian_mapping(inames, cnames, instance, concept)
    # m = frozenset(mun_sol.items())
    # unmapped = inames.union(cnames) - frozenset(dict(m).values())

    # print("Random Mapping:")
    # print(rm)
    # print("Random Unmapped:")
    # runmapped = cnames - frozenset(dict(rm).values())
    # print(runmapped)
    # print("Random Cost:")
    # rc = mapping_cost(rm, instance, concept, index)
    # print(rc)
    # print()
    #
    # print("Unmapped Mapping:")
    # print(um)
    # print("Unmapped Unmapped:")
    # uunmapped = cnames - frozenset(dict(um).values())
    # print(runmapped)
    # print("Unmapped Cost:")
    # uc = mapping_cost(um, instance, concept, index)
    # print(uc)
    # print()

    print("Greedy Best Mapping:")
    pprint(dict(gm))
    print("Greedy Best Unmapped:")
    gunmapped = cnames - frozenset(dict(gm).values())
    print(gunmapped)
    print("Greedy Best Cost:")
    gc = mapping_cost(gm, instance, concept)
    print(gc)

    # op_problem1 = OptimizationProblem((rm, runmapped), initial_cost=rc,
    # extra=rewards)
    # op_problem2 = OptimizationProblem((um, uunmapped), initial_cost=uc,
    # extra=rewards)
    op_problem3 = StructureMappingOptProblem((gm, gunmapped), initial_cost=gc,
                                             extra=(instance, concept))

    def annealing(problem):
        n = (num_objs * num_objs) // 2
        return simulated_annealing(problem, temp_length=n)

    def greedy_annealing(problem):
        n = (num_objs * num_objs) // 2
        return simulated_annealing(problem, initial_temp=0, temp_length=n)

    def hill(problem):
        return hill_climbing(problem)

    def beam1(problem):
        return local_beam_search(problem, beam_width=1)

    compare_searches([
        # op_problem1,
        # op_problem2,
        op_problem3
    ], [
        hill,
        beam1,
        annealing,
        greedy_annealing
    ])

    # s = next(annealing(op_problem))
    # pprint(dict(s.state[0]))
    # print(mapping_cost(s.state[0], rewards))
    # print(s.cost())
