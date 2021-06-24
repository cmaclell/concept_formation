import numpy as np
from scipy.stats import t
from random import random
from random import shuffle
from pprint import pprint

from concept_formation.continuous_value import ContinuousValue
from concept_formation.cobweb3 import Cobweb3Tree
from concept_formation.datasets import load_mushroom
from concept_formation.datasets import load_congressional_voting
from concept_formation.datasets import load_iris

cv_key = "#ContinuousValue#"


class Cluster:

    def __init__(self, parent=None, alpha=0.01):
        self.count = 0
        self.av = {}
        self.alpha = alpha
        self.parent = parent

        self.lam0 = 1
        self.a0 = 3
        self.mu0 = 0
        self.var0 = 1

    def p_F(self, instance):
        return np.prod([self.prob(f, instance[f]) for f in instance])

    def prob(self, attr, val):
        if self.parent:
            if attr not in self.parent.av:
                return 0.0
            num_vals = len(self.parent.av[attr])
        else:
            if attr not in self.av:
                return 0.0
            num_vals = len(self.av[attr])

        if isinstance(val, (float, int)):
            root_cv = self.parent.av.get(attr, {}).get(cv_key,
                                                       ContinuousValue())
            m0 = root_cv.unbiased_mean()
            k0 = 3
            a0 = 1
            b0 = 1

            cv = self.av.get(attr, {}).get(cv_key, ContinuousValue())
            mu = ((k0 * m0 + cv.num * cv.unbiased_mean()) /
                  (k0 + cv.num))
            k = k0 + cv.num
            alpha = a0 + cv.num/2
            beta = (b0 + 1/2 * cv.meanSq + ((cv.num * k0) / (k0 + cv.num)) *
                    (cv.unbiased_mean() - m0)**2 / 2)

            return t.pdf(val, 2 * alpha, mu, (beta * (k + 1)) / (alpha * k))

            # root_cv = self.parent.av.get(attr, {}).get(cv_key,
            #                                            ContinuousValue())
            # mu0 = root_cv.unbiased_mean()
            # var0 = (1.0 + root_cv.num * root_cv.biased_std()**2) / (root_cv.num+1)
            # mu0 = self.mu0
            # var0 = self.var0

            # cv = self.av.get(attr, {}).get(cv_key, ContinuousValue())
            # lam = self.lam0 + cv.num
            # ai = self.a0 + cv.num
            # mu = ((self.lam0 * mu0 + cv.num * cv.unbiased_mean()) /
            #       (self.lam0 + cv.num))
            # vari = ((self.a0 * var0 + (cv.num - 1) * cv.biased_std()**2 +
            #         ((self.lam0 * cv.num) / (self.lam0 + cv.num)) *
            #          (mu0 - cv.unbiased_mean())**2) /
            #         (self.a0 + cv.num))

            # var = ai * vari * (1 + 1/lam) / (ai - 2)

            # return t.pdf(val, ai, mu, np.sqrt(var))

        else:
            return ((self.av.get(attr, {}).get(val, 0.0) + self.alpha) /
                    (self.count + self.alpha * num_vals))

    def update_av(self, instance):
        for attr in instance:
            if attr not in self.av:
                self.av[attr] = {}
            if isinstance(attr, (float, int)):
                if cv_key not in self.av[attr]:
                    self.av[attr][cv_key] = ContinuousValue()
            else:
                if instance[attr] not in self.av[attr]:
                    self.av[attr][instance[attr]] = 0.0

    def insert(self, instance):
        self.count += 1
        for attr in instance:
            if attr not in self.av:
                self.av[attr] = {}
            if isinstance(instance[attr], (float, int)):
                if cv_key not in self.av[attr]:
                    self.av[attr][cv_key] = ContinuousValue()
                self.av[attr][cv_key].update(instance[attr])
            else:
                if instance[attr] not in self.av[attr]:
                    self.av[attr][instance[attr]] = 0.0
                self.av[attr][instance[attr]] += 1

    def __repr__(self):
        return repr(self.av)


class RadicalIncremental:

    def __init__(self, coupling_prob=0.3):
        self.clusters = []
        self.c = coupling_prob
        self.root = Cluster()
        self.new = Cluster(self.root)
        self.count = 0

    def ifit(self, instance):
        self.root.insert(instance)

        insert_scores = sorted([(self.score_insert(c, instance), random(), c)
                                for c in self.clusters], reverse=True)
        new_score = self.score_new(instance)

        print('insert scores')
        print([s for s, _, _ in insert_scores])

        print('new score')
        print(new_score)

        if len(insert_scores) > 0 and insert_scores[0][0] > new_score:
            insert_scores[0][2].insert(instance)
        else:
            new_c = Cluster(parent=self.root)
            new_c.insert(instance)
            self.clusters.append(new_c)

        self.count += 1.0

    def prob(self, instance, attr, val):
        if attr not in self.root.av:
            return 0.0
        if len(self.root.av[attr]) == 0:
            return 0.0

        pred = {}

        for val in self.root.av[attr]:
            pred[val] = self.p_new_given_F(instance) * self.new.prob(attr, val)

            for c in self.clusters:
                pred[val] += self.p_k_given_F(c, instance) * c.prob(attr, val)

        return pred[val]

    def predict(self, instance, attr):
        if attr not in self.root.av:
            return None
        if len(self.root.av[attr]) == 0:
            return None

        pred = {}

        for val in self.root.av[attr]:
            if val == cv_key:
                raise Exception("Ability to predict continuous attributes not"
                                " implemented yet.")
            else:
                pred[val] = self.p_new_given_F(instance) * self.new.prob(attr,
                                                                         val)

                for c in self.clusters:
                    pred[val] += self.p_k_given_F(c, instance) * c.prob(attr,
                                                                        val)

        # print(pred)

        return sorted([(pred[val], random(), val) for val in pred])[-1][2]

    def score_insert(self, clust, instance):
        return self.p_k(clust) * clust.p_F(instance)

    def score_new(self, instance):
        return self.p_new() * self.new.p_F(instance)

    def p_k_given_F(self, clust, instance):
        num = self.score_insert(clust, instance)
        if num == 0.0:
            return 0.0

        return (num /
                sum([self.p_k(c) * c.p_F(instance) for c in self.clusters] +
                    [self.p_new() * self.new.p_F(instance)]))

    def p_new_given_F(self, instance):
        num = self.score_new(instance)

        if num == 0.0:
            return 0.0

        return (num /
                sum([self.p_k(c) * c.p_F(instance) for c in self.clusters] +
                    [self.p_new() * self.new.p_F(instance)]))

    def p_k(self, clust):
        return ((self.c * clust.count) /
                ((1 - self.c) + self.c * self.count))

    def p_new(self):
        return ((1 - self.c) /
                ((1 - self.c) + self.c * self.count))


def replicate_medin_and_shafer():
    m = RadicalIncremental()
    m.root.update_av({'f1': '1', 'f2': '1', 'f3': '1', 'f4': '1', 'f5': '1'})
    m.root.update_av({'f1': '0', 'f2': '0', 'f3': '0', 'f4': '0', 'f5': '0'})

    print(m.root.av)

    inst = {'f1': '1', 'f2': '1', 'f3': '1', 'f4': '1', 'f5': '1'}
    m.ifit(inst)

    print('after 1')
    for cluster in m.clusters:
        pprint(cluster.av)
        print()

    inst = {'f1': '1', 'f2': '0', 'f3': '1', 'f4': '0', 'f5': '1'}
    m.ifit(inst)

    print('after 2')
    for cluster in m.clusters:
        pprint(cluster.av)
        print()

    inst = {'f1': '1', 'f2': '0', 'f3': '1', 'f4': '1', 'f5': '0'}
    m.ifit(inst)

    print('after 3')
    for cluster in m.clusters:
        pprint(cluster.av)
        print()

    inst = {'f1': '0', 'f2': '0', 'f3': '0', 'f4': '0', 'f5': '0'}
    m.ifit(inst)

    print('after 4')
    for cluster in m.clusters:
        pprint(cluster.av)
        print()

    inst = {'f1': '0', 'f2': '1', 'f3': '0', 'f4': '1', 'f5': '1'}
    m.ifit(inst)

    print('after 5')
    for cluster in m.clusters:
        pprint(cluster.av)
        print()

    inst = {'f1': '0', 'f2': '1', 'f3': '0', 'f4': '0', 'f5': '0'}
    m.ifit(inst)

    print('after 6')
    for cluster in m.clusters:
        pprint(cluster.av)
        print()

    print()


if __name__ == "__main__":

    # replicate_medin_and_shafer()

    data = load_iris(num_instances=300)
    target = 'class'
    # data = load_mushroom(num_instances=50)
    # target = 'classification'
    # data = load_congressional_voting(num_instances=50)
    # target = 'Class Name'

    shuffle(data)

    m = RadicalIncremental()

    preds = []
    for d in data[:-1]:
        test = {a: d[a] for a in d if a != target}
        # print('Prob:', m.prob(test, 'classification', d['classification']))
        pred = m.predict(test, target)
        # print('CORRECT?', pred, d[target], pred ==
        #       d['classification'])
        preds.append(int(pred == d[target]))

        m.ifit(d)


    print(preds)
    print('Anderson Avg Accuracy', np.mean(preds))
    print()

    for cluster in m.clusters:
        pprint(cluster.av)
        print()


    tree = Cobweb3Tree()

    preds = []
    for d in data[:-1]:
        test = {a: d[a] for a in d if a != target}
        # print('Prob:', m.prob(test, 'classification', d[target]))
        concept = tree.categorize(test)
        pred = concept.predict(target)

        #  print('CORRECT?', pred, d[target], pred ==
        #        d[target])
        preds.append(int(pred == d[target]))

        tree.ifit(d)

    print(preds)
    print('COBWEB Avg Accuracy', np.mean(preds))
    print()

    # m.predict(test, 'classification')
    # print("ACTUAL:", data[-1]['classification'])
    # print()

    # print("# Clusters =", len(m.clusters))
    # print()

    # for cluster in m.clusters:
    #     pprint(cluster.av)
    #     print()
