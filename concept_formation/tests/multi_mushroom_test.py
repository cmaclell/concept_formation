from random import shuffle
from random import seed

from concept_formation.multinomial_cobweb import MultinomialCobwebTree
from concept_formation.datasets import load_mushroom
from concept_formation.visualize import visualize
from pprint import pprint

seed(0)

if __name__ == "__main__":

    mushrooms = load_mushroom()
    shuffle(mushrooms)
    mushrooms = mushrooms[:1500]

    # tree = MultinomialCobwebTree()
    tree = MultinomialCobwebTree(True, # optimize info vs. ec
                                 True, # regularize using concept encoding cost
                                 0.1, # alpha weight
                                 False, # dynamically compute alpha
                                 False, # weight alpha by avg occurance of attr
                                 True, # weight attr by avg occurance of attr
                                 True, # categorize to basic level (true)? or leaf (false)?
                                 False, # use category utility to identify basic (True)? or prob (false)?
                                 False) # predict using mixture at root (true)? or single best (false)?

    mushrooms_no_class_multi = [{a if a != "classification" else "_classification":
                                 {mushroom[a]: 1} for a in mushroom} for mushroom in
                                 mushrooms]
    mushrooms_no_class = [{a: mushroom[a] for a in mushroom
                           if a != 'classification'} for mushroom in mushrooms]

    acc = []
    for m in mushrooms_no_class_multi:
        if tree.root.count > 0:
            leaf = tree.categorize(m)
            v = leaf.predict("classification")

            acc.append(int(v in m['_classification']))
        m['classification'] = m['_classification']
        # print(m)
        tree.ifit(m)
    print(acc)
    print(sum(acc)/len(acc))
    visualize(tree)

