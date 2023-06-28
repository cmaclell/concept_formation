from random import shuffle
from random import seed
from pprint import pprint
import time

from tqdm import tqdm

from concept_formation.multinomial_cobweb import MultinomialCobwebTree
from concept_formation.datasets import load_mushroom
from concept_formation.visualize import visualize

seed(0)

if __name__ == "__main__":

    mushrooms = load_mushroom()
    shuffle(mushrooms)
    mushrooms = mushrooms

    tree1 = MultinomialCobwebTree(True, # Use mutual information (rather than expected correct guesses)
                                 0.1, # alpha weight
                                 False, # dynamically compute alpha
                                 False) # weight attr by avg occurance of attr

    tree2 = MultinomialCobwebTree(True, # Use mutual information (rather than expected correct guesses)
                                 0.1, # alpha weight
                                 False, # dynamically compute alpha
                                 False) # weight attr by avg occurance of attr

    mushrooms_multi = [{a: {mushroom[a]: 1} for a in mushroom} for mushroom in mushrooms]
    mushrooms_no_class_multi = [{a if a != "classification" else "_classification":
                                 {mushroom[a]: 1} for a in mushroom} for mushroom in
                                 mushrooms]

    # print("Starting sync")
    # start1 = time.perf_counter()
    # fut1 = [tree1.ifit(m) for m in tqdm(mushrooms_multi)]
    # results1 = [f.wait() for f in tqdm(fut1)]
    # end1 = time.perf_counter()
    # # print("AV key wait time: {}".format(tree1.av_key_wait_time))
    # # print("Write wait time: {}".format(tree1.write_wait_time))
    # # print("Tree write count: {}".format(tree1.write_count))
    # # print("Root write wait time: {}".format(tree1.root.write_wait_time))
    # # print("Root write count: {}".format(tree1.root.write_count))
    # print("Done in {}".format(end1 - start1))

    # visualize(tree1)

    print("Starting async")
    start2 = time.perf_counter()
    # fut2 = [tree2.async_ifit(m) for m in tqdm(mushrooms_multi[:3000])]
    fut2 = [tree2.async_ifit(m) for m in tqdm(mushrooms_multi[:2000])]
    results2 = [f.wait() for f in tqdm(fut2)]
    end2 = time.perf_counter()
    # print("AV key wait time: {}".format(tree2.av_key_wait_time))
    # print("Tree write wait time: {}".format(tree2.write_wait_time))
    # print("Tree write count: {}".format(tree2.write_count))
    # print("Root write wait time: {}".format(tree2.root.write_wait_time))
    # print("Root write count: {}".format(tree2.root.write_count))
    print("Done in {}".format(end2 - start2))

    visualize(tree2)

    acc = []
    for m in tqdm(mushrooms_no_class_multi[3000:5000]):
        if tree2.root.count > 0:
            # leaf = tree.categorize(m, get_best_concept=False)
            # v = leaf.predict("classification")
            # v = leaf.get_basic_level().predict("classification")
            cf = tree2.categorize(m, get_best_concept=False)
            leaf_p = cf.predict()['classification']
            basic_p = cf.predict_basic()['classification']
            v = sorted([(basic_p[val], val) for val in basic_p])[-1][1]
            # v = best_concept.predict("classification")
            if(leaf_p['poisonous'] != basic_p['poisonous']):
                print()
                print("Leaf: {}".format(p))
                print("Basic: {}".format(p))

            acc.append(int(v in m['_classification']))
        m['classification'] = m['_classification']
        # print(m)
        # tree.ifit(m)
    print(acc)
    print(sum(acc)/len(acc))
    # visualize(tree2)

