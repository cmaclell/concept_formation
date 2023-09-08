from random import shuffle
from random import seed
# from pprint import pprint
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

    tree1 = MultinomialCobwebTree(
        True,  # Use mutual information (rather than expected correct guesses)
        0.1,  # alpha weight
        False,  # dynamically compute alpha
        True,  # weight attr by avg occurance of attr
    )

    tree2 = MultinomialCobwebTree(
        True,  # Use mutual information (rather than expected correct guesses)
        0.1,  # alpha weight
        False,  # dynamically compute alpha
        True,  # weight attr by avg occurance of attr
    )

    mushrooms_multi = [{a: {mushroom[a]: 1 if a != "classification" else 1} for a in mushroom} for mushroom in mushrooms]
    mushrooms_no_class_multi = [
        {
            a if a != "classification" else "_classification":
            {mushroom[a]: 1}
            for a in mushroom
        } for mushroom in mushrooms
    ]

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
    fut2 = [tree2.async_ifit(m) for m in tqdm(mushrooms_multi[:300])]
    results2 = [f.wait() for f in tqdm(fut2)]
    end2 = time.perf_counter()
    # print("AV key wait time: {}".format(tree2.av_key_wait_time))
    # print("Tree write wait time: {}".format(tree2.write_wait_time))
    # print("Tree write count: {}".format(tree2.write_count))
    # print("Root write wait time: {}".format(tree2.root.write_wait_time))
    # print("Root write count: {}".format(tree2.root.write_count))
    print("Done in {}".format(end2 - start2))

    visualize(tree2)

    leaf_acc = []
    basic_acc = []
    best_acc = []
    for m in tqdm(mushrooms_no_class_multi[5000:6000]):
        if tree2.root.count > 0:
            # leaf = tree.categorize(m, get_best_concept=False)
            # v = leaf.predict("classification")
            # v = leaf.get_basic_level().predict("classification")
            cf = tree2.categorize(m)
            leaf_p = cf.predict()['classification']
            leaf_v = sorted([(leaf_p[val], val) for val in leaf_p])[-1][1]
            basic_p = cf.predict_basic()['classification']
            basic_v = sorted([(basic_p[val], val) for val in basic_p])[-1][1]
            best_p = cf.predict_best(m)['classification']
            best_v = sorted([(best_p[val], val) for val in basic_p])[-1][1]
            # v = best_concept.predict("classification")
            if (leaf_v != basic_v or basic_v != best_v):
                print()
                print("Leaf: {}".format(leaf_p))
                print("Basic: {}".format(basic_p))
                print("Best: {}".format(best_p))

            leaf_acc.append(int(leaf_v in m['_classification']))
            basic_acc.append(int(basic_v in m['_classification']))
            best_acc.append(int(best_v in m['_classification']))

        m['classification'] = m['_classification']
        # print(m)
        # tree.ifit(m)
    # print(acc)
    print("leaf acc: ", sum(leaf_acc)/len(leaf_acc))
    print("basic acc: ", sum(basic_acc)/len(basic_acc))
    print("best acc: ", sum(best_acc)/len(best_acc))
    # visualize(tree2)

