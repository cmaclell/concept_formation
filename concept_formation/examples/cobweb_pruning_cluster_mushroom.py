import numpy as np
from random import seed
import time

from concept_formation.cobweb import CobwebTree
from concept_formation.datasets import load_mushroom
from concept_formation.visualize import visualize

seed(0)
num_runs = 30
num_examples = 20
mushrooms = load_mushroom()[:num_examples]

#print(mushrooms)
alphas = [0.001, 0.01, 0.1, 1, 10, 100]

for a in alphas:
    print(F"STARTING RUN WITH {num_examples} EXAMPLES AND ALPHA: {a}")
    start = time.time()
    tree = CobwebTree(alpha=a)

    for i, m in enumerate(mushrooms):
        tree.ifit(m)

    print(f"NUMBER OF CONCEPTS: {tree.root.num_concepts()}")
    visualize(tree)
    print(f"TIME TO RUN: {round((time.time() - start) / 60, 2)} mins")
    time.sleep(5)
