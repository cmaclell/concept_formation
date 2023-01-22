from random import randint
from timeit import timeit

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from concept_formation.visualize import visualize
from concept_formation.cobweb import CobwebTree


if __name__ == "__main__":

    tree = CobwebTree()

    examples = []
    examples.append({
        '_superord': 'pounder', 
        '_basic': 'hammer', 
        '_subord': 'hammer1',
        '_instance': 'hammer1-1',
        'handle': '1',
        'shaft': '1',
        'head': '1',
        'size': '1'
    })
    examples.append({
        '_superord': 'pounder', 
        '_basic': 'hammer', 
        '_subord': 'hammer1',
        '_instance': 'hammer1-2',
        'handle': '1',
        'shaft': '1',
        'head': '1',
        'size': '2'
    })
    examples.append({
        '_superord': 'pounder', 
        '_basic': 'hammer', 
        '_subord': 'hammer2',
        '_instance': 'hammer2-1',
        'handle': '1',
        'shaft': '1',
        'head': '2',
        'size': '1'
    })
    examples.append({
        '_superord': 'pounder', 
        '_basic': 'hammer', 
        '_subord': 'hammer2',
        '_instance': 'hammer2-2',
        'handle': '1',
        'shaft': '1',
        'head': '2',
        'size': '2'
    })

    examples.append({
        '_superord': 'pounder', 
        '_basic': 'brick', 
        '_subord': 'brick1',
        '_instance': 'brick1-1',
        'handle': '2',
        'shaft': '2',
        'head': '3',
        'size': '1'
    })
    examples.append({
        '_superord': 'pounder', 
        '_basic': 'brick', 
        '_subord': 'brick1',
        '_instance': 'brick1-2',
        'handle': '2',
        'shaft': '2',
        'head': '3',
        'size': '2'
    })
    examples.append({
        '_superord': 'pounder', 
        '_basic': 'brick', 
        '_subord': 'brick2',
        '_instance': 'brick2-1',
        'handle': '3',
        'shaft': '2',
        'head': '3',
        'size': '1'
    })
    examples.append({
        '_superord': 'pounder', 
        '_basic': 'brick', 
        '_subord': 'brick2',
        '_instance': 'brick2-2',
        'handle': '3',
        'shaft': '2',
        'head': '3',
        'size': '2'
    })

    examples.append({
        '_superord': 'cutter', 
        '_basic': 'knife', 
        '_subord': 'knife1',
        '_instance': 'knife1-1',
        'handle': '4',
        'shaft': '3',
        'head': '4',
        'size': '1'
    })
    examples.append({
        '_superord': 'cutter', 
        '_basic': 'knife', 
        '_subord': 'knife1',
        '_instance': 'knife1-2',
        'handle': '4',
        'shaft': '3',
        'head': '4',
        'size': '2'
    })
    examples.append({
        '_superord': 'cutter', 
        '_basic': 'knife', 
        '_subord': 'knife2',
        '_instance': 'knife2-1',
        'handle': '4',
        'shaft': '3',
        'head': '5',
        'size': '1'
    })
    examples.append({
        '_superord': 'cutter', 
        '_basic': 'knife', 
        '_subord': 'knife2',
        '_instance': 'knife2-2',
        'handle': '4',
        'shaft': '3',
        'head': '5',
        'size': '2'
    })

    examples.append({
        '_superord': 'cutter', 
        '_basic': 'pizza cutter', 
        '_subord': 'pizza1',
        '_instance': 'pizza1-1',
        'handle': '5',
        'shaft': '4',
        'head': '6',
        'size': '1'
    })
    examples.append({
        '_superord': 'cutter', 
        '_basic': 'pizza cutter', 
        '_subord': 'pizza1',
        '_instance': 'pizza1-2',
        'handle': '5',
        'shaft': '4',
        'head': '6',
        'size': '2'
    })
    examples.append({
        '_superord': 'cutter', 
        '_basic': 'pizza cutter', 
        '_subord': 'pizza2',
        '_instance': 'pizza2-1',
        'handle': '5',
        'shaft': '5',
        'head': '6',
        'size': '1'
    })
    examples.append({
        '_superord': 'cutter', 
        '_basic': 'pizza cutter', 
        '_subord': 'pizza2',
        '_instance': 'pizza2-2',
        'handle': '5',
        'shaft': '5',
        'head': '6',
        'size': '2'
    })


    tree.fit(examples)
    tree.fit(examples)
    tree.fit(examples)

    visualize(tree)

