# This is going to be the root file that you would use to work with each of the
# algorithms in this package. I went with tlc because of Trestle, Labyrinth, Cobweb
# but we can rename it later if we want

import json
#from trestle import Trestle
#from cobweb import Cobweb
from cobweb3 import Cobweb3Tree
from itertools import cycle, islice
import copy
import random

random.seed(1)

def sort_dissimilar(instances):
    original = copy.deepcopy(instances)
    for i,d in enumerate(original):
        d['_id'] = i

    data = [a for a in original]
    random.shuffle(data)
    #last_ids = []
    #ids = [a['*id'] for a in data]

    # not sure how to tell that I have converged... there is no likelihood
    # score to maximize or something... do i do cu at the root?
    for i in range(10):
    #while last_ids != ids:
        #print(levenshtein(last_ids, ids))
        #last_ids = ids
        tree = Cobweb3Tree()
        tree.fit(data)
        #print(tree.category_utility())
        ids = [a for a in order(tree.root)]
        data = [original[v] for v in ids]

    return data

def order(node):
    if not node.children:
        return list(node.av_counts['_id'].keys())

    node.children.sort(key=lambda x: x.count, reverse=True)
    items = [order(c) for c in node.children]
    return roundrobin(*items)

def levenshtein(a,b):
    "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a,b = b,a
        n,m = m,n
        
    current = range(n+1)
    for i in range(1,m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1,n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)
            
    return current[n]

def roundrobin(*iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    pending = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)
    while pending:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            pending -= 1
            nexts = cycle(islice(nexts, pending))

def cluster(tree, instances, depth=1):
    """
    Used to cluster examples incrementally and return the cluster labels.
    The final cluster labels are at a depth of 'depth' from the root. This
    defaults to 1, which takes the first split, but it might need to be 2
    or greater in cases where more distinction is needed.
    """
    temp_clusters = [tree.ifit(instance) for instance in instances]

    print(len(set([c.concept_id for c in temp_clusters])))
    clusters = []
    for i,c in enumerate(temp_clusters):
        while (c.parent and c not in c.parent.children):
            c = c.parent

        promote = True
        while c.parent and promote:
            n = c
            for i in range(depth+2):
                if not n:
                    promote = False
                    break
                n = n.parent

            if promote:
                c = c.parent

        clusters.append("Concept" + c.concept_id)

    with open('visualize/output.json', 'w') as f:
        f.write(json.dumps(tree.root.output_json()))

    return clusters

if __name__ == "__main__":
    data = [{'x': random.normalvariate(0,0.5)} for i in range(10)]
    data += [{'x': random.normalvariate(2,0.5)} for i in range(10)]
    data += [{'x': random.normalvariate(4,0.5)} for i in range(10)]
    data = sort_dissimilar(data)

    tree = Cobweb3Tree()
    clusters = cluster(tree, data)
    print(clusters)
    print(set(clusters))
