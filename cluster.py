import json
import re
import copy

def cluster_iter(tree, instances, minsplit=1, maxsplit=100000,  mod=True):
    """
    Categorize the list of instances into the tree and return an iterator that
    can  be used to iteratively return clusterings based on splitting nodes of
    the tree.

    The first clustering is derived by splitting the root, then each subsequent
    clustering is based on splitting the least coupled cluster (in terms of
    category utility). This process may halt early if it reaches the case where
    there are no more clusters to split (each instance in its own cluster).
    Because splitting is a modifying operation, a deepcopy of the tree is made
    before creating the iterator.

    Keyword arguments:
    tree -- a CobwebTree, Cobweb3Tree, or TrestleTree
    instances -- a list of instance objects
    minsplit -- The minimum number of splits to perform on the tree, must be >=1
    (default 1)
    maxsplit -- the maximum number of splits to perform on the tree, must be >= minsplit
    (default 100000)
    mod -- If True instances will be fit (i.e. modyfiying knoweldge) else they
    will be categorized (i.e. not modiyfing knowledge) (default True)
    """
    if minsplit < 1: 
        raise ValueError("minsplit must be >= 1") 
    if minsplit > maxsplit: 
        raise ValueError("maxsplit must be >= minsplit")

    tree = copy.deepcopy(tree)

    if mod:
        temp_clusters = [tree.ifit(instance) for instance in instances]
    else:
        temp_clusters = [tree.categorize(instance) for instance in instances]
    
    for nth_split in range(1,maxsplit+1):

        if nth_split >= minsplit:
            clusters = []
            for i,c in enumerate(temp_clusters):
                while (c.parent and c.parent.parent):
                    c = c.parent
                clusters.append("Concept" + c.concept_id)
            yield clusters

        split_cus = sorted([(tree.root.cu_for_split(c) -
                             tree.root.category_utility(), i, c) for i,c in
                            enumerate(tree.root.children) if c.children])

        # Exit early, we don't need to re-reun the following part for the
        # last time through
        if not split_cus:
            break

        # Split the least cohesive cluster
        tree.root.split(split_cus[-1][2])

        nth_split+=1

def cluster(tree, instances, minsplit=1, maxsplit=1, mod=True):
    """
    Categorize the list of instances into the tree and return a list of lists of
    flat cluster labelings based on different numbers of splits.

    Keyword arguments:
    tree -- a CobwebTree, Cobweb3Tree, or TrestleTree
    instances -- a list of instance objects
    minsplit -- The minimum number of splits to perform on the tree, must be >=1
    (default 1)
    maxsplit -- the maximum number of splits to perform on the tree, must be >=1
    (default 1)
    mod -- If True instances will be fit (i.e. modyfiying knoweldge) else they
    will be categorized (i.e. not modiyfing knowledge) (default True)
    """
    return [c for c in cluster_iter(tree,instances,minsplit,maxsplit,mod)]

def k_cluster(tree,instances,k=3,mod=True):
    """
    Categorize the list of instances into the tree and return a flat cluster
    where n_clusters <= k. If a split would result in n_clusters > k then fewer
    clusters will be returned.

    Keyword arguments:
    tree -- a CobwebTree, Cobweb3Tree, or TrestleTree
    instances -- a list of instance objects
    k -- a desired number of clusters (default 3)
    mod -- If True instances will be fit (i.e. modyfiying knoweldge) else they
    will be categorized (i.e. not modiyfing knowledge) (default True)
    """

    if k < 2:
        raise ValueError("k must be >=2, all nodes in Cobweb are guaranteed to have at least 2 children.")

    clustering = ["Concept" + tree.root.concept_id for i in instances]
    for c in cluster_iter(tree, instances,mod=mod):
        if len(set(c)) > k:
            break
        clustering = c

    return clustering

def generate_d3_visualization(tree, fileName):
    """
    Export a .js file that is used to visualize the tree with d3.
    """
    fname = 'visualize/'+fileName+'.js'
    with open(fname, 'w') as f:
        f.write("var output = '"+re.sub("'", '',
                                        json.dumps(tree.root.output_json()))+"';")

def depth_labels(tree,instances,mod=True):
    """
    Categorize the list of instances into the tree and return a matrix of
    labeling of each instance based on different depth cuts of the tree.

    The returned matrix is max(conceptDepth) X len(instances). Labelings are
    ordered general to specific with final_labels[0] being the root and
    final_labels[-1] being the leaves.

    Keyword attributes:
    tree -- a CobwebTree, Cobweb3Tree, or TrestleTree
    instances -- a list of instance objects
    mod -- If True instances will be fit (i.e. modyfiying knoweldge) else they
    will be categorized (i.e. not modiyfing knowledge) (default True)
    """
    if mod:
        temp_labels = [tree.ifit(instance) for instance in instances]
    else:
        temp_labels = [tree.categorize(instance) for instance in instances]

    instance_labels = []
    max_depth = 0
    for t in temp_labels:
        labs = []
        depth = 0
        label = t
        while label.parent:
            labs.append("Concept" + label.concept_id)
            depth += 1
            label = label.parent
        labs.append("Concept" + label.concept_id)
        depth += 1
        instance_labels.append(labs)
        if depth > max_depth:
            max_depth = depth

    for f in instance_labels:
        f.reverse()
        last_label = f[-1]
        while len(f) < max_depth:
            f.append(last_label)

    final_labels = []
    for d in range(len(instance_labels[0])):
        depth_n = []
        for i in instance_labels:
            depth_n.append(i[d])
        final_labels.append(depth_n)

    return final_labels

#from trestle import TrestleTree
#import pprint
#
#with open("data_files/rb_s_07_continuous.json") as dat:
#    instances = json.load(dat)[:15]
#
#    tree = TrestleTree()
#
#    clus = cluster(tree,instances,maxsplit=5)
#    pprint.pprint(clus)
