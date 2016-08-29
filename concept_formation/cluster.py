"""
The cluster model contains functions computing clustering using CobwebTrees and
their derivatives.
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
import copy
from math import log

from concept_formation.continuous_value import ContinuousValue



def cluster_iter(tree, instances, minsplit=1, maxsplit=100000,  mod=True,labels=True):
    """
    Categorize a list of instances into a tree and return an iterator over
    cluster labelings generated from successive splits of the tree.

    An inital clustering is derived by splitting the root node, then each
    subsequent clustering is based on splitting the least coupled cluster (in
    terms of category utility). The first clustering returned by the iterator is
    after a minsplit number of splits and the last one is after a maxsplit
    number of splits. This process may halt early if it reaches the case where
    there are no more clusters to split (each instance in its own cluster).
    Because splitting is a modifying operation, a deepcopy of the tree is made
    before creating the iterator.

    :param tree: A category tree to be used to generate clusters, it can be
        pre-trained or newly created.
    :param instances: A list of instances to cluster
    :param minsplit: The minimum number of splits to perform on the tree
    :param maxsplit: the maximum number of splits to perform on the tree
    :param mod: A flag to determine if instances will be fit (i.e. modifying
        knoweldge) or categorized (i.e. not modifiying knowledge)
    :param labels: If True, returns just the concept labels otherwise returns
        the concepts themselves.
    :type tree: :class:`CobwebTree <concept_formation.cobweb.CobwebTree>`,
        :class:`Cobweb3Tree <concept_formation.cobweb3.Cobweb3Tree>`, or
        :class:`TrestleTree <concept_formation.trestle.TrestleTree>`
    :type instances: [:ref:`Instance<instance-rep>`, :ref:`Instance<instance-rep>`, ...]
    :type minsplit: int
    :type maxsplit: int
    :type mod: bool
    :type labels: bool
    :returns: an iterator of clusterings based on a number of splits between minsplit and maxsplit
    :rtype: iterator

    .. warning:: minsplit must be >=1 and maxsplit must be >= minsplit
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
                if labels:
                    clusters.append("Concept" + c.concept_id)
                else:
                    clusters.append(c)
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
    Categorize a list of instances into a tree and return a list of
    flat cluster labelings based on successive splits of the tree.

    :param tree: A category tree to be used to generate clusters.
    :param instances: A list of instances to cluster
    :param minsplit: The minimum number of splits to perform on the tree
    :param maxsplit: the maximum number of splits to perform on the tree
    :param mod: A flag to determine if instances will be fit (i.e. modifying
        knoweldge) or categorized (i.e. not modifiying knowledge)
    :type tree: :class:`CobwebTree <concept_formation.cobweb.CobwebTree>`,
        :class:`Cobweb3Tree <concept_formation.cobweb3.Cobweb3Tree>`, or
        :class:`TrestleTree <concept_formation.trestle.TrestleTree>`
    :type instances: [:ref:`Instance<instance-rep>`, :ref:`Instance<instance-rep>`, ...]
    :type minsplit: int
    :type maxsplit: int
    :type mod: bool
    :returns: a list of lists of cluster labels based on successive splits between minsplit and maxsplit.
    :rtype: [[minsplit clustering], [minsplit+1 clustering], ... [maxsplit clustering]]

    .. seealso:: :meth:`cluster_iter`
    
    """
    return [c for c in cluster_iter(tree,instances,minsplit,maxsplit,mod)]

def k_cluster(tree,instances,k=3,mod=True):
    """
    Categorize a list of instances into a tree and return a flat cluster
    where ``len(set(clustering)) <= k``. 

    Clusterings are generated by successively splitting the tree until a split
    results in a clustering with > k clusters at which point the clustering just
    before that split is returned. It is possible for this process to return a
    clustering with < k clusters but not > k clusters.

    :param tree: A category tree to be used to generate clusters, it can be
        pre-trained or newly created.
    :param instances: A list of instances to cluster
    :param k: A desired number of clusters to generate
    :param mod: A flag to determine if instances will be fit (i.e. modifying
        knoweldge) or categorized (i.e. not modifiying knowledge)
    :type tree: :class:`CobwebTree <concept_formation.cobweb.CobwebTree>`,
        :class:`Cobweb3Tree <concept_formation.cobweb3.Cobweb3Tree>`, or
        :class:`TrestleTree <concept_formation.trestle.TrestleTree>`
    :type instances: [:ref:`Instance<instance-rep>`, :ref:`Instance<instance-rep>`, ...]
    :type k: int
    :type mod: bool
    :returns: a flat cluster labeling
    :rtype: [label1, label2, ...]

    .. seealso:: :meth:`cluster_iter`
    .. warning:: k must be >= 2.
    """

    if k < 2:
        raise ValueError("k must be >=2, all nodes in Cobweb are guaranteed to have at least 2 children.")

    clustering = ["Concept" + tree.root.concept_id for i in instances]
    for c in cluster_iter(tree, instances,mod=mod):
        if len(set(c)) > k:
            break
        clustering = c

    return clustering

def depth_labels(tree,instances,mod=True):
    """
    Categorize a list of instances into a tree and return a list of lists of
    labelings for  each instance based on different depth cuts of the tree.

    The returned matrix is max(conceptDepth) X len(instances). Labelings are
    ordered general to specific with final_labels[0] being the root and
    final_labels[-1] being the leaves.

    :param tree: A category tree to be used to generate clusters, it can be
        pre-trained or newly created.
    :param instances: A list of instances to cluster
    :param mod: A flag to determine if instances will be fit (i.e. modifying
        knoweldge) or categorized (i.e. not modifiying knowledge)
    :type tree: :class:`CobwebTree <concept_formation.cobweb.CobwebTree>`,
        :class:`Cobweb3Tree <concept_formation.cobweb3.Cobweb3Tree>`, or
        :class:`TrestleTree <concept_formation.trestle.TrestleTree>`
    :type instances: [:ref:`Instance<instance-rep>`, :ref:`Instance<instance-rep>`, ...]
    :type mod: bool
    :returns: a list of lists of cluster labels based on each depth cut of the tree
    :rtype: [[root labeling], [depth1 labeling], ... [maxdepth labeling]]
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

def AICc(clusters, leaves):
    """
    Calculates the Akaike Information Criterion of the a given clustering
    from a given tree and set of instances with a correction for finite sample
    sizes.

    This can be used as one of the heursitic functions in
    :meth:`cluster_split_search`.
    
    .. math :: 
        AICc = 2k - 2\\ln (\\mathcal{L}) + \\frac{2k(k+1)}{n-k-1}
    
    * :math:`\\ln(\\mathcal{L})` is the total log-likelihood of the cluster concepts
    * :math:`k` is the total number of unique attribute value pairs in
      the tree root times the number of clusters
    * :math:`n` is the number of instances.

    Given the particular math of AICc I decided that is n-k-1 = 0 then the function will return float('inf')

    :param clusters: A unique set of cluster concepts from the tree
    :type clusters: {:class:`CobwebNode<concept_formation.cobweb.CobwebNode>`,
        :class:`CobwebNode<concept_formation.cobweb.CobwebNode>`, ...}
    :param tree: The tree that the clusters come from (used to calculated number of parameters)
    :type tree: :class:`CobwebTree <concept_formation.cobweb.CobwebTree>`,
        :class:`Cobweb3Tree <concept_formation.cobweb3.Cobweb3Tree>`, or
        :class:`TrestleTree <concept_formation.trestle.TrestleTree>`
    :param instances: The set of clustered instances
    :type instances: [:ref:`Instance<instance-rep>`, :ref:`Instance<instance-rep>`, ...]
    :returns: The AIC of the clustering
    :rtype: float
    """
    ll = 0
    n = len(leaves)
    # c = len(clusters)
    # r = 0
    k=0

    for i, conc in enumerate(clusters):
        ll += conc.log_likelihood(leaves[i])

    for conc in set(clusters):
        for attr in conc.tree.root.av_counts:
            if attr[0] == '_':
                continue
            if isinstance(conc.tree.root.av_counts[attr],ContinuousValue):
                k += 3
                # r += 3
            else:
                k += len(conc.tree.root.av_counts[attr])

    # for conc in clusters:
    #     ll += conc.log_likelihood()
    #     for attr in conc.av_counts:
    #         if attr[0] == '_':
    #             continue
    #         if isinstance(conc.av_counts[attr],ContinuousValue):
    #             k += 3
    #             # r += 3
    #         else:
    #             k += len(conc.av_counts[attr]) + 1
                # r += len(conc.av_counts[attr]) + 1

    # for attr in tree.root.av_counts:
    #     if attr[0] == '_':
    #         continue
    #     if isinstance(tree.root.av_counts[attr],ContinuousValue):
    #         r += 3
    #     else:
    #         r += len(tree.root.av_counts[attr]) + 1
    # k = r * c
    if n <= k - 1:
        return float('inf')
    else:
        return 2 * k - 2 * ll + 2 * k * (k + 1)/(n - k - 1)

def AIC(clusters, leaves):
    """
    Calculates the Akaike Information Criterion of the a given clustering
    from a given tree and set of instances.

    This can be used as one of the heursitic functions in
    :meth:`cluster_split_search`.
    
    .. math :: 
        AIC = 2k - 2\\ln (\\mathcal{L})
    
    * :math:`\\ln(\\mathcal{L})` is the total log-likelihood of the cluster concepts
    * :math:`k` is the total number of unique attribute value pairs in
      the tree root times the number of clusters 

    :param clusters: A unique set of cluster concepts from the tree
    :type clusters: {:class:`CobwebNode<concept_formation.cobweb.CobwebNode>`,
        :class:`CobwebNode<concept_formation.cobweb.CobwebNode>`, ...}
    :param tree: The tree that the clusters come from (used to calculated number of parameters)
    :type tree: :class:`CobwebTree <concept_formation.cobweb.CobwebTree>`,
        :class:`Cobweb3Tree <concept_formation.cobweb3.Cobweb3Tree>`, or
        :class:`TrestleTree <concept_formation.trestle.TrestleTree>`
    :param instances: The set of clustered instances
    :type instances: [:ref:`Instance<instance-rep>`, :ref:`Instance<instance-rep>`, ...]
    :returns: The AIC of the clustering
    :rtype: float
    """
    ll = 0
    # r = 0
    # c = len(clusters)
    k=0

    for i, conc in enumerate(clusters):
        ll += conc.log_likelihood(leaves[i])

    for conc in set(clusters):
        for attr in conc.tree.root.av_counts:
            if attr[0] == '_':
                continue
            if isinstance(conc.tree.root.av_counts[attr],ContinuousValue):
                k += 3
                # r += 3
            else:
                k += len(conc.tree.root.av_counts[attr])

    # for conc in clusters:
    #     ll += conc.log_likelihood()
    #     for attr in conc.av_counts:
    #         if attr[0] == '_':
    #             continue
    #         if isinstance(conc.av_counts[attr],ContinuousValue):
    #             k += 3
    #             # r += 3
    #         else:
    #             k += len(conc.av_counts[attr]) + 1
                # r += len(conc.av_counts[attr]) + 1
    
    # for attr in tree.root.av_counts:
    #     if attr[0] == '_':
    #         continue
    #     if isinstance(tree.root.av_counts[attr],ContinuousValue):
    #         r += 3
    #     else:
    #         r += len(tree.root.av_counts[attr]) + 1
    # k = r * c
    return 2 * k - 2 * ll

def BIC(clusters, leaves):
    """
    Calculates the Bayesian Information Criterion of the a given clustering
    from a given tree and set of instances.

    This can be used as one of the heursitic functions in
    :meth:`cluster_split_search`.
    
    .. math :: 
        BIC = k\\ln (n) - 2\ln (\\mathcal{L})
    
    * :math:`\\ln(\\mathcal{L})` is the total log-likelihood of the cluster concepts
    * :math:`k` is the total number of unique attribute value pairs in
      the tree root times the number of clusters 
    * :math:`n` is the number of instances.

    :param clusters: A unique set of cluster concepts from the tree
    :type clusters: {:class:`CobwebNode<concept_formation.cobweb.CobwebNode>`,
        :class:`CobwebNode<concept_formation.cobweb.CobwebNode>`, ...}
    :param tree: The tree that the clusters come from (used to calculated number of parameters)
    :type tree: :class:`CobwebTree <concept_formation.cobweb.CobwebTree>`,
        :class:`Cobweb3Tree <concept_formation.cobweb3.Cobweb3Tree>`, or
        :class:`TrestleTree <concept_formation.trestle.TrestleTree>`
    :param instances: The set of clustered instances
    :type instances: [:ref:`Instance<instance-rep>`, :ref:`Instance<instance-rep>`, ...]
    :returns: The BIC of the clustering
    :rtype: float
    """
    # c = len(clusters)
    n = len(leaves)
    ll = 0
    # r = 0
    k = 0 
    
    for i, conc in enumerate(clusters):
        ll += conc.log_likelihood(leaves[i])

    for conc in set(clusters):
        for attr in conc.tree.root.av_counts:
            if attr[0] == '_':
                continue
            if isinstance(conc.tree.root.av_counts[attr],ContinuousValue):
                k += 3
                # r += 3
            else:
                k += len(conc.tree.root.av_counts[attr])

    # for conc in clusters:
    #     ll += conc.log_likelihood()
    #     for attr in conc.av_counts:
    #         if attr[0] == '_':
    #             continue
    #         if isinstance(conc.av_counts[attr],ContinuousValue):
    #             k += 3
    #             # r += 3
    #         else:
    #             k += len(conc.av_counts[attr]) + 1
                # r += len(conc.av_counts[attr]) + 1
    # r = 0
    # for attr in tree.root.av_counts:
    #     if attr[0] == '_':
    #         continue
    #     if isinstance(tree.root.av_counts[attr],ContinuousValue):
    #         r += 3
    #     else:
    #         r += len(tree.root.av_counts[attr]) + 1
    # k = r * c
    return -2 * ll + k * log(n)

def cluster_split_search(tree, instances, heuristic=BIC, minsplit=1, maxsplit=1, mod=True,labels=True,verbose=False):
    """
    Find a clustering of the instances given the tree that is based on
    successive splittings of the tree in order to minimize some heuristic
    function.

    :param tree: A category tree to be used to generate clusters.
    :param instances: A list of instances to cluster
    :param heuristic: A heursitic function to minimize in search
    :param minsplit: The minimum number of splits to perform on the tree
    :param maxsplit: the maximum number of splits to perform on the tree
    :param mod: A flag to determine if instances will be fit (i.e. modifying
        knoweldge) or categorized (i.e. not modifiying knowledge)
    :param verbose: If True, the process will print the heursitic at each
        split as it searches.
    :type tree: :class:`CobwebTree <concept_formation.cobweb.CobwebTree>`,
        :class:`Cobweb3Tree <concept_formation.cobweb3.Cobweb3Tree>`, or
        :class:`TrestleTree <concept_formation.trestle.TrestleTree>`
    :type instances: [:ref:`Instance<instance-rep>`, :ref:`Instance<instance-rep>`, ...]
    :type heuristic: a function.
    :type minsplit: int
    :type maxsplit: int
    :type mod: bool
    :type verbose: bool
    :returns: a list of cluster labels based on the optimal number of splits according to the heuristic
    :rtype: list

    .. seealso:: :meth:`cluster_iter`
    """
    clus_it = cluster_iter_experimental(tree,instances,heuristic=heuristic,minsplit=minsplit,maxsplit=maxsplit,mod=mod,labels=False)
    min_h = (-1,float('inf'),[])
    split = minsplit
    if verbose:
        print('S\tC\tH')
    for split_clus, h in clus_it:
        #clus = set(split_clus)
        #h = heuristic(clus,tree,instances)
        if verbose:
            print(split,len(set(split_clus)), '%.3f'%h,sep='\t')
        if h < min_h[1]:
            min_h = (split,h,["Concept" + c.concept_id for c in split_clus] if labels else split_clus)
        split += 1
    return min_h[2]


def cluster_iter_experimental(tree, instances, heuristic=BIC, minsplit=1, maxsplit=100000,  mod=True,labels=True):
    """
    This is an experimetnal implementation of the cluster iter that uses
    something other than CU to choose what to split
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

        cluster_assign = []
        child_cluster_assign = []
        if nth_split >= minsplit:
            clusters = []
            for i,c in enumerate(temp_clusters):
                child = None
                while (c.parent and c.parent.parent):
                    child = c
                    c = c.parent
                if labels:
                    clusters.append("Concept" + c.concept_id)
                else:
                    clusters.append(c)
                cluster_assign.append(c)
                child_cluster_assign.append(child)
            yield clusters, heuristic(cluster_assign, temp_clusters)

        # split_cus = sorted([(tree.root.cu_for_split(c) -
        #                      tree.root.category_utility(), i, c) for i,c in
        #                     enumerate(tree.root.children) if c.children])

        split_cus = []

        for i, target in enumerate(set(cluster_assign)):
            if len(target.children) == 0:
                continue
            c_labels = [label if label != target else child_cluster_assign[j] 
                        for j, label in enumerate(cluster_assign)]
            split_cus.append((heuristic(c_labels, temp_clusters), i, target))
    
        #for i,target in enumerate(tree.root.children):
        #    if len(target.children) == 0:
        #        continue
        #    rest = [child for child in tree.root.children if child != target]
        #    rest.extend(target.children)

        #    split_cus.append((heuristic(rest,tree,instances),i,target))

        split_cus = sorted(split_cus)

        # Exit early, we don't need to re-reun the following part for the
        # last time through
        if not split_cus:
            break

        # Split the least cohesive cluster
        tree.root.split(split_cus[0][2])

        nth_split+=1