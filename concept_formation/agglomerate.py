from concept_formation.cobweb3 import Cobweb3Tree, Cobweb3Node




def all_pairs(clusters):
    pairs = []
    for i, c1 in enumerate(clusters):
        for j, c2 in enumerate(clusters):
            if c1 == c2:
                continue
            temp_par = Cobweb3Node()
            temp_par.update_counts_from_node(c1)
            temp_par.update_counts_from_node(c2)
            c1_temp = c1.shallow_copy()
            c2_temp = c2.shallow_copy()
            temp_par.children.append(c1_temp)
            temp_par.children.append(c2_temp)
            c1_temp.parent = temp_par
            c2_temp.parent = temp_par
            pairs.append((temp_par.category_utility(),i,j,c1,c2))

    return pairs


def merge(c1,c2):
    temp_par = Cobweb3Node()
    temp_par.update_counts_from_node(c1)
    temp_par.update_counts_from_node(c2)
    temp_par.children.append(c1)
    temp_par.children.append(c2)
    c1.parent = temp_par
    c2.parent = temp_par
    return temp_par

def tree_point(node,tree):
    node.tree = tree
    for child in node.children:
        tree_point(child,tree)

def glom(instances):
    """
    Given a set of instances return a Cobweb3Tree formed by agglomorating those
    instances rather than dividing them like the normal algorithm.
    """


    clusters = []
    for i in instances:
        is_match = False
        for c in clusters:
            if c.is_exact_match(i):
                c.increment_counts(i)
                is_match = True
                break
        if not is_match:
            c = Cobweb3Node()
            c.increment_counts(i)
            clusters.append(c)

    # clusters is now a list of CobwebNodes that represent the unique set of instances
    while(len(clusters) > 1):
        pairs = all_pairs(clusters)
        pairs.sort(reverse=True)
        cu, i, j, c1, c2 = pairs[0]
        clusters.remove(c1)
        clusters.remove(c2)
        clusters.append(merge(c1,c2))


    #clusters is not a list of 1 element that should become the new root of a Cobweb3Tree
    tree = Cobweb3Tree()
    tree.root = clusters[0]
    tree_point(clusters[0],tree)

    return tree

if __name__ == '__main__':
    from concept_formation.datasets import load_mushroom
    from concept_formation.visualize import visualize_clusters
    from concept_formation.cluster import cluster_split_search,AIC,CU


    data = load_mushroom()[:50]
    tree = glom(data)
    clus = cluster_split_search(tree,data,CU,1,40,False,True,True)
    visualize_clusters(tree,clus)