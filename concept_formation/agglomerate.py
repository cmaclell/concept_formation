from concept_formation.cobweb3 import Cobweb3Tree, Cobweb3Node
from random import random

def init_tree(instances):
    nodes = []
    for i in instances:
        is_match = False
        for c in nodes:
            if c.is_exact_match(i):
                c.increment_counts(i)
                is_match = True
                break
        if not is_match:
            c = Cobweb3Node()
            c.increment_counts(i)
            nodes.append(c)
    tree = Cobweb3Tree()

    for n in nodes:
        tree.root.update_counts_from_node(n)
        tree.root.children.append(n)
        n.tree = tree
        n.parent = tree.root
    return tree

def cu_for_merge_into(node,c1,c2):
    temp = node.shallow_copy()
    t1 = c1.shallow_copy()
    t1.update_counts_from_node(c2)
    temp.children.append(t1)

    for c in node.children:
        if c == c1 or c == c2:
            continue
        temp.children.append(c.shallow_copy())

    return temp.category_utility()

def merge_into(parent, c1, c2):
    if len(c1.children) == 0:
        c1.create_child_with_current_counts()
    c1.update_counts_from_node(c2)
    parent.children.remove(c2)
    c1.children.append(c2)
    c2.parent = c1
    return c1

def get_options(parent):
    if len(parent.children) <= 2:
            return []
    options = []
    for i in range(len(parent.children)):
        c1 = parent.children[i]
        for j in range(i+1,len(parent.children)):
            c2 = parent.children[j]
            cu = cu_for_merge_into(parent,c1,c2)
            options.append((cu,random(),c1,c2))
    return options

def merge_at_node(node):
    if len(node.children) <= 2:
        return

    ops = get_options(node)
    # print([o[0] for o in ops])
    while len(ops) > 0:
        ops.sort(reverse=True)
        cu_opt, r, c1, c2 = ops[0]
        cu_curr = node.category_utility()
        if cu_opt >= cu_curr:
            merge_into(node,c1,c2)
            ops = [op for op in ops if op[1] != c2 and op[2] != c2]
            ops += get_options(node)
        else:
            break

    for child in node.children:
        merge_at_node(child)




def glom2(instances):
    tree = init_tree(instances)
    merge_at_node(tree.root)
    return tree

if __name__ == '__main__':
    from concept_formation.datasets import load_mushroom, load_forest_fires
    from random import normalvariate, uniform, shuffle, seed
    from concept_formation.visualize import visualize, visualize_clusters
    from concept_formation.cluster import cluster_split_search,AIC,CU,BIC

    seed(0)

    num_clusters = 4
    num_samples = 30
    sigma =1

    # xmean = [6,0,-9,-3]
    # ymean = [7,-2,-8,0]

    # xmean = [uniform(-6, 6) for i in range(num_clusters)]
    # ymean = [uniform(-6, 6) for i in range(num_clusters)]

    # for i in range(len(xmean)):
    #     print(i,xmean[i],ymean[i])

    # data = []


    # for i in range(num_clusters):
    #     data += [{'x': normalvariate(xmean[i], sigma), 'y':
    #               normalvariate(ymean[i], sigma), '_label': str(i)} for j in
    #              range(num_samples)]


    data = load_forest_fires()[:100]

    # data = load_mushroom()[:50]
    tree = glom2(data)
    # tree = init_tree(data)
    visualize(tree)
    # clus = cluster_split_search(tree,data,CU,1,40,False,True,True)
    # visualize_clusters(tree,clus)