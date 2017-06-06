import concept_formation.datasets as ds
from concept_formation.cluster import cluster_split_search,AIC,CU,BIC, depth_labels, cluster_iter, cluster
from concept_formation.visualize import visualize, visualize_clusters
from concept_formation.cobweb3 import Cobweb3Tree, Cobweb3Node
from concept_formation.trestle import TrestleTree
from concept_formation.structure_mapper import StructureMapper
from concept_formation.preprocessor import SubComponentProcessor
from concept_formation.preprocessor import Flattener
from concept_formation.preprocessor import Pipeline
from concept_formation.preprocessor import NameStandardizer
from concept_formation.preprocessor import ObjectVariablizer
from concept_formation.continuous_value import ContinuousValue
from concept_formation.evaluation import absolute_error
from itertools import cycle, islice
from random import random
from random import normalvariate, uniform, shuffle, seed, randint
from pprint import pprint
import datetime
from tabulate import tabulate
import uuid
import json
from sklearn.metrics import adjusted_rand_score

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

def init_tree_new(instances):
    tree = TrestleTree()
    for i in instances:
        pipe = Pipeline(NameStandardizer(tree.gensym),
                                 Flattener(), SubComponentProcessor(),
                                 StructureMapper(tree.root))
        temp_instance = pipe.transform(i)
        tree.root.increment_counts(temp_instance)
        is_match = False
        for c in tree.root.children:
            if c.is_exact_match(temp_instance):
                c.increment_counts(temp_instance)
                is_match = True
                break

        if is_match:
            continue
        else:
            node = Cobweb3Node()

            node.increment_counts(temp_instance)
            tree.root.children.append(node)
            node.parent = tree.root
            node.tree = tree
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
    if c1 not in parent.children:
        raise Exception('c1 not in parent')
    if c2 not in parent.children:
        print(json.dumps(parent.output_json()))
        print(json.dumps(c2.output_json()))
        raise Exception('c2 not in parent')
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
            ops = get_options(node)
        else:
            break

    for child in node.children:
        merge_at_node(child)




def glom2(instances):
    tree = init_tree_new(instances)
    # visualize(tree)
    merge_at_node(tree.root)
    return tree

# code from https://docs.python.org/3/library/itertools.html#itertools-recipes
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


def resort_rec(node):
    if len(node.children) == 0:
        return [k for k in node.av_counts['_key_feature']]
    else:
        child_order = [resort_rec(child) for child in sorted(node.children, key = lambda c : c.count, reverse=True)]
        child_order = [c for c in roundrobin(*child_order)]        
        return child_order
            


def resort_tree(tree,instances,iterations=1):
    instances = [i for i in instances]
    for i in instances:
        i['_key_feature'] = str(uuid.uuid4())

    tree.fit(instances)

    for i in range(iterations):
        order = resort_rec(tree.root)
        order = {v:i for i,v in enumerate(order)}
        re_train = [None for i in instances]
        for i in instances:
            re_train[order[i['_key_feature']]] = i

        tree.clear()
        tree.fit(re_train,randomize_first=False)
    return tree


def test_resort(runs = 10, hueristic=AIC):
    first_shots = []
    resorts = []


    for i in range(runs):
        seedy = datetime.datetime.now()

        seed(seedy)
        data = ds.load_rb_com_11()
        tree = TrestleTree()
        tree.fit(data)

        print('First Shot Tree')
        human_labels = [d['_human_cluster_label'] for d in data]
        s = 1
        best_split = None
        for clusters, h in cluster_iter(tree,data,hueristic,maxsplit=40,mod=False):
            if best_split is None:
                best_split =(s,len(set(clusters)),h,adjusted_rand_score(clusters,human_labels))
            elif h < best_split[2]:
                best_split =(s,len(set(clusters)),h,adjusted_rand_score(clusters,human_labels))
            print(s,len(set(clusters)),h,adjusted_rand_score(clusters,human_labels))
            s+=1

        first_shots.append(best_split)


        seed(seedy)
        data = ds.load_rb_com_11()
        tree = resort_tree(TrestleTree(),data)

        print('Resorted Tree')
        human_labels = [d['_human_cluster_label'] for d in data]
        s = 1
        best_split = None
        for clusters, h in cluster_iter(tree,data,hueristic,maxsplit=40,mod=False):
            if best_split is None:
                best_split =(s,len(set(clusters)),h,adjusted_rand_score(clusters,human_labels))
            elif h < best_split[2]:
                best_split =(s,len(set(clusters)),h,adjusted_rand_score(clusters,human_labels))
            print(s,len(set(clusters)),h,adjusted_rand_score(clusters,human_labels))
            s+=1
        resorts.append(best_split)

    print("Overall First Shots")
    print(tabulate(first_shots,headers=['S','C','H','ARI']))
    print("Overall Resorts")
    print(tabulate(resorts,headers=['S','C','H','ARI']))

def gen_numeric_data(num_clusters=4,num_samples=30,sigma=1,lop=0):
    xmean = [uniform(-6, 6) for i in range(num_clusters)]
    ymean = [uniform(-6, 6) for i in range(num_clusters)]


    # for i in range(len(xmean)):
    #     print(i,xmean[i],ymean[i])

    data = []

    for i in range(num_clusters):
        if lop > 0:
            data += [{'x': normalvariate(xmean[i], sigma), 
                  'y': normalvariate(ymean[i], sigma), 
                  '_label': str(i), 
                  '_guid':str(uuid.uuid4())} 
                for j in range(randint(max(num_samples-lop,1),randint+lop))]
        else :
            data += [{'x': normalvariate(xmean[i], sigma), 
                      'y': normalvariate(ymean[i], sigma), 
                      '_label': str(i), 
                      '_guid':str(uuid.uuid4())} 
                    for j in range(num_samples)]

    tab = [[i,xmean[i],ymean[i],len([c for c in data if c['_label']==str(i)])] for i in range(len(xmean))]
    print(tabulate(tab,headers=('Cluster','X mean','Y mean','N')))
    return data



def order_independence_test():
    data = gen_numeric_data()

    data1 = [d for d in data]
    data2 = [d for d in data]

    shuffle(data1)
    shuffle(data2)

    t1 = glom2(data1)
    print('glommed tree 1')
    t2 = glom2(data2)
    print('glommed tree 2')

    shuffle(data)

    l1 = cluster(t1,data,maxsplit=40,mod=False)
    l2 = cluster(t2,data,maxsplit=40,mod=False)

    print(tabulate([[i,len(set(l1[i])),len(set(l2[i])),adjusted_rand_score(l1[i],l2[i])]for i in range(40)],
        headers =['Split','C shuffle1','C shuffle2','ARI']))



    # for d in range(len(l1)):
    #     labels1 = [l for l in l1[d]]
    #     labels2 = [l for l in l2[d]]
    #     print(adjusted_rand_score(labels1,labels2))

    # l1 = cluster(t1,data,mod=False)[0]
    # l2 = cluster(t2,data,mod=False)[0]

    # print(adjusted_rand_score(l1,l2))




def agreement_to_human():
    data = ds.load_rb_com_11()

    shuffle(data)
    # data = data[:100]

    tree = glom2(data)

    shuffle(data)
    human_labels = [d['_human_cluster_label'] for d in data]

    s = 1
    for clusters, h in cluster_iter(tree,data,maxsplit=40,mod=False):
        print(s,h,adjusted_rand_score(clusters,human_labels))
        s+=1

def agreement_to_human2():
    data = ds.load_rb_com_11()

    shuffle(data)
    # data = data[:100]

    tree = TrestleTree()
    tree.fit(data)

    shuffle(data)
    human_labels = [d['_human_cluster_label'] for d in data]

    s = 1
    for clusters, h in cluster_iter(tree,data,maxsplit=40,mod=False):
        print(s,h,adjusted_rand_score(clusters,human_labels))
        s+=1

def recovery_test():
    data = ds.load_rb_com_11()[:60]
    shuffle(data)

    tree = glom2(data)

    shuffle(data)
    visualize(tree)
    for clusters, h in cluster_iter(tree,data,maxsplit=40,mod=False,labels=False):
        total = len(data)
        hit = 0
        for i in range(total):
            if data[i]['_guid'] in clusters[i].av_counts['_guid']:
                hit += 1
        print(hit/total)


def recover_test2():
    data = gen_numeric_data()
    tree = glom2(data)

    shuffle(data)
    visualize(tree)
    for clusters, h in cluster_iter(tree,data,maxsplit=40,mod=False,labels=False):
        total = len(data)
        hit = 0
        for i in range(total):
            if data[i]['_guid'] in clusters[i].av_counts['_guid']:
                hit += 1
        print(hit/total)


def incremental_comparison():
    seed(0)
    data = ds.load_rb_com_11()[:120]
    shuffle(data)
    tree1 = glom2(data)
    tree2 = TrestleTree()
    tree2.fit(data,randomize_first=False)

    shuffle(data)
    labels = [d['_human_cluster_label'] for d in data]

    print('Glom Tree')
    s = 1
    for clusters, h in cluster_iter(tree1,data,maxsplit=40,mod=False):
        print(s,h,adjusted_rand_score(clusters,labels))
        s += 1

    print()
    print('Ince Tree')
    s = 1
    for clusters, h in cluster_iter(tree2,data,maxsplit=40,mod=False):
        print(s,h,adjusted_rand_score(clusters,labels))
        s += 1


def projective_accuracy_comp():
    data = gen_numeric_data(num_samples=60)

    shuffle(data)

    dec = len(data)//2
    train = data[:dec]
    test = data[dec:]

    t1 = glom2(train)
    t2 = TrestleTree()
    t2.fit(train,randomize_first=False)

    cv1 = ContinuousValue()
    cv2 = ContinuousValue()

    for d in test:
        d['_y'] = d['y']
        d.pop('y',None) 

        cv1.update(absolute_error(t1,d,'y',d['_y']))
        cv2.update(absolute_error(t2,d,'y',d['_y']))

    print('Glom Result')
    print(str(cv1))

    print('Incremental Result')
    print(str(cv2))





if __name__ == '__main__':
    print('order independence test')
    order_independence_test()
    # agreement_to_human()
    # test_resort(50,BIC)
    # recover_test2()
    # incremental_comparison()
    # projective_accuracy_comp()


    # data = ds.load_rb_com_11()
    # data = ds.load_rb_s_13()
    # # data = ds.load_rb_wb_03()
    # counts = {}
    # for d in data:
    #     lab = d['_human_cluster_label']
    #     if lab not in counts:
    #         counts[lab] = 0
    #     counts[lab] += 1
    # pprint(counts)


    # seed(0)

    # num_clusters = 4
    # num_samples = 30
    # sigma =1

    # # xmean = [6,0,-9,-3]
    # # ymean = [7,-2,-8,0]

    # # xmean = [uniform(-6, 6) for i in range(num_clusters)]
    # # ymean = [uniform(-6, 6) for i in range(num_clusters)]

    # # for i in range(len(xmean)):
    # #     print(i,xmean[i],ymean[i])

    # # data = []


    # # for i in range(num_clusters):
    # #     data += [{'x': normalvariate(xmean[i], sigma), 'y':
    # #               normalvariate(ymean[i], sigma), '_label': str(i)} for j in
    # #              range(num_samples)]


    # # data = ds.load_forest_fires()[:100]
    # data = ds.load_rb_com_11()[:60]
    # ov = ObjectVariablizer()
    # data = ov.batch_transform(data)


    # # data = load_mushroom()[:50]
    # tree1 = glom2(data)
    # shuffle(data)
    # tree2 = glom2(data)


    # # tree = init_tree(data)
    # visualize(tree1)
    # visualize(tree2)
    # clus = cluster_split_search(tree,data,CU,1,40,False,True,True)
    # visualize_clusters(tree,clus)
