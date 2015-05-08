# This is going to be the root file that you would use to work with each of the
# algorithms in this package. I went with tlc because of Trestle, Labyrinth, Cobweb
# but we can rename it later if we want

import json
import re
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

def cluster(self, instances, minsplit=1, maxsplit=1, samplesize=-1, mod=True):
    """
    Returns a list of clusterings. the size of the list will be (1 +
    maxsplit - minsplit). The first clustering will be if the root was
    split, then each subsequent clustering will be if the least coupled
    cluster (in terms of category utility) is split. This might stop early
    if you reach the case where there are no more clusters to split (each
    instance in its own cluster).

    The sample parameter can be used to specify a percentage (0.0-1.0) of 
    the instances to use to build the initial concept tree against which 
    all of the instances will be categorized for clustering. If no sample
    value is specified then the tree will be built from all instances.
    Sampling is not random but based on taking an inital sublist. If
    random sampling is desired then shuffle the input list.
    """
    
    if mod:
        if samplesize > 0.0 and samplesize < 1.0 :
            temp_clusters = [self.ifit(instance) for instance in instances[:floor(len(instances)*samplesize)]]
            temp_clusters.extend([self.categorize(instance) for instance in instances[floor(len(instances)*samplesize):]])
        else:
            temp_clusters = [self.ifit(instance) for instance in instances]
    else:
        temp_clusters = [self.categorize(instance) for instance in instances]

    clusterings = []
    
    self.generate_d3_visualization('output-pre')

    for nth_split in range(minsplit, maxsplit+1):
        #print(len(set([c.concept_id for c in temp_clusters])))

        clusters = []
        for i,c in enumerate(temp_clusters):
            while (c.parent and c.parent.parent):
                c = c.parent
            clusters.append("Concept" + c.concept_id)
        clusterings.append(clusters)

        split_cus = sorted([(self.root.cu_for_split(c) -
                             self.root.category_utility(), i, c) for i,c in
                            enumerate(self.root.children) if c.children])

        # Exit early, we don't need to re-reun the following part for the
        # last time through
        if nth_split == maxsplit or not split_cus:
            break

        # Split the least cohesive cluster
        self.root.split(split_cus[-1][2])
    
    self.generate_d3_visualization('output-post')
    self.generate_d3_visualization('output')

    return clusterings

def generate_d3_visualization(tree, fileName):
    """
    Generates the .js file that is used by index.html to generate the d3 tree.
    """
    #with open('visualize/output.json', 'w') as f:
    #    f.write(json.dumps(self.root.output_json()))
    fname = 'visualize/'+fileName+'.js'
    with open(fname, 'w') as f:
        f.write("var output = '"+re.sub("'", '',
                                        json.dumps(tree.root.output_json()))+"';")

def h_label(self,instances,fit=True):
    """
    Returns a hierarchical labeling of each instance from the root down
    to the most specific leaf. It returns a 2D matrix that is 
    len(instances) X max(numConceptParents). Labels are provided general
    to specific across each row If an instance was categorized shallower in
    the tree its labeling row will contain empty cells. 
    
    instances -- a collection of instances

    fit -- a flag for whether or not the labeling should come from
    fitting (i.e. modifying) the instances or categorizing 
    (i.e. non-modifying) the instances.
    """

    if fit:
        temp_labels = [self.ifit(instance) for instance in instances]
    else:
        temp_labels = [self.categorize(instance) for instance in instances]

    final_labels = []
    max_labels = 0
    for t in temp_labels:
        labs = []
        count = 0
        label = t
        while label.parent:
            labs.append("Concept" + label.concept_id)
            count += 1
            label = label.parent
        labs.append("Concept" + label.concept_id)
        count += 1
        final_labels.append(labs)
        if count > max_labels:
            max_labels = count
   
    for f in final_labels:
       f.reverse()
       while len(f) < max_labels:
           f.append("")

    return final_labels

if __name__ == "__main__":
    data = [{'x': random.normalvariate(0,0.5)} for i in range(10)]
    data += [{'x': random.normalvariate(2,0.5)} for i in range(10)]
    data += [{'x': random.normalvariate(4,0.5)} for i in range(10)]
    data = sort_dissimilar(data)

    tree = Cobweb3Tree()
    clusters = cluster(tree, data)
    print(clusters)
    print(set(clusters))
