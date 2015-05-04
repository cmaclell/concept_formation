# The trestle.py file has a bunch of top level functions in it that are meant
# to run experiments with Trestle but don't belong is the actual trestle algorithm
# so I've moved them all here and changed them to not rely on being in Trestle ittree.
from trestle import Trestle

def kc_label(filename, length):
    """
    Used to provide a clustering of a set of examples provided in a JSON
    file. It starts by incorporating the examples into the categorization
    tree multiple times. After incorporating the instances it then
    categorizes each example (without updating the tree) and returns the
    concept it was assoicated with.
    """
    tree = Trestle()

    json_data = open(filename, "r")
    instances = json.load(json_data)
    print("%i Instances." % len(instances))
    shuffle(instances)
    instances = instances[0:length]
    o_instances = copy.deepcopy(instances)
    for instance in instances:
        #if "success" in instance:
        #    del instance['success']
        if "guid" in instance:
            del instance['guid']
    json_data.close()
    clusters = {}
    previous = {}
    g_instances = {}

    for i in instances:
        previous[tree.flatten_instance(i)] = None

    # train initially
    for x in range(1):
        shuffle(instances)
        for n, i in enumerate(instances):
            print("training instance: " + str(n))
            tree.ifit(i)

    # categorize to add guids
    mapping = {}
    for idx, inst in enumerate(o_instances):
        print("categorizing instance: %i" % idx)

        if inst['guid'] in mapping:
            # ignore duplicate states.
            print("skipping duplicate guid")
            continue

        instance = copy.deepcopy(inst)
        
        #print(instance)

        # we want the KCS for only the correct productions.
        instance['Outcome'] = 'CORRECT'
        # for now just make the action correct, but leave the action.. so
        # closest action that is correct.
        #del instance['action']
        #if 'destination' in instance:
        #    del instance['destination']

        #del instance['r1']
        #del instance['r2']

        #if "success" in instance:
        #    del instance['success']
        if "guid" in instance:
            del instance['guid']
        g_instances[inst['guid']] = instance

        #print(instance)
        
        #print()
        mapping[inst['guid']] = tree.trestle_categorize(instance)
        #print(mapping[inst['guid']].concept_name)

    # add guids
    for g in mapping:
        curr = mapping[g]
        while curr:
            curr.av_counts['has-guid'] = {"1":True}
            if 'guid' not in curr.av_counts:
                curr.av_counts['guid'] = {}
            curr.av_counts['guid'][g] = True
            curr = curr.parent
    
    for g in mapping:
        cluster = mapping[g]
        if cluster.parent:
            cluster = cluster.parent
        clusters[g] = cluster.concept_name

    with open('visualize/output.json', 'w') as f:
        f.write(json.dumps(tree.output_json()))

    # Output data for datashop KC labeling
    guidKcs = []
    for g in mapping:
        kcs = []
        temp = mapping[g]
        kcs.append(temp.concept_name)
        while temp.parent:
            temp = temp.parent
            kcs.append(temp.concept_name)
        kcs.append(g) 
        kcs.reverse()
        guidKcs.append(kcs)

    with open('kc-labels.csv', 'w') as f:
        max_len = 0
        for kc in guidKcs:
            if len(kc) > max_len:
                max_len = len(kc)

        output = []
        for kc in guidKcs:
            for i in range(max_len - len(kc)):
                kc.append(kc[-1])
            output.append(",".join(kc))

        f.write("\n".join(output))

    #print(json.dumps(tree.output_json()))

    return clusters


def noise_experiments():
    from sklearn import metrics
    import math
    
    samples = 10
    iterations = 10

    print("continuous noise\tnominal noise\taccuracy\tstd")
    for cm in range(11):
        continuous_noise = cm/10.0
        for nm in range(11):
            nominal_noise = nm/10.0
            ari = []
            for xxx in range(iterations):
                tree = Trestle()

                #Clustering
                data1 = [{'key': 'c1', 'cf1': normalvariate(1,0.001), 'nf1':
                          'one'} for i in range(samples)]
                data2 = [{'key': 'c2', 'cf1': normalvariate(2,0.001), 'nf1':
                          'two'} for i in range(samples)]
                data3 = [{'key': 'c3', 'cf1': normalvariate(3,0.001), 'nf1':
                          'three'} for i in range(samples)]
                data4 = [{'key': 'c4', 'cf1': normalvariate(4,0.001), 'nf1':
                          'four'} for i in range(samples)]

                data = data1 + data2 + data3 + data4
                shuffle(data)

                labels_true = [d[a] for d in data for a in d if a == 'key']
                data = [{a: d[a] for a in d} for d in data]
                
                noisy = []
                for d in data:
                    temp = d

                    nv = set()
                    nv.add('one')
                    nv.add('two')
                    nv.add('three')
                    nv.add('four')

                    cv = set()
                    cv.add(1)
                    cv.add(2)
                    cv.add(3)
                    cv.add(4)

                    # nominal noise
                    if random() < nominal_noise:
                        s = set()
                        s.add(temp['nf1'])
                        temp['nf1'] = choice(list(nv - s))

                    # continuous value noise
                    if random() < continuous_noise:
                        s = set()
                        s.add(round(temp['cf1']))
                        temp['cf1'] = normalvariate(choice(list(cv - s)),0.001)
                              
                    noisy.append(temp)

                clusters = tree.cluster(noisy)

                ari.append(metrics.adjusted_rand_score(labels_true, clusters))

            mean = sum(ari)/(1.0 * len(ari))
            std = math.sqrt((float(sum([(v - mean) * (v - mean) for v in ari])) /
                             (len(ari) - 1.0)))
            print("%0.4f\t%0.4f\t%0.4f\t%0.4f" % (continuous_noise, nominal_noise, mean, std))

# I don't fully understand what this function was for but it never gets used 
# and it's specific to RumbleBlocks data so I moved it out. It's possible that we
# want it for any data that has a GUID tag to sort by -eharpste.
def order_towers(tres):
    """
    Given a number of towers with GUIDs added return a better
    training ordering.
    """
    L = []
    if not tres.children:
        if 'guid' in tres.av_counts:
            for guid in tres.av_counts['guid']:
                L.append(guid)
        return L
    else:
        sorted_c = sorted(tres.children, key=lambda c: -1 * c.count)
        lists = []

        for c in sorted_c:
            lists.append(c.order_towers())

        while lists:
            for l in lists:
                if l:
                    L.append(l.pop(0))
                    
            lists = [l for l in lists if l]
        return L
