from trestle import Trestle


def predict(tree, instance):
    """
    Given an instance predict any missing attribute values without
    modifying the tree. A modification for component values.
    """
    prediction = {}

    # make a copy of the instance
    # call recursively on structured parts
    for attr in instance:
        if isinstance(instance[attr], dict):
            prediction[attr] = tree.predict(instance[attr])
        else:
            prediction[attr] = instance[attr]

    concept = tree.trestle_categorize_leaf(prediction)
    #print(concept)
    #print(tree)
    
    for attr in concept.av_counts:
        if attr in prediction:
            continue
       
        # sample to determine if the attribute should be included
        num_attr = sum([concept.av_counts[attr][val] for val in
                        concept.av_counts[attr]])
        if random() > (1.0 * num_attr) / concept.count:
            continue
        
        nominal_values = []
        component_values = []

        float_num = 0.0
        float_mean = 0.0
        float_std = 0.0
        
        if isinstance(concept.av_counts[attr], ContinuousValue):
            float_num = concept.av_counts[attr].num
            float_mean = concept.av_counts[attr].mean
            float_std = concept.av_counts[attr].std

        else:
            for val in concept.av_counts[attr]:
                if isinstance(val, Trestle):
                    component_values += [val] * concept.av_counts[attr][val] 
                else:
                    nominal_values += [val] * concept.av_counts[attr][val]

        rand = random()

        if rand < ((len(nominal_values) * 1.0) / (len(nominal_values) +
                                                  len(component_values) +
                                                  float_num)):
            prediction[attr] = choice(nominal_values)
        elif rand < ((len(nominal_values) + len(component_values) * 1.0) /
                     (len(nominal_values) + len(component_values) +
                      float_num)):
            prediction[attr] = choice(component_values).predict({})
        else:
            prediction[attr] = normalvariate(float_mean,
                                             float_std)

    return prediction


def specific_prediction(tree, instance, attr, guessing=False):
    """
    Uses the TRESTLE algorithm to make a prediction about the given
    attribute. 
    """
    concept = tree.trestle_categorize_leaf(instance)
    return concept.get_probability(attr, instance[attr])

def flexible_prediction(tree, instance, guessing=False):
    """
    A modification of flexible prediction to handle component values.
    The flexible prediction task is called on all subcomponents. To compute
    the accuracy for each subcomponent.

    Guessing is the basecase that just returns the root probability
    """
    
    probs = []
    for attr in instance:
        #TODO add support for relational attribute values 
        if isinstance(instance[attr], list):
            continue
        if isinstance(instance[attr], dict):
            probs.append(tree.flexible_prediction(instance[attr], guessing))
            continue

        # construct an object with missing attribute
        temp = {}
        for attr2 in instance:
            if attr == attr2:
                continue
            temp[attr2] = instance[attr2]

        if guessing:
            probs.append(tree.get_probability(attr, instance[attr]))
        else:
            probs.append(tree.concept_attr_value(temp, attr, instance[attr]))

    if len(probs) == 0:
        print(instance)
        return -1 
    return sum(probs) / len(probs)



def sequential_prediction(tree, filename, length, attr=None, guessing=False):
    """
    Given a json file, perform an incremental sequential prediction task. 
    Try to flexibly predict each instance before incorporating it into the 
    tree. This will give a type of cross validated result.
    """
    json_data = open(filename, "r")
    instances = json.load(json_data)
    #instances = instances[0:length]
    for instance in instances:
        if "guid" in instance:
            del instance['guid']
    accuracy = []
    nodes = []
    for j in range(1):
        shuffle(instances)
        for n, i in enumerate(instances):
            if n >= length:
                break
            if attr:
                accuracy.append(tree.specific_prediction(i, attr, guessing))
            else:
                accuracy.append(tree.flexible_prediction(i, guessing))
            nodes.append(tree.num_concepts())
            tree.ifit(i)
    json_data.close()
    return accuracy, nodes
