from random import shuffle

from utils import mean

def probability_missing(tree, instance, attr):
    """
    Returns the probability of a particular attribute missing a value in the
    instance.
    """
    if attr in instance:
        instance = {a:instance[a] for a in instance if not a == attr}
    concept = tree.categorize(instance)
    return concept.get_probability_missing(attr)

def probability(tree, instance, attr, val):
    """
    Returns the probability of a particular value of an attribute in the
    instance.

    The instance should not contain the attribute, but if it does then a
    shallow copy is created that does not have the attribute.
    """
    if attr in instance:
        instance = {a:instance[a] for a in instance if not a == attr}
    concept = tree.categorize(instance)
    return concept.get_probability(attr, val)

def flexible_probability(tree, instance, attrs=None):
    """
    Returns the average probability of each value in the instance (over all
    attributes).
    """
    if attrs is None:
        attrs = instance.keys()

    probs = []
    for attr in attrs:
        if attr in instance:
            val = instance[attr]
            probs.append(probability(tree, instance, attr, val))
        else:
            probs.append(probability_missing(tree, instance, attr))
    return mean(probs)

def incremental_prediction(tree, instances, run_length, runs=1, attr=None):
    """
    Given a set of instances, perform an incremental prediction task; i.e.,
    try to flexibly predict each instance before incorporating it into the 
    tree. This will give a type of cross validated result.
    """
    if attr is None:
        possible_attrs = set([k for i in instances for k in i.keys()])

    accuracy = []
    for j in range(runs):
        # reset the tree each run
        tree = tree.__class__()

        run_accuracy = []
        shuffle(instances)
        for instance in instances[:run_length]:
            if attr:
                run_accuracy.append(probability(tree, instance, attr, instance[attr]))
            else:
                run_accuracy.append(flexible_probability(tree, instance,
                                                         possible_attrs))
            tree.ifit(instance)
        accuracy.append(run_accuracy)
    return accuracy
