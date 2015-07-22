from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
from random import shuffle
from numbers import Number
from re import search

from concept_formation.utils import mean
#from concept_formation.structure_mapper import flatten_json
#from concept_formation.structure_mapper import flat_match
#from concept_formation.structure_mapper import rename_flat
from concept_formation.cobweb3 import ContinuousValue
from concept_formation.structure_mapper import StructureMapper

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

    if isinstance(val, dict):
        structure_mapper = StructureMapper(concept)
        temp_instance = structure_mapper.transform(instance)
        mapping = structure_mapper.get_mapping()

        #temp_instance = flatten_json(instance)
        #mapping = flat_match(concept, temp_instance)
        #temp_instance = rename_flat(temp_instance, mapping)

        probs = [concept.get_probability(sub_attr, temp_instance[sub_attr]) 
                 for sub_attr in temp_instance 
                 if search('^' + mapping[attr], sub_attr)]
        return mean(probs)
    else:
        return concept.get_probability(attr, val)

def error(tree, instance, attr, val):
    """
    Computes the error between the predicted value and the actual value for an
    attribute. 

    Not quite sure how to compute error or squared for missing values with
    continuous attributes (e.g., 0-1 vs. scale of the continuous attribute
    cannot be averaged).
    """
    if attr in instance:
        instance = {a:instance[a] for a in instance if not a == attr}

    concept = tree.categorize(instance)

    if isinstance(val, dict):
        raise Exception("Currently does not support prediction error of component attributes.")
    elif isinstance(val, Number):
        prediction = concept.predict(attr)
        if prediction is None:
            raise Exception("Not sure how to handle continuous values that are predicted to be missing.")
        e = val - prediction 
    else:
        if val is None and isinstance(tree.root.av_counts[attr],
                                      ContinuousValue):
            raise Exception("Not sure how to handle missing continuous values.")

        prediction = concept.predict(attr)

        if val == prediction:
            e = 0
        else:
            e = 1

    return e

def absolute_error(tree, instance, attr, val):
    """
    Returns the error of the tree for a particular attribute value pair.
    """
    return abs(error(tree, instance, attr, val))

def squared_error(tree, instance, attr, val):
    """
    Returns the error of the tree for a particular attribute value pair.
    """
    e = error(tree, instance, attr, val)
    return e * e

#def flexible_probability(tree, instance, attrs=None):
#    """
#    Returns the average probability of each value in the instance (over all
#    attributes).
#    """
#    if attrs is None:
#        attrs = instance.keys()
#
#    probs = []
#    for attr in attrs:
#        if attr in instance:
#            probs.append(probability(tree, instance, attr, instance[attr]))
#        else:
#            probs.append(probability(tree, instance, attr, None))
#    return mean(probs)

def incremental_prediction(tree, instances, attr, run_length, runs=1,
                           score=probability):
    """
    Given a set of instances and an attribute, perform an incremental
    prediction task; i.e., try to predict the attr for each instance before
    incorporating it into the tree. This will give a type of cross validated
    result.

    Currently different score functions are supported, this defaults to
    probaility of missing value (i.e., accuracy). 
    """
    #if attr is None:
    #    possible_attrs = set([k for i in instances for k in i.keys()])

    scores = []
    for r in range(runs):
        print(r)
        tree.clear()

        shuffle(instances)
        
        scores.append((0,0))
        tree.ifit(instances[0])

        for i,instance in enumerate(instances[1:run_length]):
            val = None
            if attr in instance:
                val = instance[attr]
            scores.append((i+1, score(tree, instance, attr, val)))
            tree.ifit(instance)

    return scores
