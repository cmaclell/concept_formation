"""
The evaluation module contains functions for evaluating the predictive
capabilities of CobwebTrees and their derivatives.
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
from random import shuffle
from re import search

from concept_formation.utils import mean
from concept_formation.utils import isNumber
from concept_formation.structure_mapper import StructureMapper

def probability(tree, instance, attr, val):
    """
    Returns the probability of a particular value of an attribute in the
    instance. One of the scoring functions for incremental_evaluation.

    If the instance currently contains the target attribute a shallow copy is
    created to allow the attribute to be predicted.

    .. warning:: This is an older function in the library and we are not quite
        sure how to set it up for component values under the new
        representation and so for the time being it will raise an Exception if
        it encounts a component.

    :param tree: A category tree to evaluate.
    :type tree: :class:`CobwebTree <concept_formation.cobweb.CobwebTree>`, :class:`Cobweb3Tree <concept_formation.cobweb3.Cobweb3Tree>`, or :class:`TrestleTree <concept_formation.trestle.TrestleTree>`
    :param instance: An instance to use query the tree with
    :type instance: {a1:v1, a2:v2, ...}
    :param attr: A target instance attribute to evaluate probability on
    :type attr: :ref:`Attribute<attributes>`
    :param val: The target value of the given attr
    :type val: A :ref:`Nominal<val-nom>` or :ref:`Numeric<val-num>` value.
    :returns: The probabily of the given instance attribute value in the given tree
    :rtype: float
    """
    if attr in instance:
        instance = {a:instance[a] for a in instance if not a == attr}
    concept = tree.categorize(instance)

    if isinstance(val, dict):
        raise Exception("Probability cannot be estimated on component attributes!")
        structure_mapper = StructureMapper(concept)
        temp_instance = structure_mapper.transform(instance)
        mapping = structure_mapper.get_mapping()

        #temp_instance = flatten_json(instance)
        #mapping = flat_match(concept, temp_instance)
        #temp_instance = rename_flat(temp_instance, mapping)

        probs = [concept.probability(sub_attr, temp_instance[sub_attr]) 
                 for sub_attr in temp_instance 
                 if search(r'^' + mapping[attr], sub_attr)]
        return mean(probs)
    else:
        return concept.probability(attr, val)

def error(tree, instance, attr, val):
    """
    Computes the error between the predicted value and the actual value for an
    attribute. One of the scoring functions for incremental_evaluation.

    .. warning:: We are not quite sure how to compute error or squared for
        a :ref:`Numeric values<val-num>` being missing (e.g., 0-1 vs. scale 
        of the numeric value cannot be averaged). So currently, this scoring
        function raises an Exception when it encounters a missing nunmeric
        value. We are also not sure how to handle error in the case of
        :ref:`Component Values<val-comp>` so it will also throw an exception 
        if encounters one of those.
    
    :param tree: A category tree to evaluate.
    :type tree: :class:`CobwebTree <concept_formation.cobweb.CobwebTree>`, :class:`Cobweb3Tree <concept_formation.cobweb3.Cobweb3Tree>`, or :class:`TrestleTree <concept_formation.trestle.TrestleTree>`
    :param instance: An instance to use query the tree with
    :type instance: {a1:v1, a2:v2, ...}
    :param attr: A target instance attribute to evaluate error on
    :type attr: :ref:`Attribute<attributes>`
    :param val: The target value of the given attr
    :type val: A :ref:`Nominal<val-nom>` or :ref:`Numeric<val-num>` value.
    :returns: The error of the given instance attribute value in the given tree
    :rtype: float, or int in the nominal case.
    """
    if attr in instance:
        instance = {a:instance[a] for a in instance if not a == attr}

    concept = tree.categorize(instance)

    if isinstance(val, dict):
        raise Exception("Currently does not support prediction error of component attributes.")
    elif isNumber(val):
        prediction = concept.predict(attr)
        if prediction is None:
            raise Exception("Not sure how to handle continuous values that are predicted to be missing.")
        e = val - prediction 
    else:
        prediction = concept.predict(attr)

        if val is None and isNumber(prediction):
            raise Exception("Not sure how to compare Continuous Values and None")

        if val == prediction:
            e = 0
        else:
            e = 1

    return e

def absolute_error(tree, instance, attr, val):
    """
    Returns the absolute error of the tree for a particular attribute value
    pair. One of the scoring functions for incremental_evaluation.

    :param tree: A category tree to evaluate.
    :type tree: :class:`CobwebTree <concept_formation.cobweb.CobwebTree>`, :class:`Cobweb3Tree <concept_formation.cobweb3.Cobweb3Tree>`, or :class:`TrestleTree <concept_formation.trestle.TrestleTree>`
    :param instance: An instance to use query the tree with
    :type instance: {a1:v1, a2:v2, ...}
    :param attr: A target instance attribute to evaluate error on
    :type attr: :ref:`Attribute<attributes>`
    :param val: The target value of the given attr
    :type val: A :ref:`Nominal<val-nom>` or :ref:`Numeric<val-num>` value.
    :returns: The error of the given instance attribute value in the given tree
    :rtype: float, or int in the nominal case.

    .. seealso:: :func:`error`
    """
    return abs(error(tree, instance, attr, val))

def squared_error(tree, instance, attr, val):
    """
    Returns the squared error of the tree for a particular attribute value
    pair. One of the scoring functions for incremental_evaluation.

    :param tree: A category tree to evaluate.
    :type tree: :class:`CobwebTree <concept_formation.cobweb.CobwebTree>`, :class:`Cobweb3Tree <concept_formation.cobweb3.Cobweb3Tree>`, or :class:`TrestleTree <concept_formation.trestle.TrestleTree>`
    :param instance: An instance to use query the tree with
    :type instance: {a1:v1, a2:v2, ...}
    :param attr: A target instance attribute to evaluate error on
    :type attr: :ref:`Attribute<attributes>`
    :param val: The target value of the given attr
    :type val: A :ref:`Nominal<val-nom>` or :ref:`Numeric<val-num>` value.
    :returns: The error of the given instance attribute value in the given tree
    :rtype: float, or int in the nominal case.

    .. seealso:: :func:`error`
    """
    e = error(tree, instance, attr, val)
    return e * e

#TODO - do we want to resurrect this one?
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

def incremental_evaluation(tree, instances, attr, run_length, runs=1,
                           score=probability, randomize_first=True):
    """
    Given a set of instances and an attribute, perform an incremental
    prediction task; i.e., try to predict the attribute for each instance
    before incorporating it into the tree. This will give a type of cross
    validated result and gives a sense of how performance improves over time.

    Incremental evaluation can use different scoring functions depending on
    the desired evaluation task:

    * :func:`probability` - The probability of the target attribute's value
      being present (i.e., accuracy). This is the default scoring function.
    * :func:`error` - The difference between the target attribute's value and
      the one predicted by the tree's concept.
    * :func:`absolute_error` - Returns the absolute value of the error.
    * :func:`squared_error` - Returns the error squared.

    :param tree: A category tree to evaluate.
    :type tree: :class:`CobwebTree <concept_formation.cobweb.CobwebTree>`, :class:`Cobweb3Tree <concept_formation.cobweb3.Cobweb3Tree>`, or :class:`TrestleTree <concept_formation.trestle.TrestleTree>`
    :param instances: A list of instances to use for evaluation
    :type instances: [:ref:`Instance<instance-rep>`, :ref:`Instance<instance-rep>`, ...]
    :param attr: A target instance attribute to use in evaluation.
    :type attr: :ref:`Attribute<attributes>`
    :param run_length: The number of training instances to use within a given run.
    :type run_length: int
    :param runs: The number of restarted runs to perform
    :type runs: int
    :param score: The scoring function to use for evaluation (default probability)
    :type score: function
    :param randomize_first: Whether to shuffle the first run of instances or not.
    :type randomize_first: bool
    :returns: A table that is `runs` x `run_length` where each row represents the score for successive instances within a run.
    :rtype: A table of scores.
    """
    scores = []
    for r in range(runs):
        #print(r)
        tree.clear()

        if randomize_first or r > 0:
            shuffle(instances)
        
        row = []

        for i,instance in enumerate(instances[:run_length+2]):
            #print(i)
            val = None
            if attr in instance:
                val = instance[attr]
            row.append(score(tree, instance, attr, val))
            tree.ifit(instance)
        scores.append(row)
    return scores
