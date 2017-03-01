"""
This module contains an number of proprocessors that can be used on various
forms of raw input data to convert an instance into a shape that Trestle would
better understand. Almost all preprocessors preserve the original semantics of
an instance and are mainly being used to prep for Trestle's internal operations.

Two abstract preprocessors are defined:

* :class:`Preprocessor` - Defines the general structure of a preprocessor.
* :class:`Pipeline` - Allows for chaining a collection of preprocessors together.

Trestle's normal implementation uses a standard pipeline of preprocessors that
run in the following order:

#. :class:`SubComponentProcessor` - Pulls any sub-components present in the
   instance to the top level of the instance and adds ``has-component``
   relations to preserve semantics.
#. :class:`Flattener` - Flattens component instances into a number of tuples
   (i.e. ``(attr,component)``) for faster hashing and access.
#. :class:`StructureMapper<concept_formation.structure_mapper.StructureMapper>` 
    - Gives any variables unique names so they can be renamed in matching without 
    colliding, and matches instances to the root concept.

The remaining preprocessors are helper classes designed to support data that is
not stored in Trestle's conventional representation:

* :class:`Tuplizer` - Looks for relation attributes denoted as strings (i.e.
  ``'(relation e1 e1)'``) and replaces the string attribute name with the
  equivalent tuple representation of the relation.
* :class:`ListProcessor` - Search for list values and extracts their elements
  into their own objects and replaces the list with ordering and element-of
  relations. Intended to preserve the semenatics of a list in JSON representation.
* :class:`ObjectVariablizer` - Looks for component objects within an instance
  and variablizes their names by prepending a ``'?'``.
* :class:`NumericToNominal` - Converts numeric values to nominal ones.
* :class:`NominalToNumeric` - Converts nominal values to numeric ones.
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

from copy import deepcopy
from numbers import Number
import collections

_gensym_counter = 0


def get_attribute_components(attribute, vars_only=True):
    """
    Gets component names out of an attribute

    >>> from pprint import pprint
    >>> attr = ('a', ('sub1', '?c1'))
    >>> get_attribute_components(attr)
    {'?c1'}

    >>> attr = '?c1'
    >>> get_attribute_components(attr)
    {'?c1'}

    >>> attr = ('a', ('sub1', 'c1'))
    >>> get_attribute_components(attr)
    set()

    >>> attr = 'c1'
    >>> get_attribute_components(attr)
    set()
    """
    names = set()

    if vars_only is not True and attribute[0] != '_':
        names.add(attribute)

    if isinstance(attribute, tuple):
        for ele in attribute:
            if isinstance(ele, tuple):
                for name in get_attribute_components(ele, vars_only):
                    names.add(name)
            else:
                if ((vars_only is not True or (len(ele) > 0 and ele[0] == '?'))
                    and (ele != '_' and len(ele) > 0 and ele[0] != '_')):
                    names.add(ele)

    elif ((vars_only is not True and attribute[0] != '_') or
          attribute[0] == '?'):
        names.add(attribute)

    return names


def default_gensym():
    """
    Generates unique names for naming renaming apart objects.

    :return: a unique object name
    :rtype: 'o'+counter
    """
    global _gensym_counter
    _gensym_counter += 1
    return '?o' + str(_gensym_counter)


def _reset_gensym():
    """
    Resets the gensym counter to 0, which is useful for doctesting. Do not call
    this function during normal operation.
    """
    global _gensym_counter
    _gensym_counter = 0


class Preprocessor(object):
    """
    A template class that defines the functions a preprocessor class should
    implement. In particular, a preprocessor should tranform an instance and
    implement a function for undoing this transformation.
    """
    def transform(self, instance):
        """
        Transforms an instance.
        """
        raise NotImplementedError("Class must implement transform")

    def undo_transform(self, instance):
        """
        Undoes a transformation to an instance.
        """
        raise NotImplementedError("Class must implement undo_transform")

    def batch_transform(self, instances):
        """
        Transforms a collection of instances.
        """
        return [self.transform(instance) for instance in instances]

    def batch_undo(self, instances):
        """
        Undoes transformation for a collection of instances
        """
        return [self.undo_transform(instance) for instance in instances]


class OneWayPreprocessor(Preprocessor):
    """
    A template class that defines a transformation function that only works in
    the forward direction. If undo_transform is called then an exact copy of
    the given object is returned.
    """

    def undo_transform(self, instance):
        """
        No-op
        """
        return {k: instance[k] for k in instance}


class Pipeline(Preprocessor):
    """
    A special preprocessor class used to chain together many preprocessors.
    Supports the same transform and undo_transform functions as a regular
    preprocessor.
    """
    def __init__(self, *preprocessors):
        self.preprocessors = preprocessors

    def transform(self, instance):
        """
        Apply a series of transformations to the instance.
        """
        for pp in self.preprocessors:
            instance = pp.transform(instance)
        return instance

    def undo_transform(self, instance):
        """
        Undo the series of transformations done to the instance.
        """
        for pp in reversed(self.preprocessors):
            instance = pp.undo_transform(instance)
        return instance


class Tuplizer(Preprocessor):
    """
    Converts all string versions of relations into tuples.

    Relation attributes are expected to be specified as a string enclosed in
    ``(`` ``)`` with values delimited by spaces. We conventionally use a prefix
    notation for relations ``(related a b)`` but this preprocessor should be
    flexible enough to handle postfix and prefix.

    This is a helper function preprocessor and so is not part of
    :class:`StructureMapper
    <concept_formation.structure_mapper.StructureMapper>`'s standard pipeline.

    >>> tuplizer = Tuplizer()
    >>> instance = {'(foo1 o1 (foo2 o2 o3))': True}
    >>> print(tuplizer.transform(instance))
    {('foo1', 'o1', ('foo2', 'o2', 'o3')): True}
    >>> print(tuplizer.undo_transform(tuplizer.transform(instance)))
    {'(foo1 o1 (foo2 o2 o3))': True}

    >>> instance = {'(place x1 12.4 9.6 (div width 18.2))':True}
    >>> tuplizer = Tuplizer()
    >>> tuplizer.transform(instance)
    {('place', 'x1', 12.4, 9.6, ('div', 'width', 18.2)): True}
    """
    def transform(self, instance):
        """
        Convert at string specified relations into tuples.
        """
        return {self._tuplize_relation(attr): instance[attr] for attr in
                instance}

    def undo_transform(self, instance):
        """
        Convert tuple relations back into their string forms.
        """
        return {self._stringify_relation(attr): instance[attr] for attr in
                instance}

    def _tuplize_relation(self, relation):
        """
        Converts a string formatted relation into a tuplized relation.

        :param attr: The relational attribute formatted as a string
        :type attr: string
        :param mapping: A dictionary of mappings with component names as keys.
            Just the keys are used (i.e., as a set) to determine if elements in
            the relation are objects.
        :type mapping: dict
        :return: A new relational attribute in tuple format
        :rtype: tuple

        >>> relation = '(foo1 o1 (foo2 o2 o3))'
        >>> tuplizer = Tuplizer()
        >>> tuplizer._tuplize_relation(relation)
        ('foo1', 'o1', ('foo2', 'o2', 'o3'))
        """
        if relation[0] != '(':
            return relation

        stack = [[]]

        for val in relation.split(' '):
            end = 0

            if val[0] == '(':
                stack.append([])
                val = val[1:]

            while val[-1] == ')':
                end += 1
                val = val[:-1]

            current = stack[-1]
            try:
                val = float(val)
            except ValueError:
                val = val
            current.append(val)

            while end > 0:
                last = tuple(stack.pop())
                current = stack[-1]
                current.append(last)
                end -= 1

        final = tuple(stack[-1][-1])
        return final

    def _stringify_relation(self, relation):
        """
        Converts a tupleized relation into a string formated relation.

        >>> relation = ('foo1', 'o1', ('foo2', 'o2', 'o3'))
        >>> tuplizer = Tuplizer()
        >>> tuplizer._stringify_relation(relation)
        '(foo1 o1 (foo2 o2 o3))'
        """
        if isinstance(relation, tuple):
            relation = [self._stringify_relation(ele) if isinstance(ele, tuple)
                        else ele for ele in relation]
            return "(" + " ".join(relation) + ")"
        else:
            return relation


def rename_relation(relation, mapping):
    """
    Takes a tuplized relational attribute (e.g., ``('before', 'o1', 'o2')``)
    and a mapping and renames the components based on the mapping. This
    function contains a special edge case for handling dot notation which is
    used in the NameStandardizer.

    :param attr: The relational attribute containing components to be renamed
    :type attr: :ref:`Relation Attribute<attr-rel>`
    :param mapping: A dictionary of mappings between component names
    :type mapping: dict
    :return: A new relational attribute with components renamed
    :rtype: tuple

    >>> relation = ('foo1', 'o1', ('foo2', 'o2', 'o3'))
    >>> mapping = {'o1': 'o100', 'o2': 'o200', 'o3': 'o300'}
    >>> rename_relation(relation, mapping)
    ('foo1', 'o100', ('foo2', 'o200', 'o300'))

    >>> relation = ('foo1', ('o1', ('o2', 'o3')))
    >>> mapping = {('o1', ('o2', 'o3')): 'o100'}
    >>> rename_relation(relation, mapping)
    ('foo1', 'o100')
    """
    return tuple(mapping[v] if v in mapping else rename_relation(v, mapping) if
                 isinstance(v, tuple) else v for v in relation)


class NameStandardizer(Preprocessor):
    """
    A preprocessor that standardizes apart object names.

    Given an instance rename all the components so they have unique names.

    .. :warning: relations cannot have dictionaries as values (i.e., cannot be
        subojects).
    .. :warning: relations can only exist at the top level, not in sub-objects.

    This will rename component attributes as well as any occurance of the
    component's name within relation attributes. This renaming is necessary to
    allow for a search between possible mappings without collisions.

    This is the first operation in :class:`StructureMapper
    <concept_formation.structure_mapper.StructureMapper>`'s standard
    pipeline.

    :param gensym: a function that returns unique object names (str) on each
        call. If None, then :func:`default_gensym` is used, which keeps a
        global object counter.
    :type gensym: a function

    # Reset the symbol generator for doctesting purposes.
    >>> _reset_gensym()
    >>> import pprint
    >>> instance = {'nominal': 'v1', 'numeric': 2.3, 'c1': {'a1': 'v1'}, '?c2':
    ...             {'a2': 'v2', '?c3': {'a3': 'v3'}}, '(relation1 c1 ?c2)':
    ...             True, 'lists': [{'c1': {'inner': 'val'}}, 's2', 's3'],
    ...             '(relation2 (a1 c1) (relation3 (a3 (?c3 ?c2))))': 4.3,
    ...             ('relation4', '?c2', '?c4'):True}
    >>> tuplizer = Tuplizer()
    >>> instance = tuplizer.transform(instance)
    >>> std = NameStandardizer()
    >>> std.undo_transform(instance)
    Traceback (most recent call last):
        ...
    Exception: Must call transform before undo_transform!
    >>> new_i = std.transform(instance)
    >>> old_i = std.undo_transform(new_i)
    >>> pprint.pprint(instance)
    {'?c2': {'?c3': {'a3': 'v3'}, 'a2': 'v2'},
     'c1': {'a1': 'v1'},
     'lists': [{'c1': {'inner': 'val'}}, 's2', 's3'],
     'nominal': 'v1',
     'numeric': 2.3,
     ('relation1', 'c1', '?c2'): True,
     ('relation2', ('a1', 'c1'), ('relation3', ('a3', ('?c3', '?c2')))): 4.3,
     ('relation4', '?c2', '?c4'): True}
    >>> pprint.pprint(new_i)
    {'?o1': {'?o2': {'a3': 'v3'}, 'a2': 'v2'},
     'c1': {'a1': 'v1'},
     'lists': [{'c1': {'inner': 'val'}}, 's2', 's3'],
     'nominal': 'v1',
     'numeric': 2.3,
     ('relation1', 'c1', '?o1'): True,
     ('relation2', ('a1', 'c1'), ('relation3', ('a3', ('?o2', '?o1')))): 4.3,
     ('relation4', '?o1', '?o3'): True}
    >>> pprint.pprint(old_i)
    {'?c2': {'?c3': {'a3': 'v3'}, 'a2': 'v2'},
     'c1': {'a1': 'v1'},
     'lists': [{'c1': {'inner': 'val'}}, 's2', 's3'],
     'nominal': 'v1',
     'numeric': 2.3,
     ('relation1', 'c1', '?c2'): True,
     ('relation2', ('a1', 'c1'), ('relation3', ('a3', ('?c3', '?c2')))): 4.3,
     ('relation4', '?c2', '?c4'): True}
    """
    def __init__(self, gensym=None):
        self.reverse_mapping = None
        if gensym:
            self.gensym = gensym
        else:
            self.gensym = default_gensym

    def transform(self, instance):
        """
        Performs the standardize apart tranformation.
        """
        mapping = {}
        new_instance = self._standardize(instance, mapping)
        self.reverse_mapping = {mapping[o]: o for o in mapping}
        return new_instance

    def undo_transform(self, instance):
        """
        Undoes the standardize apart tranformation.
        """
        if self.reverse_mapping is None:
            raise Exception("Must call transform before undo_transform!")

        return self._undo_standardize(instance)

    def _undo_standardize(self, instance):
        new_instance = {}

        for attr in instance:

            name = attr
            if attr in self.reverse_mapping:
                name = self.reverse_mapping[attr]
                if isinstance(name, tuple):
                    name = name[0]

            if isinstance(instance[attr], dict):
                new_instance[name] = self._undo_standardize(instance[attr])
            elif isinstance(instance[attr], list):
                new_instance[name] = [self._undo_standardize(ele) if
                                      isinstance(ele, dict) else ele for ele in
                                      instance[attr]]
            elif isinstance(attr, tuple):
                temp_rel = rename_relation(attr, self.reverse_mapping)
                new_instance[temp_rel] = instance[attr]
            else:
                new_instance[attr] = instance[attr]

        return new_instance

    def _standardize(self, instance, mapping={}, prefix=None):
        """
        Given an instance rename all the components so they
        have unique names.

        .. :warning: relations cannot have dictionaries as values (i.e., canno
            be subojects).
        .. :warning: relations can only exist at the top level, not in
            sub-objects.

        This will rename component attirbutes as well as any occurance of the
        component's name within relation attributes. This renaming is necessary
        to allow for a search between possible mappings without collisions.

        :param instance: An instance to be named apart.
        :param mapping: An existing mapping to add new mappings to; used for
            recursive calls.
        :type instance: :ref:`Instance<instance-rep>`
        :return: an instance with component attributes renamed
        :rtype: :ref:`Instance<instance-rep>`

        # Reset the symbol generator for doctesting purposes.
        >>> _reset_gensym()
        >>> import pprint
        >>> instance = {'nominal': 'v1', 'numeric': 2.3, '?c1': {'a1': 'v1'}, 'c2': {'a2': 'v2', 'c3': {'a3': 'v3'}}, '(relation1 ?c1 c2)': True, 'lists': ['s1', 's2', 's3'], '(relation2 (a1 ?c1) (relation3 (a3 (c2 c3))))': 4.3}
        >>> tuplizer = Tuplizer()
        >>> instance = tuplizer.transform(instance)
        >>> std = NameStandardizer()
        >>> standard = std.transform(instance)
        >>> pprint.pprint(standard)
        {'?o1': {'a1': 'v1'},
         'c2': {'a2': 'v2', 'c3': {'a3': 'v3'}},
         'lists': ['s1', 's2', 's3'],
         'nominal': 'v1',
         'numeric': 2.3,
         ('relation1', '?o1', 'c2'): True,
         ('relation2', ('a1', '?o1'), ('relation3', ('a3', ('c2', 'c3')))): 4.3}
        """
        new_instance = {}
        relations = []

        # I had to add the key function to the sort because python apparently
        # can't naturally sort strings and tuples
        # for attr in instance:
        for attr in sorted(instance, key=lambda at: str(at)):

            name = attr
            indexable = False
            try:
                attr[0] == '?'
                indexable = True
            except:
                pass

            if indexable and attr[0] == '?' and not isinstance(attr, tuple):
                if name not in mapping:
                    mapping[name] = self.gensym()
                name = mapping[name]

            value = instance[attr]

            if isinstance(value, dict):
                value = self._standardize(value, mapping, name)
            elif isinstance(value, list):
                value = [self._standardize(ele, mapping, name) if
                         isinstance(ele, dict) else ele for ele in value]

            if isinstance(name, tuple):
                for o in get_attribute_components(name):
                    if o not in mapping:
                        mapping[o] = self.gensym()
                relations.append((name, value))
            else:
                new_instance[name] = value

        for relation, val in relations:
            temp_rel = rename_relation(relation, mapping)
            new_instance[temp_rel] = val

        return new_instance


class Flattener(Preprocessor):
    """
    Flattens subobject attributes.

    Takes an instance that has already been standardized apart and flattens it.

    .. :warning: important to note that relations can only exist at the top
        level, not within subobjects. If they do exist than this function will
        return incorrect results.

    Hierarchy is represented with periods between variable names in the
    flattened attributes. However, this process converts the attributes with
    periods in them into a tuple of objects with an attribute as the last
    element, this is more efficient for later processing.

    This is the third and final operation in :class:`StructureMapper
    <concept_formation.structure_mapper.StructureMapper>`'s standard
    pipeline.

    >>> import pprint
    >>> flattener = Flattener()
    >>> instance = {'a': 1, 'c1': {'b': 1, '_c': 2}}
    >>> pprint.pprint(instance)
    {'a': 1, 'c1': {'_c': 2, 'b': 1}}
    >>> instance = flattener.transform(instance)
    >>> pprint.pprint(instance)
    {'a': 1, ('_', ('_c', 'c1')): 2, ('b', 'c1'): 1}
    >>> instance = flattener.undo_transform(instance)
    >>> pprint.pprint(instance)
    {'a': 1, 'c1': {'_c': 2, 'b': 1}}

    >>> instance = {'l1': {'l2': {'l3': {'l4': 1}}}}
    >>> pprint.pprint(instance)
    {'l1': {'l2': {'l3': {'l4': 1}}}}
    >>> instance = flattener.transform(instance)
    >>> pprint.pprint(instance)
    {('l4', ('l3', ('l2', 'l1'))): 1}
    >>> instance = flattener.undo_transform(instance)
    >>> pprint.pprint(instance)
    {'l1': {'l2': {'l3': {'l4': 1}}}}
    """

    def transform(self, instance):
        """
        Perform the flattening procedure.
        """
        return self._flatten(instance)

    def undo_transform(self, instance):
        """
        Undo the flattening procedure.
        """
        return self._structurize(instance)

    def _get_path(self, attr):
        """
        Unfolds a flattened attr to get the path

        >>> import pprint
        >>> flattener = Flattener()
        >>> attribute = ('x', ('o1', ('o2', 'o3')))
        >>> path = flattener._get_path(attribute)
        >>> pprint.pprint(path)
        ['o3', 'o2', 'o1', 'x']
        """
        path = []
        curr = attr
        while isinstance(curr, tuple) and len(curr) == 2:
            if curr[0] == '_':
                _, curr = curr
            else:
                a, curr = curr
                path.append(a)
        path.append(curr)
        path.reverse()
        return path

    def _structurize(self, instance):
        """
        This undoes the flattening process. In particular, it takes an instance
        that has unary relations and it unpacks them into structured objects
        and returns the fully structured object.

        >>> import pprint
        >>> flattener = Flattener()
        >>> instance = {('l4', ('l3', ('l2', 'l1'))): 1}
        >>> pprint.pprint(instance)
        {('l4', ('l3', ('l2', 'l1'))): 1}
        >>> instance = flattener._structurize(instance)
        >>> pprint.pprint(instance)
        {'l1': {'l2': {'l3': {'l4': 1}}}}
        """
        temp = {}
        for attr in instance:
            if (isinstance(attr, tuple) and len(attr) == 2):
                path = self._get_path(attr)
                curr = temp
                for sa in path[:-1]:
                    if sa not in curr:
                        curr[sa] = {}
                    curr = curr[sa]
                curr[path[-1]] = instance[attr]
            else:
                temp[attr] = instance[attr]

        return temp

    def _flatten(self, instance, outer_attr=None):
        """
        Takes an instance with dictionary attributes and and flattens it, so
        that there are no more dictionary attributes.

        .. :warning: important to note that relations can only exist at the top
            level, not within subobjects. If they do exist than this function
            will return incorrect results.

        To eliminate structure, the inner most attributes are pulled up to the
        top level and renamed as tuples that contain information about the
        structure.

        :param instance: An instance to be flattened.
        :type instance: instance
        :return: A copy of the instance flattend
        :rtype: :ref:`flattened instance <flattened-instance>`

        >>> import pprint
        >>> flattener = Flattener()
        >>> instance = {'a': 1, 'c1': {'b': 1, '_c': 2}}
        >>> flat = flattener.transform(instance)
        >>> pprint.pprint(flat)
        {'a': 1, ('_', ('_c', 'c1')): 2, ('b', 'c1'): 1}
        >>> instance = {'l1': {'l2': {'l3': {'l4': 1}}}}
        >>> pprint.pprint(instance)
        {'l1': {'l2': {'l3': {'l4': 1}}}}
        >>> instance = flattener._flatten(instance)
        >>> pprint.pprint(instance)
        {('l4', ('l3', ('l2', 'l1'))): 1}
        >>> instance = {'?check0': {'Position': {'X': 1.5990001, 'Y': -7.05200052}, 'Type': 'Checkpoint'}, '?cube01': {'Bounds': {'X': 1.7420001, 'Y': 1.751}, 'Name': 'cube01', 'Position': {'X': -12.0840006, 'Y': -7.1050005}, 'Rotation': {'Z': 0.0}, 'Type': 'cube'}, '?cube02': {'Bounds': {'X': 1.7420001, 'Y': 1.751}, 'Name': 'cube02', 'Position': {'X': -4.662, 'Y': -7.1050005}, 'Rotation': {'Z': 0.0}, 'Type': 'cube'}, 'Goal': {'Position': {'X': 8.599, 'Y': 0.715000033}, 'Type': 'Goal'}}
        >>> flat = flattener.transform(instance)
        >>> pprint.pprint(flat)
        {('Name', '?cube01'): 'cube01',
         ('Name', '?cube02'): 'cube02',
         ('Type', '?check0'): 'Checkpoint',
         ('Type', '?cube01'): 'cube',
         ('Type', '?cube02'): 'cube',
         ('Type', 'Goal'): 'Goal',
         ('X', ('Bounds', '?cube01')): 1.7420001,
         ('X', ('Bounds', '?cube02')): 1.7420001,
         ('X', ('Position', '?check0')): 1.5990001,
         ('X', ('Position', '?cube01')): -12.0840006,
         ('X', ('Position', '?cube02')): -4.662,
         ('X', ('Position', 'Goal')): 8.599,
         ('Y', ('Bounds', '?cube01')): 1.751,
         ('Y', ('Bounds', '?cube02')): 1.751,
         ('Y', ('Position', '?check0')): -7.05200052,
         ('Y', ('Position', '?cube01')): -7.1050005,
         ('Y', ('Position', '?cube02')): -7.1050005,
         ('Y', ('Position', 'Goal')): 0.715000033,
         ('Z', ('Rotation', '?cube01')): 0.0,
         ('Z', ('Rotation', '?cube02')): 0.0}
        """
        temp = {}
        for attr in instance:
            original = attr
            if outer_attr is not None:
                if attr[0] == "_":
                    attr = ('_', (attr, outer_attr))
                else:
                    attr = (attr, outer_attr)

            if isinstance(instance[original], dict):
                so = self._flatten(instance[original], attr)
                for so_attr in so:
                    temp[so_attr] = so[so_attr]
            else:
                temp[attr] = instance[original]
        return temp


class ListProcessor(Preprocessor):
    """
    Preprocesses out the lists, converting them into objects and relations.

    This preprocessor is a pipeline of two operations. First it extracts
    elements from any lists in the instance and makes them their own
    subcomponents with unique names. Second it removes the lists altogether and
    replaces them with a series of relations that both express that
    subcomponents are elments of the list and the order that they existed in.
    These two operations transform the list in a way that preserves the
    semenatics of the original list but makes them compatible with Trestle's
    understanding of component objects.

    None of the list operations are part of :class:`StructureMapper
    <concept_formation.structure_mapper.StructureMapper>`'s standard
    pipeline.

    .. warning:: The ListProcessor's undo_transform function is not
        guaranteed to be deterministic and attempts a best guess at a partial
        ordering.  In most cases this will be fine but in complex instances
        with multiple lists and user defined ordering relations it can break
        down. If an ordering cannot be determined then ordering relations are
        left in place.

    # Reset the symbol generator for doctesting purposes.
    >>> _reset_gensym()
    >>> import pprint
    >>> instance = {"att1": "val1", "list1":["a", "b", "a", "c", "d"]}
    >>> lp = ListProcessor()
    >>> instance = lp.transform(instance)
    >>> pprint.pprint(instance)
    {'?o1': {'val': 'a'},
     '?o2': {'val': 'b'},
     '?o3': {'val': 'a'},
     '?o4': {'val': 'c'},
     '?o5': {'val': 'd'},
     'att1': 'val1',
     'list1': {},
     ('has-element', 'list1', '?o1'): True,
     ('has-element', 'list1', '?o2'): True,
     ('has-element', 'list1', '?o3'): True,
     ('has-element', 'list1', '?o4'): True,
     ('has-element', 'list1', '?o5'): True,
     ('ordered-list', 'list1', '?o1', '?o2'): True,
     ('ordered-list', 'list1', '?o2', '?o3'): True,
     ('ordered-list', 'list1', '?o3', '?o4'): True,
     ('ordered-list', 'list1', '?o4', '?o5'): True}

    >>> instance = lp.undo_transform(instance)
    >>> pprint.pprint(instance)
    {'att1': 'val1', 'list1': ['a', 'b', 'a', 'c', 'd']}

    # Reset the symbol generator for doctesting purposes.
    >>> _reset_gensym()
    >>> instance = {'l1': ['a', {'in1': 3, 'in2': 4}, {'ag': 'b', 'ah': 'c'}, 12, 'again']}
    >>> lp = ListProcessor()
    >>> instance = lp.transform(instance)
    >>> pprint.pprint(instance)
    {'?o1': {'val': 'a'},
     '?o2': {'in1': 3, 'in2': 4},
     '?o3': {'ag': 'b', 'ah': 'c'},
     '?o4': {'val': 12},
     '?o5': {'val': 'again'},
     'l1': {},
     ('has-element', 'l1', '?o1'): True,
     ('has-element', 'l1', '?o2'): True,
     ('has-element', 'l1', '?o3'): True,
     ('has-element', 'l1', '?o4'): True,
     ('has-element', 'l1', '?o5'): True,
     ('ordered-list', 'l1', '?o1', '?o2'): True,
     ('ordered-list', 'l1', '?o2', '?o3'): True,
     ('ordered-list', 'l1', '?o3', '?o4'): True,
     ('ordered-list', 'l1', '?o4', '?o5'): True}

    >>> instance = lp.undo_transform(instance)
    >>> pprint.pprint(instance)
    {'l1': ['a', {'in1': 3, 'in2': 4}, {'ag': 'b', 'ah': 'c'}, 12, 'again']}

    # Reset the symbol generator for doctesting purposes.
    >>> _reset_gensym()
    >>> instance = {'tta': 'alpha', 'ttb':{'tlist': ['a', 'b', {'sub-a': 'c', 'sub-sub': {'s': 'd', 'sslist': ['w', 'x', 'y', {'issue': 'here'}]}}, 'g']}}
    >>> pprint.pprint(instance)
    {'tta': 'alpha',
     'ttb': {'tlist': ['a',
                       'b',
                       {'sub-a': 'c',
                        'sub-sub': {'s': 'd',
                                    'sslist': ['w', 'x', 'y', {'issue': 'here'}]}},
                       'g']}}

    >>> lp = ListProcessor()
    >>> instance = lp.transform(instance)
    >>> pprint.pprint(instance)
    {'tta': 'alpha',
     'ttb': {'?o1': {'val': 'a'},
             '?o2': {'val': 'b'},
             '?o3': {'sub-a': 'c',
                     'sub-sub': {'?o4': {'val': 'w'},
                                 '?o5': {'val': 'x'},
                                 '?o6': {'val': 'y'},
                                 '?o7': {'issue': 'here'},
                                 's': 'd',
                                 'sslist': {}}},
             '?o8': {'val': 'g'},
             'tlist': {}},
     ('has-element', ('sslist', ('sub-sub', ('?o3', 'ttb'))), '?o4'): True,
     ('has-element', ('sslist', ('sub-sub', ('?o3', 'ttb'))), '?o5'): True,
     ('has-element', ('sslist', ('sub-sub', ('?o3', 'ttb'))), '?o6'): True,
     ('has-element', ('sslist', ('sub-sub', ('?o3', 'ttb'))), '?o7'): True,
     ('has-element', ('tlist', 'ttb'), '?o1'): True,
     ('has-element', ('tlist', 'ttb'), '?o2'): True,
     ('has-element', ('tlist', 'ttb'), '?o3'): True,
     ('has-element', ('tlist', 'ttb'), '?o8'): True,
     ('ordered-list', ('sslist', ('sub-sub', ('?o3', 'ttb'))), '?o4', '?o5'): True,
     ('ordered-list', ('sslist', ('sub-sub', ('?o3', 'ttb'))), '?o5', '?o6'): True,
     ('ordered-list', ('sslist', ('sub-sub', ('?o3', 'ttb'))), '?o6', '?o7'): True,
     ('ordered-list', ('tlist', 'ttb'), '?o1', '?o2'): True,
     ('ordered-list', ('tlist', 'ttb'), '?o2', '?o3'): True,
     ('ordered-list', ('tlist', 'ttb'), '?o3', '?o8'): True}

    >>> instance = lp.undo_transform(instance)
    >>> pprint.pprint(instance)
    {'tta': 'alpha',
     'ttb': {'tlist': ['a',
                       'b',
                       {'sub-a': 'c',
                        'sub-sub': {'s': 'd',
                                    'sslist': ['w', 'x', 'y', {'issue': 'here'}]}},
                       'g']}}

    """
    def __init__(self):
        self.processor = Pipeline(ExtractListElements(), ListsToRelations())

    def transform(self, instance):
        """
        Extract list elements and replace lists with ordering relations.
        """
        return self.processor.transform(instance)

    def undo_transform(self, instance):
        """
        Attempt to reconstruct lists from ordering relations and add extracted
        list elements back to constructed lists.
        """
        return self.processor.undo_transform(instance)


class ExtractListElements(Preprocessor):
    """
    A pre-processor that extracts the elements of lists into their own objects

    Find all lists in an instance and extract their elements into their own
    subjects of the main instance.

    This is a first subprocess of the :class:`ListProcessor
    <concept_formation.preprocessor.ListProcessor>`. None of the list operations
    are part of :class:`StructureMapper
    <concept_formation.structure_mapper.StructureMapper>`'s standard pipeline.

    # Reset the symbol generator for doctesting purposes.
    >>> _reset_gensym()
    >>> import pprint
    >>> instance = {"a": "n", "list1": ["test", {"p": "q", "j": "k"}, {"n": "m"}]}
    >>> pp = ExtractListElements()
    >>> instance = pp.transform(instance)
    >>> pprint.pprint(instance)
    {'?o1': {'val': 'test'},
     '?o2': {'j': 'k', 'p': 'q'},
     '?o3': {'n': 'm'},
     'a': 'n',
     'list1': ['?o1', '?o2', '?o3']}

    # Reset the symbol generator for doctesting purposes.
    >>> _reset_gensym()
    >>> import pprint
    >>> instance = {"att1": "V1", 'subobj': {"list1": ["a", "b", "c", {"B": "C", "D": "E"}]}}
    >>> pprint.pprint(instance)
    {'att1': 'V1', 'subobj': {'list1': ['a', 'b', 'c', {'B': 'C', 'D': 'E'}]}}
    >>> pp = ExtractListElements()
    >>> instance = pp.transform(instance)
    >>> pprint.pprint(instance)
    {'att1': 'V1',
     'subobj': {'?o1': {'val': 'a'},
                '?o2': {'val': 'b'},
                '?o3': {'val': 'c'},
                '?o4': {'B': 'C', 'D': 'E'},
                'list1': ['?o1', '?o2', '?o3', '?o4']}}
    >>> instance = pp.undo_transform(instance)
    >>> pprint.pprint(instance)
    {'att1': 'V1', 'subobj': {'list1': ['a', 'b', 'c', {'B': 'C', 'D': 'E'}]}}

    """
    def __init__(self, gensym=None):
        if gensym:
            self.gensym = gensym
        else:
            self.gensym = default_gensym

    def transform(self, instance):
        """
        Find all lists in an instance and extract their elements into their own
        subjects of the main instance.
        """
        new_instance = self._extract(instance)
        return new_instance

    def undo_transform(self, instance):
        """
        Undoes the list element extraction operation.
        """
        return self._undo_extract(instance)

    def _undo_extract(self, instance):
        """
        Reverses the list element extraction process
        """
        new_instance = {}
        lists = {}
        elements = {}

        for a in instance:
            if isinstance(instance[a], list):
                lists[a] = True
                new_list = []
                for i in range(len(instance[a])):
                    elements[instance[a][i]] = True
                    obj = self._undo_extract(instance[instance[a][i]])

                    if "val" not in obj:
                        new_list.append(obj)
                    else:
                        new_list.append(obj["val"])
                new_instance[a] = new_list

        for a in instance:
            if isinstance(instance[a], list) or a in elements:
                continue
            elif isinstance(instance[a], dict):
                new_instance[a] = self._undo_extract(instance[a])
            else:
                new_instance[a] = instance[a]

        return new_instance

    def _extract(self, instance):
        """
        Unlike the utils.extract_components function this one will extract ALL
        elements into their own objects not just object literals
        """
        new_instance = {}
        for a in instance.keys():
            if isinstance(instance[a], list):

                if a[0] == '_':
                    new_instance[a] = str(instance[a])
                    continue

                new_list = []
                for el in instance[a]:

                    if isinstance(el, dict):
                        new_obj = deepcopy(el)
                    else:
                        new_obj = {"val": el}

                    new_att = self.gensym()
                    new_instance[new_att] = self._extract(new_obj)
                    new_list.append(new_att)

                new_instance[a] = new_list

            elif isinstance(instance[a], dict):
                new_instance[a] = self._extract(instance[a])
            else:
                new_instance[a] = instance[a]

        return new_instance


class ListsToRelations(Preprocessor):
    """
    Converts an object with lists into an object with sub-objects and list
    relations.

    This is a second subprocess of the :class:`ListProcessor
    <concept_formation.preprocessor.ListProcessor>`. None of the list
    operations are part of :class:`StructureMapper
    <concept_formation.structure_mapper.StructureMapper>`'s standard pipeline.

    # Reset the symbol generator for doctesting purposes.
    >>> _reset_gensym()
    >>> ltr = ListsToRelations()
    >>> import pprint
    >>> instance = {"list1": ['a', 'b', 'c']}
    >>> instance = ltr.transform(instance)
    >>> pprint.pprint(instance)
    {'list1': {},
     ('has-element', 'list1', 'a'): True,
     ('has-element', 'list1', 'b'): True,
     ('has-element', 'list1', 'c'): True,
     ('ordered-list', 'list1', 'a', 'b'): True,
     ('ordered-list', 'list1', 'b', 'c'): True}

    >>> instance = {"list1": ['a', 'b', 'c'], "list2": ['w', 'x', 'y', 'z']}
    >>> instance = ltr.transform(instance)
    >>> pprint.pprint(instance)
    {'list1': {},
     'list2': {},
     ('has-element', 'list1', 'a'): True,
     ('has-element', 'list1', 'b'): True,
     ('has-element', 'list1', 'c'): True,
     ('has-element', 'list2', 'w'): True,
     ('has-element', 'list2', 'x'): True,
     ('has-element', 'list2', 'y'): True,
     ('has-element', 'list2', 'z'): True,
     ('ordered-list', 'list1', 'a', 'b'): True,
     ('ordered-list', 'list1', 'b', 'c'): True,
     ('ordered-list', 'list2', 'w', 'x'): True,
     ('ordered-list', 'list2', 'x', 'y'): True,
     ('ordered-list', 'list2', 'y', 'z'): True}

    # Reset the symbol generator for doctesting purposes.
    >>> _reset_gensym()
    >>> ltr = ListsToRelations()
    >>> import pprint
    >>> instance = {'o1': {"list1":['c','b','a']}}
    >>> instance = ltr.transform(instance)
    >>> pprint.pprint(instance)
    {'o1': {'list1': {}},
     ('has-element', ('list1', 'o1'), 'a'): True,
     ('has-element', ('list1', 'o1'), 'b'): True,
     ('has-element', ('list1', 'o1'), 'c'): True,
     ('ordered-list', ('list1', 'o1'), 'b', 'a'): True,
     ('ordered-list', ('list1', 'o1'), 'c', 'b'): True}

    >>> instance = ltr.undo_transform(instance)
    >>> pprint.pprint(instance)
    {'o1': {'list1': ['c', 'b', 'a']}}
    """
    def transform(self, instance):
        return self._lists_to_relations(instance)

    def undo_transform(self, instance):
        """
        Traverse the instance and turns each set of totally ordered list
        relations into a list.

        If there is a cycle or a partial ordering, than the relations are not
        converted and left as they are.
        """
        return self._relations_to_lists(instance)

    def _relations_to_lists(self, instance, path=None):
        new_instance = {}

        elements = {}
        order = {}
        originals = {}

        for attr in instance:
            if isinstance(attr, tuple) and (attr[0] == 'has-element'):
                rel, lname, ele = attr
                if lname not in elements:
                    elements[lname] = []
                elements[lname].append(ele)

                if lname not in originals:
                    originals[lname] = []
                originals[lname].append((attr, instance[attr]))

            elif isinstance(attr, tuple) and attr[0] == 'ordered-list':
                rel, lname, ele1, ele2 = attr

                if lname not in order:
                    order[lname] = []

                order[lname].append((ele1, ele2))

                if lname not in originals:
                    originals[lname] = []
                originals[lname].append((attr, instance[attr]))

            else:
                new_instance[attr] = instance[attr]

        for l in elements:
            new_list = [elements[l].pop(0)]
            change = True

            while len(elements[l]) > 0 and change:
                change = False

                # chain to front
                front = True
                while front is not None:
                    front = None
                    for a, b in order[l]:
                        if b == new_list[0]:
                            change = True
                            front = (a, b)
                            elements[l].remove(a)
                            new_list.insert(0, a)
                            break
                    if front is not None:
                        order[l].remove(front)

                # chain to end
                back = True
                while back is not None:
                    back = None
                    for a, b in order[l]:
                        if a == new_list[-1]:
                            change = True
                            back = (a, b)
                            elements[l].remove(b)
                            new_list.append(b)
                            break
                    if back is not None:
                        order[l].remove(back)

            if len(elements[l]) == 0:
                path = self._get_path(l)
                current = new_instance
                while len(path) > 1:
                    current = current[path.pop(0)]
                current[path[0]] = new_list
            else:
                for attr, val in originals:
                    new_instance[attr] = val

        return new_instance

    def _get_path(self, path):
        if isinstance(path, tuple) and len(path) == 2:
            return self._get_path(path[1]) + [path[0]]
        else:
            return [path]

    def _lists_to_relations(self, instance, current=None, top_level=None):
        new_instance = {}
        if top_level is None:
            top_level = new_instance

        for attr in instance.keys():
            if current is None:
                lname = attr
            else:
                lname = (attr, current)

            if isinstance(instance[attr], list):
                new_instance[attr] = {}

                for i in range(len(instance[attr])-1):
                    rel = ("ordered-list", lname, str(instance[attr][i]),
                           str(instance[attr][i+1]))
                    top_level[rel] = True

                    rel = ("has-element", lname, instance[attr][i])
                    top_level[rel] = True

                if len(instance[attr]) > 0:
                    rel = ('has-element', lname, instance[attr][-1])
                    top_level[rel] = True

            elif isinstance(instance[attr], dict):
                new_instance[attr] = self._lists_to_relations(instance[attr],
                                                              lname, top_level)
            else:
                new_instance[attr] = instance[attr]

        return new_instance


class SubComponentProcessor(Preprocessor):
    """
    Takes a flattened instance and moves sub-objects (not sub-attributes) to be
    top-level objects and adds has-component relations to preserve semantics.

    This process is primairly used to improve matching by having all sub-
    component objects exist as their own top level objects with relations
    describing their original position in the hierarchy. This allows the
    structure mapper to partially match against subobjects.

    This is the second operation in :class:`TrestleTree
    <concept_formation.trestle.TrestleTree>`'s standard pipeline (after
    flattening).

    .. warning:: This assumes that the :class:`NameStandardizer
        <concept_formation.preprocessor.NameStandardizer>` has been run on the
        instance first otherwise there can be name collisions.

    # Reset the symbol generator for doctesting purposes.
    >>> _reset_gensym()
    >>> import pprint
    >>> flattener = Flattener()
    >>> psc = SubComponentProcessor()
    >>> instance = {"a1": "v1", "?sub1": {"a2": "v2", "a3": 3},
    ...             "?sub2": {"a4": "v4", "?subsub1": {"a5": "v5", "a6": "v6"},
    ...                       "?subsub2":{"?subsubsub": {"a8": "V8"}, "a7": 7}},
    ...             ('ordered-list', ('list1', ('?o2', '?o1')), 'b', 'a'):
    ...             True}
    >>> pprint.pprint(instance)
    {'?sub1': {'a2': 'v2', 'a3': 3},
     '?sub2': {'?subsub1': {'a5': 'v5', 'a6': 'v6'},
               '?subsub2': {'?subsubsub': {'a8': 'V8'}, 'a7': 7},
               'a4': 'v4'},
     'a1': 'v1',
     ('ordered-list', ('list1', ('?o2', '?o1')), 'b', 'a'): True}
    >>> instance = psc.transform(flattener.transform(instance))
    >>> pprint.pprint(instance)
    {'a1': 'v1',
     ('a2', '?sub1'): 'v2',
     ('a3', '?sub1'): 3,
     ('a4', '?sub2'): 'v4',
     ('a5', '?subsub1'): 'v5',
     ('a6', '?subsub1'): 'v6',
     ('a7', '?subsub2'): 7,
     ('a8', '?subsubsub'): 'V8',
     ('has-component', '?o1', '?o2'): True,
     ('has-component', '?sub2', '?subsub1'): True,
     ('has-component', '?sub2', '?subsub2'): True,
     ('has-component', '?subsub2', '?subsubsub'): True,
     ('ordered-list', ('list1', '?o2'), 'b', 'a'): True}
    >>> instance = psc.undo_transform(instance)
    >>> instance = flattener.undo_transform(instance)
    >>> pprint.pprint(instance)
    {'?sub1': {'a2': 'v2', 'a3': 3},
     '?sub2': {'?subsub1': {'a5': 'v5', 'a6': 'v6'},
               '?subsub2': {'?subsubsub': {'a8': 'V8'}, 'a7': 7},
               'a4': 'v4'},
     'a1': 'v1',
     ('ordered-list', ('list1', ('?o2', '?o1')), 'b', 'a'): True}
    """

    def transform(self, instance):
        """
        Travese the instance for objects that contain subobjects and extracts
        the subobjects to be their own objects at the top level of the
        instance.
        """
        return self._extract_sub_objects(instance)

    def undo_transform(self, instance):
        """
        Removes the has-component relations by adding the elements as
        subobjects.

        If a objects is a child in multiple has-component relationships than it
        is left in relational form (i.e., it cannot be expressed in sub-object
        form).
        """
        return self._embed_sub_objects(instance)

    def _embed_sub_objects(self, instance):
        so_mapping = {attr[2]: (attr[2], attr[1]) for attr in instance
                      if (isinstance(attr, tuple) and len(attr) == 3 and
                          attr[0] == 'has-component')}

        return {self._rename_embedding(attr, so_mapping): instance[attr] for
                attr in instance if not (isinstance(attr, tuple) and len(attr)
                                         == 3 and attr[0] == 'has-component')}

    def _rename_embedding(self, attr, so_mapping):
        if attr in so_mapping:
            new_a = so_mapping[attr]
            if isinstance(new_a, tuple) and len(new_a) == 2:
                return (new_a[0], self._rename_embedding(new_a[1], so_mapping))
            return new_a

        if (isinstance(attr, tuple) and len(attr) == 2):
            return (attr[0], self._rename_embedding(attr[1], so_mapping))

        if (isinstance(attr, tuple) and len(attr) != 2):
            return tuple(self._rename_embedding(ele, so_mapping) for ele in
                         attr)

        return attr

    def _extract_sub_objects(self, instance):
        new_instance = {}
        for a in instance:
            rels = self._get_has_components(a)
            for r in rels:
                new_instance[r] = True
            new_a = self._extract_attr(a)
            new_instance[new_a] = instance[a]
        return new_instance

    def _extract_attr(self, attr):
        if isinstance(attr, tuple) and len(attr) != 2:
            return tuple([self._extract_attr(ele) for ele in attr])

        if isinstance(attr, tuple) and len(attr) == 2:
            outer, inner = attr
            if isinstance(outer, str) and len(outer) > 0 and outer[0] == "?":
                return outer
            return (outer, self._extract_attr(inner))

        return attr

    def _get_has_components(self, attr):
        if not isinstance(attr, tuple):
            return []

        if len(attr) != 2:
            relations = []
            for ele in attr:
                relations = relations + self._get_has_components(ele)
            return relations

        last_comp = None
        inner = None
        relations = []

        while len(attr) == 2:
            a, attr = attr
            if isinstance(a, str) and len(a) > 0 and a[0] == '?':
                if last_comp is not None:
                    relations.append(('has-component', inner, last_comp))
                last_comp = a
                inner = self._extract_attr(attr)

        if last_comp is not None and (isinstance(attr, str) and len(attr) > 0
                                      and attr[0] == '?'):
            relations.append(('has-component', inner, last_comp))

        return relations


class ObjectVariablizer(OneWayPreprocessor):
    """
    Converts all attributes with dictionary values into variables by adding a
    question mark.

    Attribute names beginning with `?` are treated as bindable variables while
    all other attributes names are considered constants. This process searches
    through an instances and variablizes attributes that might not have been
    defined this way in the original data.

    This is a helper function preprocessor and so is not part of
    :class:`StructureMapper
    <concept_formation.structure_mapper.StructureMapper>`'s standard pipeline.

    >>> from pprint import pprint
    >>> instance = {"ob1":{"myX":12.4,"myY":13.1,"myType":"square"},"ob2":{"myX":9.5,"myY":12.6,"myType":"rect"}}
    >>> ov = ObjectVariablizer()
    >>> instance = ov.transform(instance)
    >>> pprint(instance)
    {'?ob1': {'myType': 'square', 'myX': 12.4, 'myY': 13.1},
     '?ob2': {'myType': 'rect', 'myX': 9.5, 'myY': 12.6}}
    >>> instance = ov.undo_transform(instance)
    >>> pprint(instance)
    {'?ob1': {'myType': 'square', 'myX': 12.4, 'myY': 13.1},
     '?ob2': {'myType': 'rect', 'myX': 9.5, 'myY': 12.6}}
    >>> instance = {"p1":{"x":12,"y":3},"p2":{"x":5,"y":14},"p3":{"x":4,"y":18},"setttings":{"x_lab":"height","y_lab":"age"}}
    >>> ov = ObjectVariablizer("p1","p2","p3")
    >>> instance = ov.transform(instance)
    >>> pprint(instance)
    {'?p1': {'x': 12, 'y': 3},
     '?p2': {'x': 5, 'y': 14},
     '?p3': {'x': 4, 'y': 18},
     'setttings': {'x_lab': 'height', 'y_lab': 'age'}}

    :param attrs: A list of specific attribute names to variablize. If left
        empty then all variables will be converted.
    :type attrs: strings
    """
    def __init__(self, *attrs):
        if len(attrs) == 0:
            self.targets = None
        else:
            self.targets = attrs

    def transform(self, instance):
        """
        Variablize target attributes.
        """
        return self._variablize(instance)

    def _variablize(self, instance, mapping={}, prefix=None):
        new_instance = {}

        mapping = {}
        relations = []
        if self.targets is None:
            attrs = [k for k in instance.keys()]
        else:
            attrs = self.targets

        for attr in instance:
            if prefix is None:
                prefix = attr
            else:
                prefix = (attr, prefix)

            if isinstance(attr, tuple):
                relations.append(attr)

            elif attr in attrs and isinstance(instance[attr], dict):
                name = attr
                if attr[0] != '?':
                    name = '?' + attr
                new_instance[name] = self._variablize(instance[attr], mapping,
                                                      prefix)
            else:
                new_instance[attr] = instance[attr]

        for rel in relations:
            new_instance[rename_relation(rel, mapping)] = instance[rel]

        return new_instance


class NumericToNominal(OneWayPreprocessor):
    """
    Converts numeric values to nominal ones.

    :class:`Cobweb3 <concept_formation.cobweb3.Cobweb3Tree>` and
    :class:`Trestle <concept_formation.trestle.TrestleTree>` will treat
    anything that passes ``isinstance(instance[attr],Number)`` as a numerical
    value. Because of how they store numerical distribution information, If
    either algorithm encounts a numerical value where it previously saw a
    nominal one it will throw an error. This preprocessor is provided as a way
    to address that problem by unifying the value types of attributes across an
    instance.

    This is a helper function preprocessor and so is not part of
    :class:`StructureMapper
    <concept_formation.structure_mapper.StructureMapper>`'s standard pipeline.

    >>> import pprint
    >>> ntn = NumericToNominal()
    >>> instance = {"x":12.5,"y":9,"z":"top"}
    >>> instance = ntn.transform(instance)
    >>> pprint.pprint(instance)
    {'x': '12.5', 'y': '9', 'z': 'top'}

    >>> ntn = NumericToNominal("y")
    >>> instance = {"x":12.5,"y":9,"z":"top"}
    >>> instance = ntn.transform(instance)
    >>> pprint.pprint(instance)
    {'x': 12.5, 'y': '9', 'z': 'top'}

    :param attrs: A list of specific attributes to convert. If left empty all
        numeric values will be converted.
    :type attrs: strings
    """
    def __init__(self, *attrs):
        if len(attrs) == 0:
            self.targets = None
        else:
            self.targets = attrs

    def transform(self, instance):
        """
        Transform target attribute values to nominal if they are numeric.
        """
        if self.targets is None:
            attrs = [k for k in instance.keys()]
        else:
            attrs = self.targets

        new_instance = {}

        for a in instance:
            if a in attrs and isinstance(instance[a], Number):
                new_instance[a] = str(instance[a])
            elif isinstance(instance[a], dict):
                new_instance[a] = self.transform(instance[a])
            else:
                new_instance[a] = instance[a]
        return new_instance


class NominalToNumeric(OneWayPreprocessor):
    """
    Converts nominal values to numeric ones.

    :class:`Cobweb3 <concept_formation.cobweb3.Cobweb3Tree>` and
    :class:`Trestle <concept_formation.trestle.TrestleTree>` will treat
    anything that passes ``isinstance(instance[attr],Number)`` as a numerical
    value. Because of how they store numerical distribution information, If
    either algorithm encounts a numerical value where it previously saw a
    nominal one it will throw an error. This preprocessor is provided as a way
    to address that problem by unifying the value types of attributes across an
    instance.

    Because parsing numbers is a less automatic function than casting things to
    strings this preprocessor has an extra parameter from
    :class:`NumericToNominal`. The on_fail parameter determines what should be
    done in the event of a parsing error and provides 3 options:

    * ``'break'`` - Simply raises the ValueError that caused the problem and
      fails. **(Default)**
    * ``'drop'``  - Drops any attributes that fail to parse. They would be
      treated as missing by categorization.
    * ``'zero'``  - Replaces any problem values with ``0.0``.

    This is a helper function preprocessor and so is not part of
    :class:`StructureMapper
    <concept_formation.structure_mapper.StructureMapper>`'s standard pipeline.

    >>> import pprint
    >>> ntn = NominalToNumeric()
    >>> instance = {"a":"123","b":"12.1241","c":"134"}
    >>> instance = ntn.transform(instance)
    >>> pprint.pprint(instance)
    {'a': 123.0, 'b': 12.1241, 'c': 134.0}

    >>> ntn = NominalToNumeric(on_fail='break')
    >>> instance = {"a":"123","b":"12.1241","c":"bad"}
    >>> instance = ntn.transform(instance)
    Traceback (most recent call last):
        ...
    ValueError: could not convert string to float: 'bad'

    >>> ntn = NominalToNumeric(on_fail="drop")
    >>> instance = {"a":"123","b":"12.1241","c":"bad"}
    >>> instance = ntn.transform(instance)
    >>> pprint.pprint(instance)
    {'a': 123.0, 'b': 12.1241}

    >>> ntn = NominalToNumeric(on_fail="zero")
    >>> instance = {"a":"123","b":"12.1241","c":"bad"}
    >>> instance = ntn.transform(instance)
    >>> pprint.pprint(instance)
    {'a': 123.0, 'b': 12.1241, 'c': 0.0}

    >>> ntn = NominalToNumeric("break","a","b")
    >>> instance = {"a":"123","b":"12.1241","c":"bad"}
    >>> instance = ntn.transform(instance)
    >>> pprint.pprint(instance)
    {'a': 123.0, 'b': 12.1241, 'c': 'bad'}

    :param on_fail: defines what should be done in the event of a numerical parse error
    :type on_fail: 'break', 'drop', or 'zero'
    :param attrs: A list of specific attributes to convert. If left empty all
        non-component values will be converted.
    :type attrs: strings
    """

    def __init__(self, on_fail='break', *attrs):
        if len(attrs) == 0:
            self.targets = None
        else:
            self.targets = attrs

        if on_fail not in ["break","drop","zero"]:
            on_fail = "break"
        self.on_fail = on_fail

    def transform(self,instance):
        """
        Transform target attribute values to numeric if they are valid nominals.
        """
        if self.targets is None:
            attrs = [k for k in instance.keys()]
        else:
            attrs = self.targets

        new_instance = {}

        for a in instance:
            if a in attrs:
                try:
                    val = float(instance[a])
                except ValueError as e:
                    if self.on_fail == "break":
                        raise e
                        return None
                    elif self.on_fail == "drop":
                        continue
                    elif self.on_fail == "zero":
                        val = 0.0
                new_instance[a] = val
            elif isinstance(instance[a],dict):
                new_instance[a] = self.transform(instance[a])
            else:
                new_instance[a] = instance[a]
        
        return new_instance


class Sanitizer(OneWayPreprocessor):
    """
    This is a preprocessor that santizes instances to adhere to the general
    expectations of either Cobweb, Cobweb3 or Trestle. In general this
    means enforcing that attribute keys are either of type str or tuple and
    that relational tuples contain only values of str or tuple. The  main
    reason for having this preprocessor is because many other things are valid
    dictionary keys in python and its possible to have weird behavior as a
    result.


    >>> from pprint import pprint
    >>> instance = {'a1':'v1','a2':2,'a3':{'aa1':'1','aa2':2},1:'v2',len:'v3',('r1',2,'r3'):'v4',('r4','r5'):{'aa3':4,3:'v6'}}
    >>> pprint(instance)
    {<built-in function len>: 'v3',
     1: 'v2',
     'a1': 'v1',
     'a2': 2,
     'a3': {'aa1': '1', 'aa2': 2},
     ('r1', 2, 'r3'): 'v4',
     ('r4', 'r5'): {3: 'v6', 'aa3': 4}}
    >>> san = Sanitizer('cobweb')
    >>> inst = san.transform(instance)
    >>> pprint(inst)
    {'1': 'v2',
     '<built-in function len>': 'v3',
     'a1': 'v1',
     'a2': 2,
     'a3': "{'aa1':'1','aa2':2}",
     ('r1', 2, 'r3'): 'v4',
     ('r4', 'r5'): "{3:'v6','aa3':4}"}
    >>> san = Sanitizer('trestle')
    >>> inst = san.transform(instance)
    >>> pprint(inst)
    {'1': 'v2',
     '<built-in function len>': 'v3',
     'a1': 'v1',
     'a2': 2,
     'a3': {'aa1': '1', 'aa2': 2},
     ('r1', '2', 'r3'): 'v4',
     ('r4', 'r5'): {'3': 'v6', 'aa3': 4}}
    """

    def __init__(self,spec='trestle'):
        if spec.lower() not in ['trestle','cobweb','cobweb3']:
            raise ValueError("Invalid Spec: must be one of: 'trestle','cobweb','cobweb3'")
        self.spec = spec

    def transform(self, instance):
        return self._sanitize(instance)

    def _cob_str(self,d):
        """
        Calling str on a dictionary is not gauranteed to print its keys
        deterministically so we need this function to ensure any stringified
        subobjects will be treated the same.
        """
        if isinstance(d,str):
            return "'"+d+"'"
        if isinstance(d,dict):
            return '{'+','.join([ self._cob_str(k)+':'+self._cob_str(d[k]) for k in sorted(d.keys(),key=str)])+'}'
        else:
            return str(d)

    def _sanitize_tuple(self,tup):
        ret = []
        for v in tup:
            if isinstance(v,str):
                ret.append(v)
            elif isinstance(v,tuple):
                ret.append(self._sanitize_tuple(v))
            else:
                ret.append(str(v))
        return tuple(ret)

    def _sanitize(self, instance):
        ret = {}
        for attr in instance:
            val = instance[attr]
            if not isinstance(attr,str) and not isinstance(attr,tuple):
                if str(attr) in instance:
                    print('Santitizing',attr,'is clobbering an existing value')
                
                if self.spec == 'trestle':
                    if isinstance(val,collections.Hashable):
                        ret[str(attr)] = val
                    elif isinstance(val,dict):
                        ret[str(attr)] = self._sanitize(val)
                    else:
                        ret[str(attr)] = self._cob_str(val)
                else:
                    ret[str(attr)] = val if isinstance(val,collections.Hashable) else self._cob_str(val)
            if isinstance(attr,str):
                if self.spec == 'trestle':
                    if isinstance(val,collections.Hashable):
                        ret[attr] = val
                    elif isinstance(val,dict):
                        ret[attr] = self._sanitize(val)
                    else:
                        ret[attr] = self._cob_str(val)
                else:
                    ret[attr] = val if isinstance(val,collections.Hashable) else self._cob_str(val)
                    
            if isinstance(attr,tuple):
                if self.spec == 'trestle':
                    san_tup = self._sanitize_tuple(attr)
                    if san_tup != attr and san_tup in instance:
                        print('Sanitizing',attr,'is clobbering an existing vlaue')
                    
                    if isinstance(val,collections.Hashable):
                        ret[san_tup] = val
                    elif isinstance(val,dict):
                        ret[san_tup] = self._sanitize(val)
                    else:
                        ret[san_tup] = self._cob_str(val)
                else:
                    ret[attr] = val if isinstance(val,collections.Hashable) else self._cob_str(val)
        return ret
