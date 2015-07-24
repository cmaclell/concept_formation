from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

_gensym_counter = 0;

def gensym():
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
        raise NotImplementedError("Class must implement transform function")

    def undo_transform(self, instance):
        """
        Undoes a transformation to an instance.
        """
        raise NotImplementedError("Class must implement undo_transform function")

class Pipeline(Preprocessor):
    """
    A special preprocessor class used to chain together many preprocessors.
    Supports the same same transform and undo_transform functions as a regular
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

    This is the first operation in :class:`StructureMapper
    <concept_formation.structure_mapper.StructureMapper>`'s standard
    pipeline.

    >>> tuplizer = Tuplizer()
    >>> instance = {'(foo1 o1 (foo2 o2 o3))': True}
    >>> print(tuplizer.transform(instance))
    {('foo1', 'o1', ('foo2', 'o2', 'o3')): True}
    >>> print(tuplizer.undo_transform(tuplizer.transform(instance)))
    {'(foo1 o1 (foo2 o2 o3))': True}
    """
    def transform(self, instance):
        """
        Convert at string specified relations into tuples.
        """
        return {self._tuplize_relation(attr): instance[attr] for attr in instance}

    def undo_transform(self, instance):
        """
        Convert tuple relations back into their string forms.
        """
        return {self._stringify_relation(attr): instance[attr] for attr in instance}

    def _tuplize_relation(self, relation):
        """
        Converts a string formatted relation into a tuplized relation. 

        :param attr: The relational attribute formatted as a string
        :type attr: string
        :param mapping: A dictionary of mappings with component names as keys. Just
            the keys are used (i.e., as a set) to determine if elements in the relation
            are objects.
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
            current.append(val)
            
            while end > 0:
                last = tuple(stack.pop())
                current = stack[-1]
                current.append(last)
                end -= 1

        final = tuple(stack[-1][-1])
        #final = self.tuplize_elements(final)
        return final

    def _stringify_relation(self, relation):
        """
        Converts a tupleized relation into a string formated relation.

        >>> relation = ('foo1', 'o1', ('foo2', 'o2', 'o3'))
        >>> tuplizer = Tuplizer()
        >>> tuplizer._stringify_relation(relation)
        '(foo1 o1 (foo2 o2 o3))'
        """
        #relation = convert_unary_to_dot(relation)
        if isinstance(relation, tuple):
            relation = [self._stringify_relation(ele) if isinstance(ele, tuple)
                        else ele for ele in relation]
            return "(" + " ".join(relation) + ")"
        else:
            return relation

class NameStandardizer(Preprocessor):
    """
    A preprocessor that standardizes apart object names.

    Given a :ref:`raw instance <raw-instance>` rename all the components so they
    have unique names.

    .. :warning: relations cannot have dictionaries as values (i.e., cannot be
        subojects).
    .. :warning: relations can only exist at the top level, not in sub-objects.

    This will rename component attributes as well as any occurance of the
    component's name within relation attributes. This renaming is necessary to
    allow for a search between possible mappings without collisions.

    This is the second operation in :class:`StructureMapper
    <concept_formation.structure_mapper.StructureMapper>`'s standard
    pipeline.

    >>> _reset_gensym()
    >>> # We reset the symbol generator for doctesting purposes. 
    >>> import pprint
    >>> instance = {'nominal': 'v1', 'numeric': 2.3, 'c1': {'a1': 'v1'}, '?c2': {'a2': 'v2', '?c3': {'a3': 'v3'}}, '(relation1 c1 ?c2)': True, 'lists': [{'c1': {'inner': 'val'}}, 's2', 's3'], '(relation2 (a1 c1) (relation3 (a3 (?c3 ?c2))))': 4.3}
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
     ('relation2', ('a1', 'c1'), ('relation3', ('a3', ('?c3', '?c2')))): 4.3}
    >>> pprint.pprint(new_i)
    {'?o1': {'?o2': {'a3': 'v3'}, 'a2': 'v2'},
     'c1': {'a1': 'v1'},
     'lists': [{'c1': {'inner': 'val'}}, 's2', 's3'],
     'nominal': 'v1',
     'numeric': 2.3,
     ('relation1', 'c1', '?o1'): True,
     ('relation2', ('a1', 'c1'), ('relation3', ('a3', '?o2'))): 4.3}
    >>> pprint.pprint(old_i)
    {'?c2': {'?c3': {'a3': 'v3'}, 'a2': 'v2'},
     'c1': {'a1': 'v1'},
     'lists': [{'c1': {'inner': 'val'}}, 's2', 's3'],
     'nominal': 'v1',
     'numeric': 2.3,
     ('relation1', 'c1', '?c2'): True,
     ('relation2', ('a1', 'c1'), ('relation3', ('a3', ('?c3', '?c2')))): 4.3}
    """

    def __init__(self):
        self.reverse_mapping = None

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
        Given a :ref:`raw instance <raw-instance>` rename all the components so they
        have unique names.

        .. :warning: relations cannot have dictionaries as values (i.e., cannot be
            subojects).
        .. :warning: relations can only exist at the top level, not in sub-objects.

        This will rename component attirbutes as well as any occurance of the
        component's name within relation attributes. This renaming is necessary to
        allow for a search between possible mappings without collisions.

        :param instance: An instance to be named apart.
        :param mapping: An existing mapping to add new mappings to; used for
            recursive calls.
        :type instance: :ref:`raw instance <raw-instance>`
        :return: an instance with component attributes renamed
        :rtype: :ref:`standardized instance <standard-instance>`

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

        # I had to add the key function to the sort because python apparently can't
        # naturally sort strings and tuples
        #for attr in instance:
        for attr in sorted(instance, key=lambda at: str(at)):

            if prefix is None:
                new_a = attr
            else:
                new_a = (attr, prefix)

            if attr[0] == '?':
                mapping[new_a] = gensym()

            if isinstance(attr, tuple):
                relations.append((attr, instance[attr]))

            elif isinstance(instance[attr], dict):
                name = attr
                if attr[0] == '?':
                    name = mapping[new_a]
                new_instance[name] = self._standardize(instance[attr],
                                                       mapping, new_a)
            elif isinstance(instance[attr], list):
                name = attr
                if attr[0] == '?':
                    name = mapping[new_a]
                new_instance[name] = [self._standardize(ele, mapping, new_a) 
                                       if isinstance(ele, dict) else ele for
                                       ele in instance[attr]]
            else:
                new_instance[attr] = instance[attr]

        for relation, val in relations:
            temp_rel = rename_relation(relation, mapping)
            new_instance[temp_rel] = val

        return new_instance

class Flattener(Preprocessor):
    """
    Flattens subobject attributes.

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
    """

    def transform(self, instance):
        return self.flatten_json(instance)

    def undo_transform(self, instance):
        return self.structurize(instance)

    def structurize(self, instance):
        """
        Undoes the flattening.
        """
        temp = {}
        for attr in instance:
            if (isinstance(attr, tuple) and len(attr) == 2): 

                if attr[0] == '_':
                    rel, (sub_attr, name) = attr
                else:
                    sub_attr, name = attr

                if name not in temp:
                    temp[name] = {}
                temp[name][sub_attr] = instance[attr]

            else:
                temp[attr] = instance[attr]

        return temp

    def flatten_json(self, instance):
        """
        Takes a :ref:`raw instance <raw-instance>` that has already been
        standardized apart and flattens it.

        .. :warning: important to note that relations can only exist at the top level,
            not within subobjects. If they do exist than this function will return
            incorrect results.

        Hierarchy is represented with periods between variable names in the
        flattened attributes. However, this process converts the attributes with
        periods in them into a tuple of objects with an attribute as the last
        element, this is more efficient for later processing.

        :param instance: An instance to be flattened.
        :type instance: :ref:`raw instance <raw-instance>`
        :return: A copy of the instance flattend
        :rtype: :ref:`flattened instance <flattened-instance>`

        >>> import pprint
        >>> flattener = Flattener()
        >>> instance = {'a': 1, 'c1': {'b': 1, '_c': 2}}
        >>> flat = flattener.flatten_json(instance)
        >>> pprint.pprint(flat)
        {'a': 1, ('_', ('_c', 'c1')): 2, ('b', 'c1'): 1}        
        """
        
        temp = {}
        for attr in instance:
            if isinstance(instance[attr], dict):
                for so_attr in instance[attr]:
                    if so_attr[0] == '_':
                        new_attr = ('_', (so_attr, attr))
                    else:
                        new_attr = (so_attr, attr)
                    temp[new_attr] = instance[attr][so_attr]
            else:
                temp[attr] = instance[attr]
        return temp


class ListProcessor(Preprocessor):
    """
    Preprocesses out the lists, converting them into objects and relations.

    >>> import pprint
    >>> _reset_gensym()
    >>> instance = {"att1":"val1","list1":["a","b","a","c","d"]}
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

    >>> _reset_gensym()
    >>> instance = {'l1':['a',{'in1':3,'in2':4},{'ag':'b','ah':'c'},12,'again']}
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

    >>> _reset_gensym()
    >>> instance = {'tta':'alpha','ttb':{'tlist':['a','b',{'sub-a':'c','sub-sub':{'s':'d','sslist':['w','x','y',{'issue':'here'}]}},'g']}}
    >>> pprint.pprint(instance)
    {'tta': 'alpha',
     'ttb': {'tlist': ['a',
                       'b',
                       {'sub-a': 'c',
                        'sub-sub': {'s': 'd',
                                    'sslist': ['w',
                                               'x',
                                               'y',
                                               {'issue': 'here'}]}},
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
                                    'sslist': ['w',
                                               'x',
                                               'y',
                                               {'issue': 'here'}]}},
                       'g']}}

    """
    def __init__(self):
        self.processor = Pipeline(ExtractListElements(), ListsToRelations())

    def transform(self, instance):
        return self.processor.transform(instance)        

    def undo_transform(self, instance):
        return self.processor.undo_transform(instance)

class ExtractListElements(Preprocessor):
    """
    A pre-processor that extracts the elements of lists into their own objects

    >>> _reset_gensym()
    >>> import pprint
    >>> instance = {"att1":"V1",'subobj':{"list1":["a","b","c",{"B":"C","D":"E"}]}}
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
    def transform(self, instance):
        """
        Peforms the list element extraction operation.
        """
        new_instance = self.extract(instance)
        return new_instance

    def undo_transform(self, instance):
        """
        Undoes the list element extraction operation.
        """
        return self.undo_extract(instance)

    def undo_extract(self,instance):
        """
        Reverses the list element extraction process
        """
        new_instance = {}
        lists = {}
        elements = {}

        for a in instance:
            if isinstance(instance[a],list):
                lists[a] = True
                new_list = []
                for i in range(len(instance[a])):
                    elements[instance[a][i]] = True
                    obj = self.undo_extract(instance[instance[a][i]])

                    if "val" not in obj:
                        new_list.append(obj)
                    else:
                        new_list.append(obj["val"])
                new_instance[a] = new_list

        for a in instance:
            if isinstance(instance[a],list) or a in elements:
                continue
            elif isinstance(instance[a], dict):
                new_instance[a] = self.undo_extract(instance[a])
            else:
                new_instance[a] = instance[a]

        return new_instance

    def extract(self,instance):
        """
        Find all lists in an instance and extract their elements into their own
        subjects of the main instance.

        Unlike the utils.extract_components function this one will extract ALL
        elements into their own objects not just object literals

        >>> _reset_gensym()
        >>> import pprint
        >>> instance = {"a":"n","list1":["test",{"p":"q","j":"k"},{"n":"m"}]}
        >>> pp = ExtractListElements()
        >>> instance = pp.extract(instance)
        >>> pprint.pprint(instance)
        {'?o1': {'val': 'test'},
         '?o2': {'j': 'k', 'p': 'q'},
         '?o3': {'n': 'm'},
         'a': 'n',
         'list1': ['?o1', '?o2', '?o3']}
        """
        new_instance = {}
        for a in instance.keys():
            if isinstance(instance[a],list):

                if a[0] == '_':
                    new_instance[a] = str(instance[a])
                    continue

                new_list = []
                for el in instance[a]:
                    
                    # TODO do we want to deep copy in the case we find a dict?
                    if isinstance(el,dict):
                        new_obj = el
                    else :
                        new_obj = {"val": el}

                    new_att = gensym()
                    new_instance[new_att] = self.extract(new_obj)
                    new_list.append(new_att)

                new_instance[a] = new_list

            elif isinstance(instance[a],dict):
                new_instance[a] = self.extract(instance[a])
            else :
                new_instance[a] = instance[a]

        return new_instance

class ListsToRelations(Preprocessor):
    """
    Converts an object with lists into an object with sub-objects and list
    relations.

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
        return self.lists_to_relations(instance)

    def undo_transform(self, instance):
        return self.relations_to_lists(instance)

    def relations_to_lists(self, instance, path=None):
        """
        Traverse the instance and turns each set of totally ordered list
        relations into a list.
        
        If there is a cycle or a partial ordering, than the relations are not
        converted and left as they are. 
        """
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
                    for a,b in order[l]:
                        if b == new_list[0]:
                            change = True
                            front = (a,b)
                            elements[l].remove(a)
                            new_list.insert(0, a)
                            break
                    if front is not None:
                        order[l].remove(front)
                
                # chain to end
                back = True
                while back is not None:
                    back = None
                    for a,b in order[l]:
                        if a == new_list[-1]:
                            change = True
                            back = (a,b)
                            elements[l].remove(b)
                            new_list.append(b)
                            break
                    if back is not None:
                        order[l].remove(back)
            
            if len(elements[l]) == 0:
                path = self.get_path(l)
                current = new_instance
                while len(path) > 1:
                    current = current[path.pop(0)]
                current[path[0]] = new_list
            else:
                for attr, val in originals:
                    new_instance[attr] = val

        return new_instance

    def get_path(self, path):
        if isinstance(path, tuple) and len(path) == 2:
            return self.get_path(path[1]) + [path[0]]
        else:
            return [path]

    def lists_to_relations(self, instance, current=None, top_level=None):
        """
        Travese the instance and turn any list elements into 
        a series of relations.

        >>> _reset_gensym()
        >>> ltr = ListsToRelations()
        >>> import pprint
        >>> instance = {"list1":['a','b','c']}
        >>> instance = ltr.lists_to_relations(instance)
        >>> pprint.pprint(instance)
        {'list1': {},
         ('has-element', 'list1', 'a'): True,
         ('has-element', 'list1', 'b'): True,
         ('has-element', 'list1', 'c'): True,
         ('ordered-list', 'list1', 'a', 'b'): True,
         ('ordered-list', 'list1', 'b', 'c'): True}
        
        >>> instance = {"list1":['a','b','c'],"list2":['w','x','y','z']}
        >>> instance = ltr.lists_to_relations(instance)
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
        """

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

                #if isinstance(lname, tuple):
                #    rel = ('has-component', current, lname)
                #    top_level[rel] = True

            elif isinstance(instance[attr],dict):
                new_instance[attr] = self.lists_to_relations(instance[attr],
                                                        lname,
                                                        top_level)
            else:
                new_instance[attr] = instance[attr]
        
        return new_instance

class SubComponentProcessor(Preprocessor):
    """
    Removes sub-objects and add has-component relations.

    >>> _reset_gensym()
    >>> import pprint
    >>> psc = SubComponentProcessor()
    >>> instance = {"a1":"v1","sub1":{"a2":"v2","a3":3},"sub2":{"a4":"v4","subsub1":{"a5":"v5","a6":"v6"},"subsub2":{"subsubsub":{"a8":"V8"},"a7":7}}}
    >>> pprint.pprint(instance)
    {'a1': 'v1',
     'sub1': {'a2': 'v2', 'a3': 3},
     'sub2': {'a4': 'v4',
              'subsub1': {'a5': 'v5', 'a6': 'v6'},
              'subsub2': {'a7': 7, 'subsubsub': {'a8': 'V8'}}}}
    >>> instance = psc.transform(instance)
    >>> pprint.pprint(instance)
    {'a1': 'v1',
     'sub1': {'a2': 'v2', 'a3': 3},
     'sub2': {'a4': 'v4'},
     'subsub1': {'a5': 'v5', 'a6': 'v6'},
     'subsub2': {'a7': 7},
     'subsubsub': {'a8': 'V8'},
     ('has-component', 'sub2', 'subsub1'): True,
     ('has-component', 'sub2', 'subsub2'): True,
     ('has-component', 'subsub2', 'subsubsub'): True}
    >>> instance = psc.undo_transform(instance)
    >>> pprint.pprint(instance)
    {'a1': 'v1',
     'sub1': {'a2': 'v2', 'a3': 3},
     'sub2': {'a4': 'v4',
              'subsub1': {'a5': 'v5', 'a6': 'v6'},
              'subsub2': {'a7': 7, 'subsubsub': {'a8': 'V8'}}}}
    """
    
    def transform(self, instance):
        return self.hoist_sub_objects(instance)

    def undo_transform(self, instance):
        return self.add_sub_objects(instance)

    def add_sub_objects(self, instance):
        """
        Removes the has-component relations by adding the elements as
        subobjects.

        If a objects is a child in multiple has-component relationships than it
        is left in relational form (i.e., it cannot be expressed in sub-object
        form).
        """
        new_instance = {}

        parents = {}
        children = {}
        leave_alone = set()

        for attr in instance:
            if isinstance(attr, tuple) and attr[0] == 'has-component':
                rel, parent, child = attr
                if child in leave_alone:
                    new_instance[attr] = instance[attr]
                elif child in parents:
                    new_instance[attr] = instance[attr]
                    rel = ('has-component', parents[child],
                           children[parents[child]])
                    new_instance[rel] = True
                    leave_alone.add(child)

                    p = parents[child]
                    del children[p]
                    del parents[child]
                else:
                    parents[child] = parent
                    children[parent] = child
            else:
                new_instance[attr] = copy.deepcopy(instance[attr])

        while True:
            child = None
            for c in parents:
                if c not in children:
                    child = c
                    break
            if child is not None:
                new_instance[parents[child]][child] = new_instance[child]
                del new_instance[child]
                dlist = [ele for ele in children if children[ele] == child]
                for ele in dlist:
                    del children[ele]
                del parents[child]
            else:
                break

        return new_instance

    def hoist_sub_objects(self, instance):
        """
        Travese the instance for objects that contain subobjects and hoists the
        subobjects to be their own objects at the top level of the instance. 
        
        >>> _reset_gensym()
        >>> import pprint
        >>> psc = SubComponentProcessor()
        >>> instance = {"a1":"v1","sub1":{"a2":"v2","a3":3},"sub2":{"a4":"v4","subsub1":{"a5":"v5","a6":"v6"},"subsub2":{"subsubsub":{"a8":"V8"},"a7":7}}}
        >>> pprint.pprint(instance)
        {'a1': 'v1',
         'sub1': {'a2': 'v2', 'a3': 3},
         'sub2': {'a4': 'v4',
                  'subsub1': {'a5': 'v5', 'a6': 'v6'},
                  'subsub2': {'a7': 7, 'subsubsub': {'a8': 'V8'}}}}

        >>> instance = psc.hoist_sub_objects(instance)
        >>> pprint.pprint(instance)
        {'a1': 'v1',
         'sub1': {'a2': 'v2', 'a3': 3},
         'sub2': {'a4': 'v4'},
         'subsub1': {'a5': 'v5', 'a6': 'v6'},
         'subsub2': {'a7': 7},
         'subsubsub': {'a8': 'V8'},
         ('has-component', 'sub2', 'subsub1'): True,
         ('has-component', 'sub2', 'subsub2'): True,
         ('has-component', 'subsub2', 'subsubsub'): True}
        """
        new_instance = {}
        
        for a in instance.keys() :
            # this is a subobject
            if isinstance(instance[a],dict):
                new_instance[a] = self._hoist_sub_objects_rec(instance[a],a,new_instance)
            else :
                new_instance[a] = instance[a]

        return new_instance

    def _hoist_sub_objects_rec(self, sub, attr, top_level):
        """
        The recursive version of subobject hoisting.
        """
        new_sub = {}
        for a in sub.keys():
            # this is a sub-sub object
            if isinstance(sub[a],dict):
                top_level[a] = self._hoist_sub_objects_rec(sub[a],a,top_level)
                rel = ("has-component", str(attr), str(a))
                top_level[rel] = True
            else :
                new_sub[a] = sub[a]
        return new_sub



class ObjectVariablizer(Preprocessor):
    """
    Converts all attributes with dictionary values into variables by adding a
    question mark.
    """
    def transform(self, instance):
        return self.variablize(instance)

    def undo_transform(self, instance):
        raise NotImplemented("no reverse transformation currently implemented")

    def variablize(self, instance, mapping={}, prefix=None):
        new_instance = {}

        mapping = {}
        relations = []

        for attr in instance:
            if prefix == None:
                prefix = attr
            else:
                prefix = (attr, prefix)

            if isinstance(attr, tuple):
                relations.append(attr)

            elif isinstance(instance[attr], dict):
                name = attr
                if attr[0] != '?':
                    name = '?' + attr
                new_instance[name] = self.variablize(instance[attr], 
                                                     mapping, prefix)
            else:
                new_instance[attr] = instance[attr]

        for rel in relations:
            new_instance[rename_relation(rel, mapping)] = instance[rel]

        return new_instance
