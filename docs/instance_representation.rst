.. _instance-rep:

Instance Representation
=======================

The primary classes in the concept_formation package
(:class:`CobwebTree<concept_formation.cobweb.CobwebTree>`,
:class:`Cobweb3Tree<concept_formation.cobweb3.Cobweb3Tree>`, and
:class:`TrestleTree<concept_formation.trestle.TrestleTree>`) learn from
instances that are represented as python dictionaries (i.e., lists of attribute
values). This representation is different from the feature vector representation
that is used by most machine learning packages (e.g., `ScikitLearn <http
://scikit-learn.org/stable/>`__). The concept_formation package uses the
dictionary format instead of feature vectors for two reasons: dictionaries are
more human readable and dictionaries offer more flexibility in the kinds of
data that can be represented (e.g., attributes in dictionaries can have other
dictionaries as values). Furthermore, it is a general format that many other
representations, such as JSON, can be easily converted into. In fact, the
concept_formation package has methods for facilitating such conversions.

.. _attributes:

Attributes
----------

The concept_formation package supports four kinds of attributes:

.. _attr-const:

Constant Attributes
    The default attribute type. Constant attributes are typically strings but
    any attribute that does not satisfy the conditions for the other categories
    will be assumed to be constant.

.. _attr-var:

Variable Attributes
    Any attribute that can be renamed to maximize mapping between an instance
    and a concept. This allows for matching attributes based on the similarity
    of their values rather than strictly on their attribute names. Variable are
    denoted with a question mark ``'?'`` as their first element (e.g.,
    ``'?variable-attribute'``).

.. _attr-rel:

Relational Attributes
    An attribute that represents a relationship between other attributes or
    values of the instance. Relation attributes are represented as tuples (e.g.,
    ``('relation', 'obj1', 'obj2')``). Relations can only be in the top level of
    the instance (i.e., component values, described below, cannot contain
    relations). If a relationship needs to be expressed between attributes of
    component values, then preorder unary relations can be used. For example, to
    express a relationship of feature1 of subobject1 I might have:
    ``('relation', ('feature1', 'subobject1')``).

.. _attr-hid:

Hidden Attributes
    Attributes that are maintained in the concept knowledge base but are not
    considered during concept formation. These are useful for propagating unique
    ids or other bookkeeping labels into the knoweldge base without biasing
    concept formation. Hidden attributes are denoted as constant or relational
    attributes that have an ``'_'`` as their first element (i.e., ``attribute[0]
    == '_'``). For constants, this means that the first character is an
    underscore (e.g., ``"_hidden"``). For relations, this means that the first
    element in the tuple is an string underscore (e.g., 
    ``('_', 'hidden-relation', 'obj')``).

Only the **constant** and **hidden** attributes are supported by
:class:`CobwebTree<concept_formation.cobweb.CobwebTree>` and
:class:`Cobweb3Tree<concept_formation.cobweb3.Cobweb3Tree>`.
:class:`TrestleTree<concept_formation.trestle.TrestleTree>` supports all
attribute types. 

In general attribute names must be hashable (so they can be used in a
dictionary and must be zero index-able (e.g., ``attribute[0]``, so that they
can be tested to determine if they are hidden.

.. _values:

Values
------

For each of these attribute type, the concept_formation package supports three
kinds of values:

.. _val-nom:

Nominal Values
    All non-numerical values (typically strings or booleans).

.. _val-num:

Numerical Values
    All values that are recognized by Python as numbers (i.e.,
    ``isinstance(val, Number)``).

.. _val-comp:

Component Values
    All dictionary values (i.e., sub-instances). All component values are
    internally converted into unary relations, so unary relations can also be
    used directly. For example ``{'subobject: {'attr': 'value'}}`` is equivalent
    to  ``{('attr', 'subobject'): 'value'}``.  Note that sub-instances cannot
    contain relations. Instead include the relations in the top-level instance
    and use unary relations to refer to elements of sub-instances (e.g.,
    ``('relation1' ('att1', 'subobject'))``).

The :class:`CobwebTree<concept_formation.cobweb.CobwebTree>` class supports
only **nominal** values. The
:class:`Cobweb3Tree<concept_formation.cobweb3.Cobweb3Tree>` supports both
**nominal** and **numeric** values. Finally, the
:class:`TrestleTree<concept_formation.trestle.TrestleTree>` supports all value
types. 

Example Instance
----------------
    
Here is an instance that provides an example of each of these different
attribute-value type combinations:

.. ipython::

    # Data is stored in a list of dictionaries where values can be either nominal,
    # numeric, hidden, component, unbound attributes, or relational.
    In [1]: instance = {'f1': 'v1', # constant attribute with nominal value
       ...:             'f2': 2.6, # constant attribute with numerical value
       ...:             'f3': {'sub-feature1': 'v1'}, # constant attribute with component value
       ...:             '?f4': 'v1', # variable attribute with nominal value
       ...:             '?f5': 2.6, # variable attribute with numerical value
       ...:             '?f6': {'sub-feature1': 'v1'}, # variable attribute with component value
       ...:             ('some-relation', 'f3', '?f4'): True, #relation attribute with nominal value
       ...:             ('some-relation2', 'f3', '?f4'): 2.6, #relation attribute with numeric value
       ...:             ('some-relation3', 'f3', '?f4'): {'sub-feature1': 'v1'}, #relation attribute with component value
       ...:             ('some-relation4', 'f3', ('sub-feature1', '?f4')): True, # relation attribute that uses unary relation to access sub-feature1 of ?f4. It also has a nominal value.
       ...:             '_f7': 'v1', # hidden attribute with nominal value
       ...:             '_f8': 2.6, # hidden attribute with numeric value
       ...:             '_f9': {'sub-feature1': 'v1'}, # hidden attribute with component value
       ...:            }
