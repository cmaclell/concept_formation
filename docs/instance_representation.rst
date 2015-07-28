Instance Representation
=======================

The primary classes in the concept_formation package
(:class:`CobwebTree<concept_formation.cobweb.CobwebTree>`,
:class:`Cobweb3Tree<concept_formation.cobweb3.Cobweb3Tree>`, and
:class:`TrestleTree<concept_formation.trestle.TrestleTree>`) learn from
instances that are represented as python dictionaries (i.e., lists of attribute
values). This representation is different from the feature vector representation
that is used by most machine learning packages (e.g., ScikitLearn). The
concept_formation package uses the dictionary format instead of feature vectors
for two reasons: dictionaries are more human readable and dictionaries offers
more flexibility in the kinds of data that can be represented (e.g., attributes
in dictionaries can have other dictionaries as values). Furthermore, it is a
general format that many other representations, such as JSON, can be easily
converted into. In fact, the concept_formation package has methods for
facilitating such conversions.

The concept_formation package supports four kinds of attributes:

* Constants - the default attribute type (typically a string, but if the
  conditions for the other attribute types are not met than it is assumed to be
  a constant).
* Variables - constant attributes that have a question mark '?' as their first
  element (e.g., "?variable-attribute").
* Hidden - constant or relational attributes that has an '_' as their first
  element (i.e., attribute[0] == '_'). For constants, this means that the
  first character is an underscore (e.g., "_hidden"). For relations, this means
  that the first element in the tuple is an string underscore (e.g., ('_',
  'hidden-relation', 'obj')). 
* Relational - an attribute that is represented as a tuple (e.g., ('relation',
  'obj1', 'obj2')). Relations can only be in the top level of the instance
  (i.e., component values, described below, cannot contain relations). If a
  relationship needs to be expressed between attributes of component values,
  than preorder unary relations can be used. For example, to express a
  relationship of feature1 of subobject1 I might have: ('relation',
  ('feature1', 'subobject1')). 

For each of these attribute types, the concept_formation package supports three
kinds of values:

* Nominal values
* Numeric values
* Component values

The concept_formation package supports six different kinds of attribute-values:

* Nominals - Attributes with non-numerical values (e.g., a string or boolean)
* Numerics - Attributes with numerical values (i.e., isinstance(val, number)) 
* Hidden attribute names ('_') - Attribute names that have an underscore as
  their first value (i.e., attr[0] == '_'). * Components - Attributes with other instances (i.e., dictionaries) as values.
  Note that these sub-instances cannot contain relations. Instead include the
  relations in the top level instance use unary relations to refer to elements
  of sub-relations (e.g., ('att1', 'subobject')). 
* Unbound attribute names ('?')
* Relations

Here is an instance that provides an example of each of these attribute-value
types:

.. ipython::

    # Data is stored in a list of dictionaries where values can be either nominal,
    # numeric, hidden, component, unbound attributes, or relational.
    In [1]: instance = {'f1': 'v1', #nominal
       ...:             'f2': 2.6, #numeric
       ...:             '_hidden-attribute': 'v1', # hidden attribute
       ...:             'f3': {'sub-feature1': 'v1'}, # component
       ...:             '?f4': {'sub-feature1': 'v1'}, # component with unbound attribute
       ...:             ('some-relation', 'f3', '?f4'): True #relation
       ...:             ('another-relation', 'f3', ('?f4', 'sub-feature1')): True #relation that references attribute of component using unary relation
       ...:            }

Here we will describe instances. There are a couple of main points to cover:
    * Nominals
    * Numerics
    * Relations
    * Components 
    * Unbound attribute names ('?')
    * Hidden attribute names ('_')
    * What type of tree supports which format.

