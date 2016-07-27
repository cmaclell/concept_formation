=================
Concept Formation
=================

This is a Python library of algorithms that perform concept formation written by
Christopher MacLellan (http://www.christopia.net) and Erik Harpstead
(http://www.erikharpstead.net). 

Overview
========

In this library, the `COBWEB
<http://axon.cs.byu.edu/~martinez/classes/678/Papers/Fisher_Cobweb.pdf>`_ and
`COBWEB/3
<http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.97.4676&rep=rep1&type=pdf>`_
algorithms are implemented. These systems accept a stream of instances, which
are represented as dictionaries of attributes and values (where values can be
nominal for COBWEB and either numeric or nominal for COBWEB/3), and learns a
concept hierarchy. The resulting hierarchy can be used for clustering and
prediction.

This library also includes
`TRESTLE <http://christopia.net/data/articles/publications/maclellan-trestle-2016.pdf>`_,
an extension of COBWEB and COBWEB/3 that support structured and relational data
objects. This system employs partial matching to rename new objects to align
with previous examples, then categorizes these renamed objects.

Lastly, we have extended the COBWEB/3 algorithm to support three key
improvements. First, COBWEB/3 now uses an `unbiased estimator
<https://en.wikipedia.org/wiki/Unbiased_estimation_of_standard_deviation>`_ to
calculate the standard deviation of numeric values. This is particularly useful
in situations where the number of available data points is low. Second,
COBWEB/3 supports online normalization of the continuous values, which is
useful in situations where numeric values are on different scales and helps to
ensure that numeric values do not impact the model more than nominal values.
Finally, it is assumed that there is some base noise in measuring continuous
values, this noise insures that the probability of any one value never exceeds
1, even when the standard deviation is small. 

Installation
============

You can install this software using pip::

    pip install -U concept_formation

You can install the latest version of the code directly from github::
    
    pip install -U git+https://github.com/cmaclell/concept_formation@master

Important Links
===============

- Source code: `<https://github.com/cmaclell/concept_formation>`_
- Documentation: `<http://concept-formation.readthedocs.org>`_

Examples
========

We have created a number of examples to demonstrate the basic functionality of
this library. The examples can be found 
`here <http://concept-formation.readthedocs.io/en/latest/examples.html>`_.  

Citing this Software 
====================

If you use this software in a scientific publiction, then we would appreciate
citation of the following paper:

MacLellan, C.J., Harpstead, E., Aleven, V., Koedinger K.R. (2016) `TRESTLE: A
Model of Concept Formation in Structured Domains
<http://christopia.net/media/publications/maclellan-trestle-2016.pdf>`_.
Advances in Cognitive Systems, 4, 131-150.

Bibtex entry::

    @article{trestle:2016a,
    author={MacLellan, C.J. and Harpstead, E. and Aleven, V. and Koedinger, K.R.},
    title={TRESTLE: A Model of Concept Formation in Structured Domains},
    journal={Advances in Cognitive Systems},
    volume={4},
    year={2016}
    }
