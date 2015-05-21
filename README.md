# Overview

This is a Python library of algorithms that perform concept formation written by
Christopher MacLellan (http://www.christopia.net) and Erik Harpstead
(http://www.erikharpstead.net). **Note, this libary has been developed for
python 3 and is incompatible with python 2 because of the way that it treats
integer division.** If you choose to try and use this library on python 2 then
use the following imports:
```
from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division
```
as described [here](http://www.dwheeler.com/essays/python3-in-python2.html)

In this library, the
[COBWEB](http://axon.cs.byu.edu/~martinez/classes/678/Papers/Fisher_Cobweb.pdf)
and
[COBWEB/3](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.97.4676&rep=rep1&type=pdf)
algorithms are implemented. These systems accept a stream of instances, which are
represented as dictionaries of attributes and values (where values can be
nominal for COBWEB and either numeric or nominal for COBWEB/3), and learns a
concept hierarchy. The resulting hierarchy can be used for clustering and
prediction.

This library also includes
[TRESTLE](http://christopia.net/data/articles/publications/maclellan1-2015.pdf),
an extension of COBWEB and COBWEB/3 that support structured and relational data
objects. This system employs partial matching to rename new objects to align
with previous examples, then categorizes these renamed objects.

Lastly, we have extended the COBWEB/3 algorithm to support two key
improvements. First, COBWEB/3 now uses an [unbiased
estimator](https://en.wikipedia.org/wiki/Unbiased_estimation_of_standard_deviation)
to calculate the standard deviation of numeric values. This is particularly
useful in situations where the number of available data points is low. Second,
COBWEB/3 supports online normalization of the continuous values, which is useful
in situations where numeric values are on different scales and helps to 
ensure that numeric values do not impact the model more than nominal values.

# Citing this Software 

If you use this software in a scientific publiction, then we would appreciate
citation of the following paper:

MacLellan, C.J., Harpstead, E., Aleven, V., Koedinger, K.R. (2015) [TRESTLE:
Incremental Learning in Structured Domains using Partial Matching and
Categorization.](http://christopia.net/data/articles/publications/maclellan1-2015.pdf)
The Third Annual Conference on Advances in Cognitive Systems.
Atlanta, GA. May 28-31, 2015.

Bibtex entry:

```
@inproceedings{trestle:2015a,
author={MacLellan, C.J. and Harpstead, E. and Aleven, V. and Koedinger, K.R.},
title={TRESTLE: Incremental Learning in Structured Domains using Partial
       Matching and Categorization.},
booktitle = {The Annual Third Conference on Advances in Cognitive Systems},
year={2015}
}
```
