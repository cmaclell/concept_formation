from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division

from concept_formation.datasets import load_forest_fires


def test_load_forest_fires():
    data = load_forest_fires(num_instances=1)
    known = {'DC': 94.3, 'DMC': 26.2, 'FFMC': 86.2, 'ISI': 5.1, 'RH': 51.0,
             'area': 0.0, 'day': 'fri', 'month': 'mar', 'rain': 0.0, 'temp':
             8.2, 'wind': 6.7, 'x-axis': 7.0, 'y-axis': 5.0}
    assert known == data[0]
