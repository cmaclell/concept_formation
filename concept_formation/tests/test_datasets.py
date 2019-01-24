from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division
import random

from concept_formation.datasets import load_iris
from concept_formation.datasets import load_mushroom
from concept_formation.datasets import load_molecule
from concept_formation.datasets import load_rb_s_07
from concept_formation.datasets import load_rb_s_13
from concept_formation.datasets import load_rb_wb_03
from concept_formation.datasets import load_rb_com_11
from concept_formation.datasets import load_quadruped
from concept_formation.datasets import load_forest_fires
from concept_formation.datasets import load_congressional_voting
from concept_formation.datasets import load_rb_s_07_human_predictions


def test_load_forest_fires():
    data = load_forest_fires(num_instances=1)
    known = {'DC': 94.3, 'DMC': 26.2, 'FFMC': 86.2, 'ISI': 5.1, 'RH': 51.0,
             'area': 0.0, 'day': 'fri', 'month': 'mar', 'rain': 0.0, 'temp':
             8.2, 'wind': 6.7, 'x-axis': 7.0, 'y-axis': 5.0}
    assert known == data[0]


def test_load_congressional_voting():
    data = load_congressional_voting(num_instances=1)
    known = {'Class Name': 'republican', 'adoption-of-the-budget-resolution':
             'n', 'aid-to-nicaraguan-contras': 'n', 'anti-satellite-test-ban':
             'n', 'crime': 'y', 'duty-free-exports': 'n', 'education-spending':
             'y', 'el-salvador-aid': 'y',
             'export-administration-act-south-africa': 'y',
             'handicapped-infants': 'n', 'immigration': 'y', 'mx-missile': 'n',
             'physician-fee-freeze': 'y', 'religious-groups-in-schools': 'y',
             'superfund-right-to-sue': 'y', 'water-project-cost-sharing': 'y'}
    assert known == data[0]


def test_load_iris():
    data = load_iris(num_instances=1)
    known = {'class': 'Iris-setosa', 'petal length': 1.4, 'petal width': 0.2,
             'sepal length': 5.1, 'sepal width': 3.5}
    assert known == data[0]


def test_load_mushroom():
    data = load_mushroom(num_instances=1)
    known = {'bruises?': 'yes', 'cap-color': 'brown', 'cap-shape': 'convex',
             'cap-surface': 'smooth', 'classification': 'poisonous',
             'gill-attachment': 'free', 'gill-color': 'black', 'gill-size':
             'narrow', 'gill-spacing': 'closed', 'habitat': 'urban', 'odor':
             'pungent', 'population': 'scattered', 'ring-number': 'one',
             'ring-type': 'pendant', 'spore-print-color': 'black',
             'stalk-color-above-ring': 'white', 'stalk-color-below-ring':
             'white', 'stalk-root': 'equal', 'stalk-shape': 'enlarging',
             'stalk-surface-above-ring': 'smooth', 'stalk-surface-below-ring':
             'smooth', 'veil-color': 'white', 'veil-type': 'partial'}
    assert known == data[0]


def test_load_rb_com_11():
    data = load_rb_com_11(num_instances=1)
    known = {'_guid': 'ea022d3d-5c9e-46d7-be23-8ea718fe7816',
             '_human_cluster_label': '0', 'component0': {'b': 1.0, 'l': 0.0,
                                                         'r': 1.0, 't': 2.0,
                                                         'type': 'cube0'},
             'component1': {'b': 3.0, 'l': 2.0, 'r': 3.0, 't': 4.0, 'type':
                            'cube0'}, 'component14': {'b': 4.0, 'l': 1.0, 'r':
                                                      4.0, 't': 5.0, 'type':
                                                      'ufoo0'}, 'component2':
             {'b': 1.0, 'l': 1.0, 'r': 4.0, 't': 2.0, 'type': 'plat0'},
             'component3': {'b': 2.0, 'l': 1.0, 'r': 4.0, 't': 3.0, 'type':
                            'plat0'}, 'component4': {'b': 0.0, 'l': 0.0, 'r':
                                                     5.0, 't': 1.0, 'type':
                                                     'rect0'}}
    assert known == data[0]


def test_load_rb_s_07():
    data = load_rb_s_07(num_instances=1)
    known = {'_guid': '660ac76d-93b3-4ce7-8a15-a3213e9103f5', 'component0':
             {'b': 0.0, 'l': 0.0, 'r': 3.0, 't': 1.0, 'type': 'plat0'},
             'component1': {'b': 1.0, 'l': 1.0, 'r': 2.0, 't': 4.0, 'type':
                            'plat90'}, 'component8': {'b': 4.0, 'l': 0.0, 'r':
                                                      3.0, 't': 5.0, 'type':
                                                      'ufoo0'}, 'success': '0'}
    assert known == data[0]


def test_load_rb_s_13():
    data = load_rb_s_13(num_instances=1)
    known = {'_guid': '684b4ce5-0f55-481c-ae9a-1474de8418ea',
             '_human_cluster_label': '0', 'component0': {'b': 3.0, 'l': 2.0,
                                                         'r': 3.0, 't': 4.0,
                                                         'type': 'cube0'},
             'component1': {'b': 4.0, 'l': 2.0, 'r': 3.0, 't': 5.0, 'type':
                            'cube0'}, 'component14': {'b': 0.0, 'l': 0.0, 'r':
                                                      4.0, 't': 1.0, 'type':
                                                      'trap0'}, 'component15':
             {'b': 5.0, 'l': 1.0, 'r': 3.0, 't': 6.0, 'type': 'ufoo0'},
             'component2': {'b': 1.0, 'l': 0.0, 'r': 3.0, 't': 2.0, 'type':
                            'plat0'}, 'component3': {'b': 2.0, 'l': 0.0, 'r':
                                                     3.0, 't': 3.0, 'type':
                                                     'plat0'}}
    assert data[0] == known


def test_load_rb_wb_03():
    data = load_rb_wb_03(num_instances=1)
    known = {'_guid': 'aa5eff72-0572-4eff-a007-3def9a82ba5b',
             '_human_cluster_label': '0', 'component0': {'b': 2.0, 'l': 2.0,
                                                         'r': 3.0, 't': 3.0,
                                                         'type': 'cube0'},
             'component1': {'b': 2.0, 'l': 3.0, 'r': 4.0, 't': 3.0, 'type':
                            'cube0'}, 'component11': {'b': 3.0, 'l': 1.0, 'r':
                                                      4.0, 't': 4.0, 'type':
                                                      'ufoo0'}, 'component2':
             {'b': 1.0, 'l': 2.0, 'r': 5.0, 't': 2.0, 'type': 'plat0'},
             'component3': {'b': 0.0, 'l': 0.0, 'r': 5.0, 't': 1.0, 'type':
                            'rect0'}}
    assert known == data[0]


def test_rb_s_07_human_predictions():
    data = load_rb_s_07_human_predictions()
    known = ['user_id,instance_guid,time,order,prediction,correctness',
             '1,2fda0bde-95a7-4bda-9851-785275c3f56d,2015-02-15 '
             '19:21:14.327344+00:00,1,0,1']
    assert known == data[0:2]


def test_load_quadruped():
    random.seed(0)
    data = load_quadruped(10)
    assert len(data) == 10

    assert 'head' in data[0]
    assert 'leg1' in data[0]
    assert 'tail' in data[0]


def test_load_molecule():
    data = load_molecule()
    known = {'(bond Single Not_stereo ?atom0001 ?atom0003)': True,
             '(bond Single Not_stereo ?atom0001 ?atom0014)': True,
             '(bond Single Not_stereo ?atom0002 ?atom0004)': True,
             '(bond Single Not_stereo ?atom0002 ?atom0012)': True,
             '(bond Single Not_stereo ?atom0002 ?atom0013)': True,
             '(bond Single Not_stereo ?atom0003 ?atom0004)': True,
             '(bond Single Not_stereo ?atom0003 ?atom0005)': True,
             '(bond Single Not_stereo ?atom0003 ?atom0006)': True,
             '(bond Single Not_stereo ?atom0004 ?atom0007)': True,
             '(bond Single Not_stereo ?atom0004 ?atom0008)': True,
             '(bond Single Not_stereo ?atom0005 ?atom0009)': True,
             '(bond Single Not_stereo ?atom0005 ?atom0010)': True,
             '(bond Single Not_stereo ?atom0005 ?atom0011)': True,
             '?atom0001': {'charge': 'outside_limits',
                           'hydrogen_count': 'H0',
                           'mass_diff': '0',
                           'stereo_parity': 'not_stereo',
                           'symbol': 'O',
                           'valence': 'no marking',
                           'x': 2.5369,
                           'y': 0.75,
                           'z': 0.0},
             '?atom0002': {'charge': 'outside_limits',
                           'hydrogen_count': 'H0',
                           'mass_diff': '0',
                           'stereo_parity': 'not_stereo',
                           'symbol': 'N',
                           'valence': 'no marking',
                           'x': 5.135,
                           'y': 0.25,
                           'z': 0.0},
             '?atom0003': {'charge': 'outside_limits',
                           'hydrogen_count': 'H0',
                           'mass_diff': '0',
                           'stereo_parity': 'unmarked',
                           'symbol': 'C',
                           'valence': 'no marking',
                           'x': 3.403,
                           'y': 0.25,
                           'z': 0.0},
             '?atom0004': {'charge': 'outside_limits',
                           'hydrogen_count': 'H0',
                           'mass_diff': '0',
                           'stereo_parity': 'not_stereo',
                           'symbol': 'C',
                           'valence': 'no marking',
                           'x': 4.269,
                           'y': 0.75,
                           'z': 0.0},
             '?atom0005': {'charge': 'outside_limits',
                           'hydrogen_count': 'H0',
                           'mass_diff': '0',
                           'stereo_parity': 'not_stereo',
                           'symbol': 'C',
                           'valence': 'no marking',
                           'x': 3.403,
                           'y': -0.75,
                           'z': 0.0},
             '?atom0006': {'charge': 'outside_limits',
                           'hydrogen_count': 'H0',
                           'mass_diff': '0',
                           'stereo_parity': 'not_stereo',
                           'symbol': 'H',
                           'valence': 'no marking',
                           'x': 3.403,
                           'y': 1.1,
                           'z': 0.0},
             '?atom0007': {'charge': 'outside_limits',
                           'hydrogen_count': 'H0',
                           'mass_diff': '0',
                           'stereo_parity': 'not_stereo',
                           'symbol': 'H',
                           'valence': 'no marking',
                           'x': 4.6675,
                           'y': 1.225,
                           'z': 0.0},
             '?atom0008': {'charge': 'outside_limits',
                           'hydrogen_count': 'H0',
                           'mass_diff': '0',
                           'stereo_parity': 'not_stereo',
                           'symbol': 'H',
                           'valence': 'no marking',
                           'x': 3.8705,
                           'y': 1.225,
                           'z': 0.0},
             '?atom0009': {'charge': 'outside_limits',
                           'hydrogen_count': 'H0',
                           'mass_diff': '0',
                           'stereo_parity': 'not_stereo',
                           'symbol': 'H',
                           'valence': 'no marking',
                           'x': 2.783,
                           'y': -0.75,
                           'z': 0.0},
             '?atom0010': {'charge': 'outside_limits',
                           'hydrogen_count': 'H0',
                           'mass_diff': '0',
                           'stereo_parity': 'not_stereo',
                           'symbol': 'H',
                           'valence': 'no marking',
                           'x': 3.403,
                           'y': -1.37,
                           'z': 0.0},
             '?atom0011': {'charge': 'outside_limits',
                           'hydrogen_count': 'H0',
                           'mass_diff': '0',
                           'stereo_parity': 'not_stereo',
                           'symbol': 'H',
                           'valence': 'no marking',
                           'x': 4.023,
                           'y': -0.75,
                           'z': 0.0},
             '?atom0012': {'charge': 'outside_limits',
                           'hydrogen_count': 'H0',
                           'mass_diff': '0',
                           'stereo_parity': 'not_stereo',
                           'symbol': 'H',
                           'valence': 'no marking',
                           'x': 5.672,
                           'y': 0.56,
                           'z': 0.0},
             '?atom0013': {'charge': 'outside_limits',
                           'hydrogen_count': 'H0',
                           'mass_diff': '0',
                           'stereo_parity': 'not_stereo',
                           'symbol': 'H',
                           'valence': 'no marking',
                           'x': 5.135,
                           'y': -0.37,
                           'z': 0.0},
             '?atom0014': {'charge': 'outside_limits',
                           'hydrogen_count': 'H0',
                           'mass_diff': '0',
                           'stereo_parity': 'not_stereo',
                           'symbol': 'H',
                           'valence': 'no marking',
                           'x': 2.0,
                           'y': 0.44,
                           'z': 0.0},
             '_name': '4',
             '_software': '-OEChem-03201502492D',
             '_version': 'V2000',
             'chiral': True}

    assert known == data[3]
