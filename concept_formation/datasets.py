"""
The dataset module has functions for loading a variety of datasets that
are properly formated for use with CobwebTrees and their derivatives.
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
from os.path import dirname
from os.path import join
import json

from concept_formation.data_files.generate_quadruped import generate_animals

def _load_json(filename):
    """
    Loads a json file and returns a python object generated from parsing the
    json.
    """
    module_path = dirname(__file__)
    with open(join(module_path, 'data_files', filename)) as dat:
        output = json.load(dat)
    return output

def _load_file(filename):
    """
    Reads the rows of a file and returns them as an array.
    """
    module_path = dirname(__file__)
    with open(join(module_path, 'data_files', filename)) as dat:
        output = [row[:-1] for row in dat]
    return output

def load_forest_fires():
    """
    Load the forest fires dataset.

    This is an example of instances with :ref:`Nominal<val-nom>` and
    :ref:`Numeric<val-num>` values and :ref:`Constant<attr-const>` attributes.

    This dataset was downloaded from the `UCI machine learning repository
    <http://archive.ics.uci.edu/ml/datasets/Forest+Fires>`__.
    We processed the data to be in dictionary format with human readable
    labels. 

    >>> import pprint
    >>> data = load_forest_fires()
    >>> print(len(data))
    517
    >>> pprint.pprint(data[0])
    {'DC': 94.3,
     'DMC': 26.2,
     'FFMC': 86.2,
     'ISI': 5.1,
     'RH': 51.0,
     'area': 0.0,
     'day': 'fri',
     'month': 'mar',
     'rain': 0.0,
     'temp': 8.2,
     'wind': 6.7,
     'x-axis': 7.0,
     'y-axis': 5.0}

    """
    return _load_json('forest_fires.json')

def load_congressional_voting():
    """
    Load the voting dataset.

    This is an example of instances with only :ref:`Nominal<val-nom>` values
    and :ref:`Constant<attr-const>` attributes but some attributes are
    occasionally missing.

    This dataset was downloaded from the `UCI machine learning repository
    <http://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records>`__.
    We processed the data to be in dictionary format with human readable
    labels. 

    >>> import pprint
    >>> data = load_congressional_voting()
    >>> print(len(data))
    435
    >>> pprint.pprint(data[0])
    {'Class Name': 'republican',
     'adoption-of-the-budget-resolution': 'n',
     'aid-to-nicaraguan-contras': 'n',
     'anti-satellite-test-ban': 'n',
     'crime': 'y',
     'duty-free-exports': 'n',
     'education-spending': 'y',
     'el-salvador-aid': 'y',
     'export-administration-act-south-africa': 'y',
     'handicapped-infants': 'n',
     'immigration': 'y',
     'mx-missile': 'n',
     'physician-fee-freeze': 'y',
     'religious-groups-in-schools': 'y',
     'superfund-right-to-sue': 'y',
     'water-project-cost-sharing': 'y'}

    """
    return _load_json('congressional_voting.json')

def load_iris():
    """
    Load the iris dataset.

    This is an example of instances with :ref:`Nominal<val-nom>` and
    :ref:`Numeric<val-num>` values and :ref:`Constant<attr-const>` attributes.

    This dataset was downloaded from the `UCI machine learning repository
    <https://archive.ics.uci.edu/ml/datasets/Iris>`__. We processed the data
    to be in dictionary format with human readable labels. 

    >>> import pprint
    >>> data = load_iris()
    >>> print(len(data))
    150
    >>> pprint.pprint(data[0])
    {'class': 'Iris-setosa',
     'petal length': 1.4,
     'petal width': 0.2,
     'sepal length': 5.1,
     'sepal width': 3.5}

    """
    return _load_json('iris.json')

def load_mushroom():
    """
    Load the mushroom dataset.

    This is an example of instances with only :ref:`Nominal<val-nom>` values
    and :ref:`Constant<attr-const>` attributes.

    This dataset was downloaded from the `UCI machine learning repository
    <https://archive.ics.uci.edu/ml/datasets/Mushroom>`__. We processed the data
    to be in dictionary format with human readable labels. 

    >>> import pprint
    >>> data = load_mushroom()
    >>> print(len(data))
    8124
    >>> pprint.pprint(data[0])
    {'bruises?': 'yes',
     'cap-color': 'brown',
     'cap-shape': 'convex',
     'cap-surface': 'smooth',
     'classification': 'poisonous',
     'gill-attachment': 'free',
     'gill-color': 'black',
     'gill-size': 'narrow',
     'gill-spacing': 'closed',
     'habitat': 'urban',
     'odor': 'pungent',
     'population': 'scattered',
     'ring-number': 'one',
     'ring-type': 'pendant',
     'spore-print-color': 'black',
     'stalk-color-above-ring': 'white',
     'stalk-color-below-ring': 'white',
     'stalk-root': 'equal',
     'stalk-shape': 'enlarging',
     'stalk-surface-above-ring': 'smooth',
     'stalk-surface-below-ring': 'smooth',
     'veil-color': 'white',
     'veil-type': 'partial'}
    """
    return _load_json('mushrooms.json')

def load_rb_com_11():
    """
    Load the RumbleBlocks, Center of Mass Level 11, dataset.

    This is an example of instances with all the attribute and value types
    described in the :ref:`instance-rep`.

    >>> import pprint
    >>> data = load_rb_com_11()
    >>> print(len(data))
    251
    >>> pprint.pprint(data[0])
    {'_guid': 'ea022d3d-5c9e-46d7-be23-8ea718fe7816',
     '_human_cluster_label': '0',
     'component0': {'b': 1.0, 'l': 0.0, 'r': 1.0, 't': 2.0, 'type': 'cube0'},
     'component1': {'b': 3.0, 'l': 2.0, 'r': 3.0, 't': 4.0, 'type': 'cube0'},
     'component14': {'b': 4.0, 'l': 1.0, 'r': 4.0, 't': 5.0, 'type': 'ufoo0'},
     'component2': {'b': 1.0, 'l': 1.0, 'r': 4.0, 't': 2.0, 'type': 'plat0'},
     'component3': {'b': 2.0, 'l': 1.0, 'r': 4.0, 't': 3.0, 'type': 'plat0'},
     'component4': {'b': 0.0, 'l': 0.0, 'r': 5.0, 't': 1.0, 'type': 'rect0'}}
    """
    return _load_json('rb_com_11_continuous.json')

def load_rb_s_07():
    """
    Load the RumbleBlocks, Symmetry Level 7, dataset.

    This is an example of instances with all the attribute and value types
    described in the :ref:`instance-rep`.

    >>> import pprint
    >>> data = load_rb_s_07()
    >>> print(len(data))
    141
    >>> pprint.pprint(data[0])
    {'_guid': '660ac76d-93b3-4ce7-8a15-a3213e9103f5',
     'component0': {'b': 0.0, 'l': 0.0, 'r': 3.0, 't': 1.0, 'type': 'plat0'},
     'component1': {'b': 1.0, 'l': 1.0, 'r': 2.0, 't': 4.0, 'type': 'plat90'},
     'component8': {'b': 4.0, 'l': 0.0, 'r': 3.0, 't': 5.0, 'type': 'ufoo0'},
     'success': '0'}
    """
    return _load_json('rb_s_07_continuous.json')

def load_rb_s_13():
    """
    Load the RumbleBlocks, Symmetry Level 13, dataset.

    This is an example of instances with all the attribute and value types
    described in the :ref:`instance-rep`.

    >>> import pprint
    >>> data = load_rb_s_13()
    >>> print(len(data))
    249
    >>> pprint.pprint(data[0])
    {'_guid': '684b4ce5-0f55-481c-ae9a-1474de8418ea',
     '_human_cluster_label': '0',
     'component0': {'b': 3.0, 'l': 2.0, 'r': 3.0, 't': 4.0, 'type': 'cube0'},
     'component1': {'b': 4.0, 'l': 2.0, 'r': 3.0, 't': 5.0, 'type': 'cube0'},
     'component14': {'b': 0.0, 'l': 0.0, 'r': 4.0, 't': 1.0, 'type': 'trap0'},
     'component15': {'b': 5.0, 'l': 1.0, 'r': 3.0, 't': 6.0, 'type': 'ufoo0'},
     'component2': {'b': 1.0, 'l': 0.0, 'r': 3.0, 't': 2.0, 'type': 'plat0'},
     'component3': {'b': 2.0, 'l': 0.0, 'r': 3.0, 't': 3.0, 'type': 'plat0'}}
    """
    return _load_json('rb_s_13_continuous.json')

def load_rb_wb_03():
    """
    Load the RumbleBlocks, Wide Base Level 03, dataset.

    This is an example of instances with all the attribute and value types
    described in the :ref:`instance-rep`.

    >>> import pprint
    >>> data = load_rb_wb_03()
    >>> print(len(data))
    254
    >>> pprint.pprint(data[0])
    {'_guid': 'aa5eff72-0572-4eff-a007-3def9a82ba5b',
     '_human_cluster_label': '0',
     'component0': {'b': 2.0, 'l': 2.0, 'r': 3.0, 't': 3.0, 'type': 'cube0'},
     'component1': {'b': 2.0, 'l': 3.0, 'r': 4.0, 't': 3.0, 'type': 'cube0'},
     'component11': {'b': 3.0, 'l': 1.0, 'r': 4.0, 't': 4.0, 'type': 'ufoo0'},
     'component2': {'b': 1.0, 'l': 2.0, 'r': 5.0, 't': 2.0, 'type': 'plat0'},
     'component3': {'b': 0.0, 'l': 0.0, 'r': 5.0, 't': 1.0, 'type': 'rect0'}}
    """
    return _load_json('rb_wb_03_continuous.json')

def load_rb_s_07_human_predictions():
    """
    Load the Human Predictions Data for the RumbleBlocks, Symmetry Level 7,
    dataset.

    This is data collected from mechanical turk, where workers were tasked with
    predicting a concept label (success) given a picture of the tower. The
    element contains labels for the data and subsequent rows contain the actual
    data.

    >>> import pprint
    >>> data = load_rb_s_07_human_predictions()
    >>> print(len(data))
    601
    >>> pprint.pprint(data[0:2])
    ['user_id,instance_guid,time,order,prediction,correctness',
     '1,2fda0bde-95a7-4bda-9851-785275c3f56d,2015-02-15 '
     '19:21:14.327344+00:00,1,0,1']
    """
    return _load_file('human_s_07_success_predictions.csv')

def load_quadruped(num_instances):
    """
    Returns a randomly generated quadruped dataset of size `num_instances`
    using the procedure employed in: 
    
    Gennari, J. H., Langley, P., & Fisher, D. H. (1989). Models of incremental
    concept formation. Artificial Intelligence, 40, 11–61. 

    This dataset contains four kinds of quadruped animals: dogs, cats, horses,
    and giraffes. The type of each component is included as a hidden variable,
    so that structure mapping can be tested. Additionally, the type of animal
    (e.g., dog) is also included as a hidden variable. 

    >>> import pprint
    >>> import random
    >>> random.seed(0)
    >>> data = load_quadruped(10)
    >>> print(len(data))
    10
    >>> pprint.pprint(data[0:1])
    [{'_type': 'giraffe',
      'head': {'_type': 'head',
               'axisX': 1,
               'axisY': -0.23376215459531377,
               'axisZ': 0,
               'height': 19.069373148228724,
               'locationX': 71.71171645023995,
               'locationY': 0,
               'locationZ': 49.26645266304532,
               'radius': 4.05626484907961,
               'texture': 177.5670433982545},
      'leg1': {'_type': 'leg1',
               'axisX': 0.25279896094692916,
               'axisY': 0,
               'axisZ': -1,
               'height': 60.13197726212744,
               'locationX': 35.29119556606559,
               'locationY': 12.845931778870957,
               'locationZ': -42.91192040993468,
               'radius': 3.597944849223721,
               'texture': 179.23727389536953},
      'leg2': {'_type': 'leg2',
               'axisX': 0,
               'axisY': 0,
               'axisZ': -1,
               'height': 60.13197726212744,
               'locationX': 35.29119556606559,
               'locationY': -12.845931778870957,
               'locationZ': -42.91192040993468,
               'radius': 2.009043416794043,
               'texture': 174.58392827108403},
      'leg3': {'_type': 'leg3',
               'axisX': 0,
               'axisY': 0,
               'axisZ': -1,
               'height': 60.13197726212744,
               'locationX': -35.29119556606559,
               'locationY': 12.845931778870957,
               'locationZ': -42.91192040993468,
               'radius': 2.348946587645933,
               'texture': 178.9283460962157},
      'leg4': {'_type': 'leg4',
               'axisX': 0.28802829434429883,
               'axisY': 0,
               'axisZ': -1,
               'height': 60.13197726212744,
               'locationX': -35.29119556606559,
               'locationY': -12.845931778870957,
               'locationZ': -42.91192040993468,
               'radius': 2.9029316087251233,
               'texture': 171.86316987918838},
      'neck': {'_type': 'neck',
               'axisX': 1,
               'axisY': 0,
               'axisZ': 1,
               'height': 51.49861653022255,
               'locationX': 53.50145600815277,
               'locationY': 0,
               'locationZ': 31.05619222095814,
               'radius': 7.87732253394808,
               'texture': 177.14627952379485},
      'tail': {'_type': 'tail',
               'axisX': -1,
               'axisY': 0.24883477194257322,
               'axisZ': -0.531438665320418,
               'height': 20.918101962779517,
               'locationX': -49.66428916935166,
               'locationY': 0,
               'locationZ': 0,
               'radius': 0.9455145384298446,
               'texture': 177.24907471005645},
      'torso': {'_type': 'torso',
                'axisX': 1,
                'axisY': 0,
                'axisZ': 0,
                'height': 70.58239113213118,
                'locationX': 0,
                'locationY': 0,
                'locationZ': 0,
                'radius': 12.845931778870957,
                'texture': 171.2283287965781}}]
    """
    return generate_animals(num_instances) 

def load_molecule():
    """Load a dataset of 100 molecules from the pubchem database

    This dataset was downloaded from the `Pubchem databse
    <https://www.ncbi.nlm.nih.gov/pccompound>`__. We used a custom `molfile
    parser<https://github.com/eharpste/molparser>`__ to process the data to be
    in dictionary format with human readable labels.

    >>> import pprint
    >>> data = load_molecule()
    >>> print(len(data))
    101
    >>> pprint.pprint(data[3])
    {'(bond Single Not_stereo ?atom0001 ?atom0003)': True,
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
    """
    return _load_json('molecule.json')

