from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division
import json
from os.path import dirname
from os.path import join

def load_json(filename):
    """
    Loads a json file and returns a python object generated from parsing the
    json.
    """
    module_path = dirname(__file__)
    with open(join(module_path, 'data_files', filename)) as dat:
        output = json.load(dat)
    return output

def load_file(filename):
    """
    Reads the rows of a file and returns them as an array.
    """
    module_path = dirname(__file__)
    with open(join(module_path, 'data_files', filename)) as dat:
        output = [row for row in dat]
    return output

def load_iris():
    """
    Load the iris dataset.
    """
    return load_json('iris.json')

def load_mushroom():
    """
    Load the mushroom dataset.
    """
    return load_json('mushrooms.json')

def load_rb_com_11():
    """
    Load the RumbleBlocks, Center of Mass Level 11, dataset.
    """
    return load_json('rb_com_11_continuous.json')

def load_rb_s_07():
    """
    Load the RumbleBlocks, Symmetry Level 7, dataset.
    """
    return load_json('rb_s_07_continuous.json')

def load_rb_s_13():
    """
    Load the RumbleBlocks, Symmetry Level 13, dataset.
    """
    return load_json('rb_s_13_continuous.json')

def load_rb_wb_03():
    """
    Load the RumbleBlocks, Wide Base Level 03, dataset.
    """
    return load_json('rb_wb_03_continuous.json')

def load_rb_s_07_human_predictions():
    """
    Load the Human Predictions Data for the RumbleBlocks, Symmetry Level 7,
    dataset.
    """
    return load_file('human_s_07_success_predictions.csv')
