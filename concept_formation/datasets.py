import json

from os.path import dirname
from os.path import join

def load_json(filename):
    module_path = dirname(__file__)
    with open(join(module_path, 'data_files', filename)) as dat:
        output = json.load(dat)
    return output

def load_file(filename):
    module_path = dirname(__file__)
    with open(join(module_path, 'data_files', filename)) as dat:
        output = [row for row in dat]
    return output

def load_iris():
    return load_json('iris.json')

def load_mushroom():
    return load_json('mushrooms.json')

def load_rb_com_11():
    return load_json('rb_com_11_continuous.json')

def load_rb_s_07():
    return load_json('rb_s_07_continuous.json')

def load_rb_s_13():
    return load_json('rb_s_13_continuous.json')

def load_rb_wb_03():
    return load_json('rb_wb_03_continuous.json')

def load_rb_s_07_human_predictions():
    return load_file('human_s_07_success_predictions.csv')
