"""
The visualize module provides functions for generating html visualizations of
trees created by the other modules of concept_formation.
"""
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
from os.path import realpath
from os.path import dirname
from os.path import join
from os.path import exists
from shutil import copy
import webbrowser
import json
import math

from concept_formation.cobweb import CobwebNode


def _copy_file(filename, target_dir):
    module_path = dirname(__file__)
    src = join(module_path, 'visualization_files', filename)
    dst = join(target_dir, filename)
    copy(src, dst)
    return dst


def _gen_output_file(js_ob):
    return '(function (){ window.trestle_output='+js_ob+'; })();'


def _gen_viz(js_ob, dst, recreate_html):
    if dst is None:
        module_path = dirname(__file__)
        output_file = join(module_path, 'visualization_files', 'output.js')
        with open(output_file, 'w') as out:
            out.write(_gen_output_file(js_ob))
        viz_html_file = join(module_path, 'visualization_files', 'viz.html')
        webbrowser.open('file://'+realpath(viz_html_file))
    else:
        if recreate_html or not exists(join(dst, 'viz.html')):
            viz_file = _copy_file('viz.html', dst)
            _copy_file('viz_logic.js', dst)
            _copy_file('viz_styling.css', dst)
        else:
            viz_file = join(dst, 'viz.html')
        with open(join(dst, 'output.js'), 'w') as out:
            out.write(_gen_output_file(js_ob))
        webbrowser.open('file://' + realpath(viz_file))

def entropy_component_k(n, p):
    info = -n * p * math.log(p) if p > 0 else 0
    for xi in range(n + 1):
        info += math.comb(n, xi) * math.pow(p, xi) * math.pow((1 - p), (n - xi)) * math.lgamma(xi + 1)
    return info

def frex(root, tree_data, w=0.5):

    if len(tree_data['children']) == 0:
        return tree_data

    new = {'name': tree_data['name'],
           'size': tree_data['size'],
           'children': [],
           'attr_counts': tree_data['attr_counts'],
           'counts': tree_data['counts']}

    for child in tree_data['children']:
        new['children'].append({'name': child['name'],
                             'size': child['size'],
                             'children': child['children'],
                             'attr_counts': child['attr_counts'],
                             'counts': {}})

    # get normalization
    normalize = {}
    for attr in tree_data['counts']:
        if attr not in normalize:
            normalize[attr] = {}
        for val in tree_data['counts'][attr]:
            normalize[attr][val] = 0
            for child in tree_data['children']:
                if attr in child['counts'] and val in child['counts'][attr]:
                    normalize[attr][val] += child['counts'][attr][val] / child['attr_counts'][attr]

    phis = []
    freqs = []
    for child in tree_data['children']:
        phi_c = []
        freq_c = []
        for attr in child['counts']:
            for val in child['counts'][attr]:
                p = child['counts'][attr][val] / child['attr_counts'][attr]
                p_root = root['counts'][attr][val] / root['attr_counts'][attr]
                freq_c.append(p)
                # phi_c.append(entropy_component_k(1, p_root) - entropy_component_k(1, p))
                phi_c.append(p / p_root)
                # phi_c.append(p / normalize[attr][val])
        freqs.append(freq_c)
        phis.append(phi_c)

    for i, child in enumerate(tree_data['children']):
        new_child = new['children'][i]
        phi_c = phis[i]
        freq_c = freqs[i]
        for attr in child['counts']:
            if attr not in new_child['counts']:
                new_child['counts'][attr] = {}
            for val in child['counts'][attr]:
                p = child['counts'][attr][val] / child['attr_counts'][attr]
                p_root = root['counts'][attr][val] / root['attr_counts'][attr]
                # phi = entropy_component_k(8, p_root) - entropy_component_k(8, p)
                phi = p / p_root
                # phi = p / normalize[attr][val]
                ex_cdf = sum([phi >= x for x in phi_c]) / len(phi_c)
                p_cdf = sum([p >= x for x in freq_c]) / len(freq_c)
                # new_child['counts'][attr][val] = phi
                new_child['counts'][attr][val] = 1 / ((w / ex_cdf) + ((1-w) / p_cdf))

    new['children'] = [frex(root, child, w) for child in new['children']]
    return new

def visualize(tree, dst=None, recreate_html=True):
    """
    Create an interactive visualization of a concept_formation tree and open
    it in your browswer.

    If a destination directory is specified this function will create html,
    js, and css files in the destination directory provided. By default this
    will always recreate the support html, js, and css files but a flag can
    turn this off.

    :param tree: A category tree to visualize
    :param dst: A directory to generate visualization files into. If None no
        files will be generated
    :param create_html: A flag for whether new supporting html files should be
        created
    :type tree: :class:`CobwebTree <concept_formation.cobweb.CobwebTree>`,
        :class:`Cobweb3Tree <concept_formation.cobweb3.Cobweb3Tree>`, or
        :class:`TrestleTree <concept_formation.trestle.TrestleTree>`
    :type dst: str
    :type create_html: bool
    """
    # tree_data = json.loads(tree.root.output_json())
    # tree_data = json.dumps(frex(tree_data, tree_data))
    # _gen_viz(tree_data, dst, recreate_html)

    _gen_viz(tree.root.output_json(), dst, recreate_html)


def _trim_leaves(j_ob):
    ret = {k: j_ob[k] for k in j_ob if k != 'children'}
    ret['children'] = [_trim_leaves(
        child) for child in j_ob['children'] if len(child['children']) > 0]
    return ret


def visualize_no_leaves(tree, cuts=1, dst=None, recreate_html=True):
    """
    Create an interactive visualization of a concept_formation tree cuts levels
    above the leaves and open it in your browswer.

    This visualization differs from the normal one by trimming the leaves from
    the tree. This is often useful in seeing patterns when the individual
    leaves are overly frequent visual noise.

    If a destination directory is specified this function will create html,
    js, and css files in the destination directory provided. By default this
    will always recreate the support html, js, and css files but a flag can
    turn this off.

    :param tree: A category tree to visualize
    :param cuts: The number of times to trim up the leaves
    :param dst: A directory to generate visualization files into. If None no
        files will be generated
    :param create_html: A flag for whether new supporting html files should be
        created
    :type tree: :class:`CobwebTree <concept_formation.cobweb.CobwebTree>`,
        :class:`Cobweb3Tree <concept_formation.cobweb3.Cobweb3Tree>`, or
        :class:`TrestleTree <concept_formation.trestle.TrestleTree>`
    :type cuts: int
    :type dst: str
    :type create_html: bool
    """
    j_ob = tree.root.output_json()
    for i in range(cuts):
        j_ob = _trim_leaves(j_ob)
    _gen_viz(j_ob, dst, recreate_html)


def _trim_to_clusters(j_ob, clusters):
    ret = {k: j_ob[k] for k in j_ob if k != 'children'}
    if j_ob['name'] not in clusters:
        ret['children'] = [_trim_to_clusters(
            child, clusters) for child in j_ob['children']]
    else:
        ret['children'] = []
    return ret


def visualize_clusters(tree, clusters, dst=None, recreate_html=True):
    """
    Create an interactive visualization of a concept_formation tree trimmed to
    the level specified by a clustering from the cluster module.

    This visualization differs from the normal one by trimming the tree to the
    level of a clustering. Basically the output traverses down the tree but
    stops recursing if it hits a node in the clustering. Both label or concept
    based clusterings are supported as the relevant names will be extracted.

    If a destination directory is specified this function will create html,
    js, and css files in the destination directory provided. By default this
    will always recreate the support html, js, and css files but a flag can
    turn this off.

    :param tree: A category tree to visualize
    :param clusters: A list of cluster labels or concept nodes generated by
        the cluster module.
    :param dst: A directory to generate visualization files into. If None no
        files will be generated
    :param create_html: A flag for whether new supporting html files should be
        created
    :type tree: :class:`CobwebTree <concept_formation.cobweb.CobwebTree>`,
        :class:`Cobweb3Tree <concept_formation.cobweb3.Cobweb3Tree>`, or
        :class:`TrestleTree <concept_formation.trestle.TrestleTree>`
    :type clusters: list
    :type dst: str
    :type create_html: bool
    """
    if isinstance(clusters[0], CobwebNode):
        clusters = {str(c.concept_id) for c in clusters}
    else:
        clusters = set(clusters)

    j_ob = tree.root.output_json()
    j_ob = _trim_to_clusters(j_ob, clusters)
    _gen_viz(j_ob, dst, recreate_html)
