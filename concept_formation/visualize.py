"""
The visualize module provides functions for generating html visualizations of
trees created by the other modules of concept_formation.
"""
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
from os.path import dirname
from os.path import join
from os.path import exists
from shutil import copy
import webbrowser
import json

def _copy_file(filename, target_dir):
    module_path = dirname(__file__)
    src = join(module_path, 'visualization_files', filename)
    dst = join(target_dir, filename)
    copy(src,dst)

def visualize(tree,dst='.',recreate_html=True):
    """
    Create an interactive visualization of a concept_formation tree and open
    it in your browswer.

    Note that this function will create html, js, and css files in the
    destination directory provided. By default this will always recreate the
    support html, js, and css files but a flag can turn this off.

    :param tree: A category tree to visualize
    :param dst: A directory to generate visualization files into
    :param create_html: A flag for whether new supporting html files should be
        created
    :type tree: :class:`CobwebTree <concept_formation.cobweb.CobwebTree>`,
        :class:`Cobweb3Tree <concept_formation.cobweb3.Cobweb3Tree>`, or
        :class:`TrestleTree <concept_formation.trestle.TrestleTree>`
    :type dst: str
    :type create_html: bool
    """
    if recreate_html or not exists(join(dst,'viz.html')):
        _copy_file('viz.html',dst)
        _copy_file('viz_logic.js',dst)
        _copy_file('viz_styling.css',dst)
    with open(join(dst,'output.js'),'w') as out:
        out.write('var trestle_output = ')
        out.write(json.dumps(tree.root.output_json()))
        out.write(';')
    webbrowser.open(join(dst,'viz.html'))