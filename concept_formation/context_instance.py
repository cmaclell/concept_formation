from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division


class ContextInstance:
    """
    """

    def __init__(self, tenative_path):
        """constructor

        :param tenative_path: The path its eventual instance is currently
            going to be added to.
        :type tenative_path: Sequence<ContetxualCobwebNode>"""
        # Holds the final existant node in the tree on the path
        self.instance = tenative_path[-1]
        # Holds the tenative path or None if it has been finalized
        self.tenative_path = set(tenative_path)

    def __hash__(self):
        return hash(self.instance)

    def __eq__(self, __o):
        return (type(__o) == ContextInstance
                and (__o is self or
                     (__o.instance == self.instance
                      and self.tenative_path is None)))

    def __str__(self):
        if self.tenative_path is None:
            return 'CobwebNode{}'.format(self.instance.concept_id)
        return 'Unadded leaf of CobwebNode{}'.format(self.instance.concept_id)

    def set_instance(self, node):
        assert self.tenative_path is not None, ("Cannot set Con"
                                                "textInstance more than once")
        self.tenative_path = None
        self.instance = node
        cur_node = node
        while cur_node:
            cur_node.descendants.add(self.instance)
            cur_node = cur_node.parent
        return self.instance

    def set_path(self, path):
        """
        :param tenative_path: The path its eventual instance is currently
            going to be added to.
        :type tenative_path: Sequence<ContetxualCobwebNode>
        """
        self.instance = path[-1]
        self.tenative_path = path

    def desc_of(self, node):
        """
        Returns whether context is or planned to be descendant of node.

        :param node: node to check
        :type node: ContextualCobwebNode
        :return: (whether is descendant, whether it is planned to be added
            as a direct child of node but has not yet been added)
        :rtype: (bool, bool)
        """
        if self.tenative_path is None:
            return (self.instance in node.descendants, False)
        else:
            return (node in self.tenative_path, node == self.instance)

    def output_json(self):
        raise NotImplementedError
