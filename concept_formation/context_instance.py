from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division


class ContextInstance:
    """
    """

    def __init__(self, tenative_path=None):
        """constructor

        :param tenative_path: The path its eventual instance is currently
            going to be added to.
        :type tenative_path: Set<ContetxualCobwebNode>"""
        self.instance = None
        self.tenative_path = tenative_path

    def __hash__(self):
        return hash(self.instance)

    def __eq__(self, __o):
        return (type(__o) == ContextInstance
                and (__o.instance == self.instance is not None
                     or __o is self))

    def set_instance(self, node):
        assert self.instance is None, ("Cannot set ContextInstance "
                                       "more than once")
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
        :type tenative_path: SetContetxualCobwebNode>
        """
        self.tenative_path = path

    def desc_of(self, node):
        """
        Returns whether context is or planned to be descendant of node.

        :param node: node to check
        :type node: ContextualCobwebNode
        """
        if self.instance is not None:
            return self.instance in node.descendants
        else:
            return node in self.tenative_path

    def output_json(self):
        raise NotImplementedError
