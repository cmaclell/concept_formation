from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division


class ContextInstance:
    """
    The ContextInstance class provides a wrapper for paths in the tree to make
    updating and finalizing paths for leaf nodes faster and easier. By having
    several instances store the same context object, paths only need to be
    updated once per instance being added. Its tenative_path can be changed
    many times, but once a final path is set, it is final.

    desc_of is constant time, and the initializer/set_path is linear in the
    length of the path.

    :param tenative_path: The path its eventual instance is currently
        going to be added to. Can change later.
    :type tenative_path: Sequence<ContetxualCobwebNode>
    """

    def __init__(self, tenative_path):
        """
        Context constructor.
        """
        # Holds the final existant node in the tree on the path
        self.instance = tenative_path[-1]
        # Holds the tenative path or None if it has been finalized
        self.tenative_path = set(tenative_path)

    def __str__(self):
        return repr(self)

    def __repr__(self):
        if self.tenative_path is None:
            return 'Node{}'.format(self.instance.concept_id)
        return 'Unadded leaf of Node{}'.format(self.instance.concept_id)

    def set_instance(self, leaf):
        """
        Finalize this ContextInstance by associating it with a leaf node.
        Also updates the tree nodes' list of descended leaves to include the
        new node (though the tree is now responsibility for maintaining this).

        :param leaf: leaf node for this ContextInstance
        :type leaf: ContextualCobwebNode
        :return: the inputted leaf node
        :rtype: ContextualCobwebNode
        """
        assert self.tenative_path is not None, ("Cannot change the leaf of a "
                                                "finalized ContextInstance")
        self.tenative_path = None
        self.instance = leaf
        cur_node = leaf
        while cur_node:
            cur_node.descendants.add(self.instance)
            cur_node = cur_node.parent
        return self.instance

    def set_path(self, path):
        """
        Change the current path of this ContextInstance.

        :param tenative_path: The path its eventual instance is currently
            going to be added to.
        :type tenative_path: Sequence<ContetxualCobwebNode>
        """
        assert self.tenative_path is not None, ("Cannot change the leaf of a "
                                                "finalized ContextInstance")
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
