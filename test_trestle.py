from trestle import Trestle


def verify_counts(tres, from_root=False):
    """
    Checks the property that the counts of the children sum to the same
    count as the parent. This is/was useful when debugging. If you are
    doing some kind of matching at each step in the categorization (i.e.,
    renaming such as with Labyrinth) then this will start throwing errors.
    """
    assert isinstance(tres,Trestle)
    if from_root:
        tres = tres.get_root()


    if len(tres.children) == 0:
        return 

    temp = {}
    temp_count = tres.count
    for attr in tres.av_counts:
        if isinstance(tres.av_counts[attr], ContinuousValue):
            temp[attr] = tres.av_counts[attr].num
        else:
            if attr not in temp:
                temp[attr] = {}
            for val in tres.av_counts[attr]:
                temp[attr][val] = tres.av_counts[attr][val]

    for child in tres.children:
        temp_count -= child.count
        for attr in child.av_counts:
            assert attr in temp
            if isinstance(child.av_counts[attr], ContinuousValue):
                temp[attr] -= child.av_counts[attr].num
            else:
                for val in child.av_counts[attr]:
                    if val not in temp[attr]:
                        print(val.concept_name)
                        print(attr)
                        print(tres.
                    assert val in temp[attr]
                    temp[attr][val] -= child.av_counts[attr][val]

    #if temp_count != 0:
    #    print(tres.count)
    #    for child in tres.children:
    #        print(child.count)
    assert temp_count == 0

    for attr in temp:
        if isinstance(temp[attr], int):
            assert temp[attr] == 0.0
        else:
            for val in temp[attr]:
                #if temp[attr][val] != 0.0:
                #    print(tres.

                assert temp[attr][val] == 0.0

    for child in tres.children:
        verify_counts(child,False)

def verify_parent_pointers(tres,from_root=False):
    """
    A function to verify the integrity of parent pointers throughout the tree.
    """
    assert isinstance(tres,Trestle)
    if from_root:
        tres = tres.get_root()
    for c in tres.children:
        assert c.parent == tres
        verify_parent_pointers(c, False)     


def verify_component_values(tres,from_root=False):
    """
    A function to verify that all values in a Trestle node's attribute value table
    are leaves of the tree rather than intermediate nodes.
    """
    assert isinstance(tres,Trestle)
    if from_root:
        tres = tres.get_root()

    for attr in tres.av_counts:
        for val in tres.av_counts[attr]:
            if isinstance(val, Trestle):
                assert not val.children
    for c in self.children:
        verify_component_values(c,False)
