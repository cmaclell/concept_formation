"""
The contents of this module are currently experimental and under active
development. More thorough documentation will be done when its development has
settled.
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
from itertools import product
import inspect

import numpy as np

from concept_formation.structure_mapper import flattenJSON
from concept_formation.search import Node
#from concept_formation.search import BestFGS
from concept_formation.search import BeamGS

def levenshtein(source, target):
    """ 
    The levenshtein edit distance, code take from here: 
        http://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python
    """
    if len(source) < len(target):
        return levenshtein(target, source)
 
    # So now we have len(source) >= len(target).
    if len(target) == 0:
        return len(source)
 
    # We call tuple() to force strings to be used as sequences
    # ('c', 'a', 't', 's') - numpy uses them as values by default.
    source = np.array(tuple(source))
    target = np.array(tuple(target))
 
    # We use a dynamic programming algorithm, but with the
    # added optimization that we only need the last two rows
    # of the matrix.
    previous_row = np.arange(target.size + 1)
    for s in source:
        # Insertion (target grows longer than source):
        current_row = previous_row + 1
 
        # Substitution or matching:
        # Target and source items are aligned, and either
        # are different (cost of 1), or are the same (cost of 0).
        current_row[1:] = np.minimum(
                current_row[1:],
                np.add(previous_row[:-1], target != s))
 
        # Deletion (target grows shorter than source):
        current_row[1:] = np.minimum(
                current_row[1:],
                current_row[0:-1] + 1)
 
        previous_row = current_row
 
    return previous_row[-1]

class ActionPlanner:

    def __init__(self, actions):
        self.actions = actions

    def explain_value(self, state, value):
        """ 
        This function uses a planner compute the given value from the current
        state. The function returns a plan.
        """
        initial = Node((tuple(state.items()), value))
        solution = next(BeamGS(initial, self.successorfn, self.testfn,
                               self.heuristicfn, initialBeamWidth=1))
        #solution = next(BestFGS(initial, self.successorfn, self.testfn, self.heuristicfn))

        return solution

    def successorfn(self, node):
        state, goal = node.state
        for action in self.actions:
            num_args = len(inspect.getargspec(self.actions[action]).args)
            for tupled_args in product(state, repeat=num_args):
                names = [a for a, v in tupled_args]
                values = [v for a, v in tupled_args]
                new_state = list(state)
                action_name = tuple([action] + names)
                ## TODO correctly output actions as relations.
                #print([action] + names)
                try:
                    new_state.append((action_name, self.actions[action](*values)))
                    yield Node((tuple(new_state), goal), node, action_name,
                               node.cost+1, node.depth+1)
                except Exception as e:
                    pass
                    #print(e)

    def testfn(self, node):
        s, goal = node.state
        for k, v in s:
            if v == goal:
                return True
        return False

    def heuristicfn(self, node):
        state, goal = node.state

        h = float('inf')
        if isinstance(goal, (int, float)):
            for a,v in state:
                if isinstance(v, (int, float)):
                    dist = (goal - v) * (goal - v)
                    if dist < h:
                        h = dist

        elif isinstance(goal, str):
            for a,v in state:
                if isinstance(v, str):
                    dist = levenshtein(v, goal)
                    if dist < h:
                        h = dist

        return h

def car(x):
    if isinstance(x, str) and len(x) > 1:
        return x[0]

def cdr(x):
    if isinstance(x, str) and len(x) > 2:
        return x[1:]

def append(x, y):
    if isinstance(x, str) and isinstance(y, str):
        return x + y

def tostring(x):
    return str(x)

def successor(x):
    return x+1

def add(x,y):
    return x+y

def subtract(x,y):
    return x-y

def multiply(x,y):
    return x*y

def divide(x,y):
    return x/y

def toFloat(x):
    return float(x)

if __name__ == "__main__":
    ap = ActionPlanner({'add': add,
                        'subtract': subtract,
                        'multiply': multiply,
                        'divide': divide,
                        'car': car,
                        'cdr': cdr,
                        'append': append,
                        'tostring': tostring})

    s = {'v1': {'value': 5},
         'v2': {'value': 3}}
    
    FlatS = flattenJSON(s)
    print(ap.explain_value(FlatS, 5*5+3))



