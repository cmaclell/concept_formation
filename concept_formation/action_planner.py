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
from numbers import Number

import numpy as np

from py_search.search import Node
from py_search.search import Problem
from py_search.search import best_first_search
from py_search.search import compare_searches


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

class ActionPlannerProblem(Problem):

    def successor(self, node):
        state, goal = node.state
        actions = node.extra

        for action in actions:
            num_args = len(inspect.getargspec(actions[action]).args)
            for tupled_args in product(state, repeat=num_args):
                names = [a for a, v in tupled_args]
                values = [v for a, v in tupled_args]
                new_state = list(state)
                action_name = tuple([action] + names)
                ## TODO correctly output actions as relations.
                #print([action] + names)
                try:
                    new_state.append((action_name, actions[action](*values)))
                    path_cost = node.cost() + 1
                    yield Node((tuple(new_state), goal), node, action_name,
                               path_cost, node.extra)
                except Exception as e:
                    pass
                    #print(e)

    def goal_test(self, node):
        s, goal = node.state
        for k, v in s:
            if v == goal:
                return True
        return False

    def heuristic(self, node):
        state, goal = node.state

        h = float('inf')
        if isinstance(goal, Number):
            for a,v in state:
                try:
                    v = float(v)
                    vmin = -1000
                    vmax = 1000
                    diff = max(min((goal - v) * (goal - v), vmax), vmin)
                    dist = (diff + 1000) / 2000
                    if dist < h:
                        h = dist
                except:
                    pass

        #elif isinstance(goal, str):
        #    for a,v in state:
        #        if isinstance(v, str):
        #            diff = min(100, levenshtein(v, goal))
        #            diff = min(1000, levenshtein(v, goal)) - 500
        #            dist = 1/(1+math.exp(diff))
        #            if dist < h:
        #                h = dist

        return h

    def node_value(self, node):
        return node.cost() + self.heuristic(node)

class NoHeuristic(ActionPlannerProblem):

    def node_value(self, node):
        return node.cost()

class ActionPlanner:

    def __init__(self, actions):
        self.actions = actions

    def explain_value(self, state, value):
        """ 
        This function uses a planner compute the given value from the current
        state. The function returns a plan.
        """
        problem = ActionPlannerProblem((tuple(state.items()), value),
                                       extra=self.actions)

        try:
            solution = next(best_first_search(problem, cost_limit=4))
            return solution.path()[-1]

        except StopIteration:
            print("FAILED")
            return value



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
    if isinstance(x, str):
        x = float(x)
    if isinstance(y, str):
        y = float(y)
    return x+1

def add(x,y):
    if isinstance(x, str):
        x = float(x)
    if isinstance(y, str):
        y = float(y)
    return x+y

def subtract(x,y):
    if isinstance(x, str):
        x = float(x)
    if isinstance(y, str):
        y = float(y)
    return x-y

def multiply(x,y):
    if isinstance(x, str):
        x = float(x)
    if isinstance(y, str):
        y = float(y)
    return x*y

def divide(x,y):
    if isinstance(x, str):
        x = float(x)
    if isinstance(y, str):
        y = float(y)
    return x/y

def toFloat(x):
    return float(x)

if __name__ == "__main__":
    actions = {'add': add,
               'subtract': subtract, 
               'multiply': multiply, 
               'divide': divide
                        #'toFloat': toFloat,
                        #'car': car,
                        #'cdr': cdr,
                        #'append': append,
                        #'tostring': tostring
                       }
    ap = ActionPlanner(actions)

    s = {('value', 'v1'): '5',
         ('value', 'v2'): '3'}
    explain = 11
    
    #print(ap.explain_value(s, explain))

    problem = ActionPlannerProblem((tuple(s.items()), explain),
                                   extra=actions)
    problem2 = NoHeuristic((tuple(s.items()), explain),
                                   extra=actions)

    print(s)
    def cost_limited(problem):
        return best_first_search(problem, cost_limit=4)

    compare_searches([problem, problem2], [cost_limited])
