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
from random import choice
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
        actions = node.extra["actions"]

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
        epsilon = node.extra["epsilon"]

        for k, v in s:
            if isinstance(goal,Number) and isinstance(v,Number):
                if abs(goal-v) <= epsilon:
                    return True
            if v == goal:
                return True
        return False

    def heuristic(self, node):
        state, goal = node.state

        h = float('inf')

        is_number_goal = True

        try:
            goal = float(goal)
        except ValueError:
            is_number_goal = False

        if is_number_goal:
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
        else:
            for a,v in state:
                if isinstance(v, str):
                    vmin = -1000
                    vmax = 1000
                    diff = max(min(levenshtein(v, goal), vmax), vmin)
                    dist = (diff + 1000) / 2000
                    if dist < h:
                        h = dist

        return h

    def node_value(self, node):
        return node.cost() + self.heuristic(node)

class NoHeuristic(ActionPlannerProblem):

    def node_value(self, node):
        return node.cost()

class ActionPlanner:

    def __init__(self, actions, epsilon=0.0):
        self.actions = actions
        self.epsilon = epsilon

    def explain_value(self, state, value):
        """ 
        This function uses a planner compute the given value from the current
        state. The function returns a plan.
        """
        extra = {}
        extra["actions"] = self.actions
        extra["epsilon"] = self.epsilon

        state = {k:state[k] for k in state if k[0] != '_'}

        problem = ActionPlannerProblem((tuple(state.items()), value),
                                       extra=extra)
        try:
            solution = next(best_first_search(problem, cost_limit=4))
            if len(solution.path()) > 0:
                return solution.path()[-1]
            elif isinstance(value, Number):
                attrs = [attr for attr in state if isinstance(state[attr], Number) and abs(state[attr] - value) <= self.epsilon]
                return choice(attrs)
            else:
                attrs = [attr for attr in state if state[attr] == value]
                return choice(attrs)

        except StopIteration:
            print("FAILED")
            return value

#def car(x):
#    if isinstance(x, str) and len(x) > 1:
#        return x[0]
#
#def cdr(x):
#    if isinstance(x, str) and len(x) > 2:
#        return x[1:]
#
#def append(x, y):
#    if isinstance(x, str) and isinstance(y, str):
#        return x + y
#
#def tostring(x):
#    return str(x)

def add(x,y):
    if isinstance(x, str) and isinstance(y,str):
        x = float(x)
        y = float(y)
        return "%i" % (x+y)
    elif isinstance(x,Number) and isinstance(y,Number):
        return x+y
    else:
        raise TypeError("Arguments must both be strings or both be Numbers")


def subtract(x,y):
    if isinstance(x, str) and isinstance(y,str):
        x = float(x)
        y = float(y)
        return "%i" % (x-y)
    elif isinstance(x,Number) and isinstance(y,Number):
        return x-y
    else:
        raise TypeError("Arguments must both be strings or both be Numbers")
    

def multiply(x,y):
    if isinstance(x, str) and isinstance(y,str):
        x = float(x)
        y = float(y)
        return "%i" % (x*y)
    elif isinstance(x,Number) and isinstance(y,Number):
        return x*y
    else:
        raise TypeError("Arguments must both be strings or both be Numbers")
    

def divide(x,y):
    if isinstance(x, str) and isinstance(y,str):
        x = float(x)
        y = float(y)
        return "%i" % (x/y)
    elif isinstance(x,Number) and isinstance(y,Number):
        return x/y
    else:
        raise TypeError("Arguments must both be strings or both be Numbers")


def execute_plan(plan, state, actions):
    if plan in state:
        return state[plan]

    if not isinstance(plan, tuple):
        return plan

    args = tuple(execute_plan(ele, state, actions) for ele in plan[1:])
    action = plan[0]

    return actions[action](*args)

if __name__ == "__main__":
    actions = {'add': add,
               'subtract': subtract, 
               'multiply': multiply, 
               'divide': divide }
    epsilon = 0.85
    ap = ActionPlanner(actions,epsilon)

    s = {('value', 'v1'): -1.03}
    explain = -2.05
    
    plan = ap.explain_value(s, explain)

    print(plan)
    #print(execute_plan(plan, s, actions))

    extra = {}
    extra['actions'] = actions
    extra['epsilon'] = epsilon

    problem = ActionPlannerProblem((tuple(s.items()), explain), extra=extra)
    problem2 = NoHeuristic((tuple(s.items()), explain), extra=extra)

    #print(s)
    def cost_limited(problem):
        return best_first_search(problem, cost_limit=4)

    compare_searches([problem, problem2], [cost_limited])
