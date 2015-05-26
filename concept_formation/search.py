from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
from collections import deque
from heapq import heappush
from heapq import heappop
from heapq import heapify
import random

class Node(object):
    
    def __init__(self, state, parent=None, action=None, cost=0, depth=0,
                 extra=None):
        self.state = state
        self.extra = extra
        self.parent = parent
        self.action = action
        self.cost = cost
        self.depth = depth

    def getSolution(self):
        actions = []
        current = self
        while current.parent:
            actions.append(current.action)
            current = current.parent
        actions.reverse()
        return actions

    def __str__(self):
        return str(self.state) + str(self.extra)

    def __repr__(self):
        return repr(self.state)

    def __hash__(self):
        return hash(self.state)

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __ne__(self, other):
        return not self.__eq__(other)

def IDDFS(initial, successorFn, goalTestFn, initialDepthLimit=1):
    """
    Depth Limited Depth First Search
    """
    depthLimit = initialDepthLimit
    maxDepth = False
    nodeCount = 0
    successorFn = successorFn
    goalTestFn = goalTestFn

    while not maxDepth:
        maxDepth = True
        fringe = deque()
        getNext = fringe.pop
        fringe.append(initial)

        while fringe:
            current = getNext()
            nodeCount += 1
            if goalTestFn(current):
                print("Succeeded!")
                print("Nodes evaluated: %i" % nodeCount)
                yield current

            if current.depth < depthLimit:
                fringe.extend(successorFn(current))
            else:
                maxDepth = False
        depthLimit += 1
        print("Increasing depth limit to: %i" % depthLimit)

    print("Failed")
    print("Nodes evaluated: %i" % nodeCount)

def DLDFS(initial, successorFn, goalTestFn, depthLimit=float('inf')):
    """
    Depth Limited Depth First Search
    """
    fringe = deque()
    getNext = fringe.pop
    nodeCount = 0
    fringe.append(initial)
    successorFn = successorFn
    goalTestFn = goalTestFn

    while fringe:
        current = getNext()

        nodeCount += 1
        if goalTestFn(current):
            print("Succeeded!")
            print("Nodes evaluated: %i" % nodeCount)
            yield current

        if current.depth < depthLimit:
            fringe.extend(successorFn(current))

    print("Failed")
    print("Nodes evaluated: %i" % nodeCount)

def DepthFS(initial, successorFn, goalTestFn):
    """
    Depth first search
    """
    fringe = deque()
    getNext = fringe.pop
    nodeCount = 0
    fringe.append(initial)
    successorFn = successorFn
    goalTestFn = goalTestFn

    while fringe:
        current = getNext()

        nodeCount += 1
        if goalTestFn(current):
            print("Succeeded!")
            print("Nodes evaluated: %i" % nodeCount)
            yield current

        fringe.extend(successorFn(current))

    print("Failed")
    print("Nodes evaluated: %i" % nodeCount)

def DepthFGS(initial, successorFn, goalTestFn):
    """
    Depth first grid search (removes duplicates)
    """
    nodeCount = 0
    fringe = deque()
    closedList = set()
    openList = set()

    fringe.append(initial)
    openList.add(initial)

    while fringe:
        current = fringe.pop()
        openList.remove(current)
        closedList.add(current)

        if goalTestFn(current):
            #print("Succeeded!")
            #print("Nodes evaluated: %i" % nodeCount)
            yield current

        # Trick to push the looping into C execution
        added = [fringe.append(s) or openList.add(s) for s in
                 successorFn(current) if s not in closedList and s not in
                 openList]
        nodeCount += len(added)
        #for s in successorFn(current):
        #    nodeCount += 1
        #    if s not in closedList and s not in openList:
        #        fringe.append(s)
        #        openList.add(s)

    #print("Failed")
    #print("Nodes evaluated: %i" % nodeCount)

def BreadthFS(initial, successorFn, goalTestFn):
    """
    Breadth First search
    """
    nodeCount = 1
    fringe = deque()
    fringe.append(initial)

    while fringe:
        current = fringe.popleft()

        nodeCount += 1
        if goalTestFn(current):
            print("Succeeded!")
            print("Nodes evaluated: %i" % nodeCount)
            yield current

        for s in successorFn(current):
            nodeCount += 1
            fringe.append(s)

    print("Failed")
    print("Nodes evaluated: %i" % nodeCount)

def BreadthFGS(initial, successorFn, goalTestFn):
    """
    Breadth First Graph Search
    """
    nodeCount = 1
    fringe = deque()
    closedList = set()
    openList = set()
    fringe.append(initial)
    openList.add(initial)

    while fringe:
        current = fringe.popleft()
        openList.remove(current)
        closedList.add(current)

        if goalTestFn(current):
            #print("Succeeded!")
            #print("Nodes evaluated: %i" % nodeCount)
            yield current

        added = [fringe.append(s) or openList.add(s) for s in
                 successorFn(current) if s not in closedList and s not in
                 openList]
        nodeCount += len(added)
        #for s in successorFn(current):
        #    nodeCount += 1
        #    if s not in closedList and s not in openList:
        #        fringe.append(s)
        #        openList.add(s)

    #print("Failed")
    #print("Nodes evaluated: %i" % nodeCount)

def IDBFS(initial, successorFn, goalTestFn, heuristicFn,
          initialCostLimit=1, costInc=1):
    """
    Cost limited Best first search
    """
    nodeCount = 1
    successorFn = successorFn
    heuristicFn = heuristicFn
    goalTestFn = goalTestFn

    costMax = False
    costLimit = initialCostLimit

    while not costMax:
        costMax = True
        fringe = []
        openList = {}
        heappush(fringe, (0, nodeCount, initial))
        openList[initial] = 0

        while fringe:
            cost, counter, current = heappop(fringe)
            del openList[current]

            if goalTestFn(current):
                print("Succeeded!")
                print("Nodes evaluated: %i" % nodeCount)
                yield current

            for s in successorFn(current):
                nodeCount += 1
                sCost = s.cost + heuristicFn(s)
                if sCost <= costLimit:
                    if s in openList:
                        if sCost < openList[s]:
                            #fringe.remove((openList[s], s))
                            fringe = [e for e in fringe if e[2] != s]
                            heapify(fringe)
                            openList[s] = sCost
                            heappush(fringe, (sCost, nodeCount, s))
                    else:
                        openList[s] = sCost
                        heappush(fringe, (sCost, nodeCount, s))
                else:
                    costMax = False
        costLimit += costInc
        #print("Increasing cost limit to: %i" % costLimit)

    print("Failed")
    print("Nodes evaluated: %i" % nodeCount)

def CLBFS(initial, successorFn, goalTestFn, heuristicFn,
          costLimit=float('inf')):
    """
    Cost limited Best first search
    """
    nodeCount = 1

    fringe = []
    openList = {}
    heappush(fringe, (0, nodeCount, initial))
    openList[initial] = 0

    while fringe:
        cost, counter, current = heappop(fringe)
        del openList[current]

        if goalTestFn(current):
            print("Succeeded!")
            print("Nodes evaluated: %i" % nodeCount)
            yield current

        for s in successorFn(current):
            nodeCount += 1
            sCost = s.cost + heuristicFn(s)
            if sCost <= costLimit:
                if s in openList:
                    if sCost < openList[s]:
                        #fringe.remove((openList[s], len(fringe), s))
                        fringe = [e for e in fringe if e[2] != s]
        
                        heapify(fringe)
                        openList[s] = sCost
                        heappush(fringe, (sCost, nodeCount, s))
                else:
                    openList[s] = sCost
                    heappush(fringe, (sCost, nodeCount, s))
            else:
                print("Cost limit hit")

    print("Failed")
    print("Nodes evaluated: %i" % nodeCount)

def BestFS(initial, successorFn, goalTestFn, heuristicFn):
    """
    Best first search
    """
    nodeCount = 1
    fringe = []
    openList = {}
    heappush(fringe, (0, nodeCount, initial))
    openList[initial] = 0.0

    while fringe:
        cost, counter, current = heappop(fringe)
        del openList[current]

        if goalTestFn(current):
            print("Succeeded!")
            print("Nodes evaluated: %i" % nodeCount)
            yield current

        for s in successorFn(current):
            nodeCount += 1
            sCost = s.cost + heuristicFn(s)
            if s in openList:
                if sCost < openList[s]:
                    #fringe.remove((openList[s], ), s))
                    fringe = [e for e in fringe if e[2] != s]
                    heapify(fringe)
                    openList[s] = sCost
                    heappush(fringe, (sCost, nodeCount, s))
            else:
                openList[s] = sCost
                heappush(fringe, (sCost, nodeCount, s))

    print("Failed")
    print("Nodes evaluated: %i" % nodeCount)

def BestFGS(initial, successorFn, goalTestFn, heuristicFn):
    nodeCount = 1
    fringe = []
    closedList = {}
    openList = {}
    heappush(fringe, (0, nodeCount, initial))
    openList[initial] = 0.0

    while fringe:
        cost, counter, current = heappop(fringe)
        del openList[current]
        closedList[current] = cost

        if goalTestFn(current):
            #print("Succeeded!")
            #print("Nodes evaluated: %i" % nodeCount)
            yield current

        for s in successorFn(current):
            nodeCount += 1
            sCost = s.cost + heuristicFn(s)
            if s in openList:
                if sCost < openList[s]:
                    fringe = [e for e in fringe if e[2] != s]
                    heapify(fringe)
                    openList[s] = sCost
                    heappush(fringe, (sCost, nodeCount, s))
            elif s in closedList:
                if sCost < closedList[s]:
                    del closedList[s]
                    openList[s] = sCost
                    heappush(fringe, (sCost, nodeCount, s))
            else:
                openList[s] = sCost
                heappush(fringe, (sCost, nodeCount, s))

    #print("Failed")
    #print("Nodes evaluated: %i" % nodeCount)

def BeamS(initial, successorFn, goalTestFn, heuristicFn, initialBeamWidth=1):
    """
    Beam Search (allows duplicates)
    """
    nodeCount = 1
    beamMax = False
    beamWidth = initialBeamWidth
    while not beamMax:
        beamMax = True
        fringe = []
        openList = {}
        heappush(fringe, (0, nodeCount, initial))
        openList[initial] = 0

        while fringe:
            cost, counter, current = heappop(fringe)

            if goalTestFn(current):
                #print("Succeeded!")
                #print("Nodes evaluated: %i" % nodeCount)
                yield current

            for s in successorFn(current):
                nodeCount += 1
                sCost = s.cost + heuristicFn(s)
                if s in openList:
                    if sCost < openList[s]:
                        #fringe.remove((openList[s], s))
                        fringe = [e for e in fringe if e[2] != s]
                        heapify(fringe)
                        openList[s] = sCost
                        heappush(fringe, (sCost, nodeCount, s))
                else:
                    openList[s] = sCost
                    heappush(fringe, (sCost, nodeCount, s))

            if len(fringe) > beamWidth:
                best = []
                openList = {}
                for i in range(beamWidth):
                    if fringe:
                        c, s = heappop(fringe)
                        openList[s] = c
                        heappush(best, (cost, nodeCount, s))
                fringe = best
                beamMax = False
        
        beamWidth += 1
        #print("Increasing beam width to: %i" % beamWidth)

    #print("Failed")
    #print("Nodes evaluated: %i" % nodeCount)

def BeamGS(initial, successorFn, goalTestFn, heuristicFn, initialBeamWidth=1):
    """
    Beam Grid Search (no duplicates)
    """
    nodeCount = 1

    beamMax = False
    beamWidth = initialBeamWidth
    while not beamMax:
        beamMax = True
        fringe = []
        openList = {}
        closedList = {}
        heappush(fringe, (0, nodeCount, initial))
        openList[initial] = 0

        while fringe:
            cost, counter, current = heappop(fringe)
            del openList[current]
            closedList[current] = cost

            if goalTestFn(current):
                #print("Succeeded!")
                #print("Nodes evaluated: %i" % nodeCount)
                yield current

            for s in successorFn(current):
                nodeCount += 1
                sCost = s.cost + heuristicFn(s)
                if s in openList:
                    if sCost < openList[s]:
                        #fringe.remove((openList[s], s))
                        fringe = [e for e in fringe if e[2] != s]
                        heapify(fringe)
                        openList[s] = sCost
                        heappush(fringe, (sCost, nodeCount, s))
                elif s in closedList:
                    if sCost < closedList[s]:
                        del closedList[s]
                        openList[s] = sCost
                        heappush(fringe, (sCost, nodeCount, s))
                else:
                    openList[s] = sCost
                    heappush(fringe, (sCost, nodeCount, s))

            if len(fringe) > beamWidth:
                best = []
                openList = {}
                for i in range(beamWidth):
                    if fringe:
                        c, count, s = heappop(fringe)
                        openList[s] = c
                        heappush(best, (c, count, s))
                fringe = best
                beamMax = False
        
        beamWidth += 1
        print("Increasing beam width to: %i" % beamWidth)

    print("Failed")
    print("Nodes evaluated: %i" % nodeCount)

class eightPuzzle:

    def __init__(self):
        self.state = (0,1,2,3,4,5,6,7,8)

    def __hash__(self):
        return hash(self.state)

    def __eq__(self, other):
        if isinstance(other, eightPuzzle):
            return self.state == other.state
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def copy(self):
        new = eightPuzzle()
        new.state = tuple([i for i in self.state])
        return new

    def randomize(self):
        state = [0,1,2,3,4,5,6,7,8]
        self.state = tuple(state)

        for i in range(100):
            self.executeAction(random.choice([a for a in self.legalActions()]))
 
    def __repr__(self):
        return str(self)

    def __str__(self):
        return "%i%i%i\n%i%i%i\n%i%i%i" % (self.state[0], self.state[1],
                                           self.state[2], self.state[3],
                                           self.state[4], self.state[5],
                                           self.state[6], self.state[7],
                                           self.state[8])

    def executeAction(self, action):
        zeroIndex = self.state.index(0)
        successor = list(self.state)

        if action == 'up':
            successor[zeroIndex] = successor[zeroIndex + 3]
            successor[zeroIndex + 3] = 0
        elif action == 'left':
            successor[zeroIndex] = successor[zeroIndex + 1]
            successor[zeroIndex + 1] = 0
        elif action == 'right':
            successor[zeroIndex] = successor[zeroIndex - 1]
            successor[zeroIndex - 1] = 0
        elif action == 'down':
            successor[zeroIndex] = successor[zeroIndex - 3]
            successor[zeroIndex - 3] = 0

        self.state = tuple(successor)

    def legalActions(self):
        zeroIndex = self.state.index(0)

        if zeroIndex in set([0,1,2,3,4,5]):
            yield "up"
        if zeroIndex in set([0,3,6,1,4,7]):
            yield "left"
        if zeroIndex in set([2,5,8,1,4,7]):
            yield "right"
        if zeroIndex in set([3,4,5,6,7,8]):
            yield "down"

def successorFn8Puzzle(node):
    for action in node.state.legalActions():
        newState = node.state.copy()
        newState.executeAction(action)
        yield Node(newState, node, action, node.cost+1, node.depth+1)

def heuristicFn8Puzzle(node):
    """
    Currently the misplaced tile count
    """
    goal = eightPuzzle()
    h = 0
    for i,v in enumerate(node.state.state):
        if node.state.state[i] != goal.state[i]:
            h += 1
    return h

def goalTestFn8Puzzle(node):
    goal = eightPuzzle()
    return node.state == goal

if __name__ == "__main__":

    puzzle = eightPuzzle()
    puzzle.randomize()
    #puzzle.executeAction('left')
    #puzzle.executeAction('up')
    #puzzle.executeAction('up')

    initial = puzzle
    print("INITIAL:")
    print(puzzle)
    print()

    #print("BREADTH FIRST GRAPH SEARCH")
    #sol = next(BreadthFGS(Node(initial), successorFn8Puzzle, goalTestFn8Puzzle))
    #print("Solution Length = %i" % len(sol.getSolution()))
    #print()

    #print("DEPTH FIRST GRAPH SEARCH")
    #sol = next(DepthFGS(Node(initial), successorFn8Puzzle, goalTestFn8Puzzle))
    #print("Solution Length = %i" % len(sol.getSolution()))
    #print()
    #
    #print("BEST FIRST GRAPH SEARCH")
    #sol = next(BestFGS(Node(initial), successorFn8Puzzle, goalTestFn8Puzzle,
    #              heuristicFn8Puzzle))
    #print("Solution Length = %i" % len(sol.getSolution()))
    #print()

    print("BEAM GRAPH SEARCH")
    sol = next(BeamGS(Node(initial), successorFn8Puzzle, goalTestFn8Puzzle,
                  heuristicFn8Puzzle, 10))
    print("Solution Length = %i" % len(sol.getSolution()))
    print()

#    print("ITERATIVE DEEPENING DEPTH FIRST SEARCH")
#    sol = next(IDDFS(Node(initial), successorFn8Puzzle, goalTestFn8Puzzle))
#    print("Solution Length = %i" % len(sol.getSolution()))
#    print()
#
    #print("ITERATIVE DEEPENING BEST FIRST SEARCH")
    #sol = next(IDBFS(Node(initial), successorFn8Puzzle, goalTestFn8Puzzle,
    #              heuristicFn8Puzzle))
    #print("Solution Length = %i" % len(sol.getSolution()))
    #print()

    #print("BEAM SEARCH")
    #sol = next(BeamS(Node(initial), successorFn8Puzzle, goalTestFn8Puzzle,
    #              heuristicFn8Puzzle))
    #print("Solution Length = %i" % len(sol.getSolution()))
    #print()

    #print("BREADTH FIRST SEARCH")
    #sol = next(BreadthFS(Node(initial), successorFn8Puzzle, goalTestFn8Puzzle))
    #print("Solution Length = %i" % len(sol.getSolution()))
    #print()

    #print("DEPTH FIRST SEARCH")
    #sol = next(DepthFS(Node(initial), successorFn8Puzzle, goalTestFn8Puzzle))
    #print("Solution Length = %i" % len(sol.getSolution()))
    #print()



