# Imports

import pandas as pd
import numpy as np
from math import sqrt
import time

# Algorithm Definitions

globalNodes = 0

class Node:
    def __init__(self, state, f=0, g=0, h=0):
        self.state = state
        self.f = f
        self.g = g
        self.h = h
        global globalNodes
        globalNodes += 1
    def __repr__(self):
        return "Node(" + repr(self.state) + ", f=" + repr(self.f) + \
               ", g=" + repr(self.g) + ", h=" + repr(self.h) + ")"

def aStarSearch(startState, actionsF, takeActionF, goalTestF, hF):
    h = hF(startState)
    startNode = Node(state=startState, f=0+h, g=0, h=h)
    return aStarSearchHelper(startNode, actionsF, takeActionF, goalTestF, hF, float('inf'))

def aStarSearchHelper(parentNode, actionsF, takeActionF, goalTestF, hF, fmax):
    if goalTestF(parentNode.state):
        return ([parentNode.state], parentNode.g)
    ## Construct list of children nodes with f, g, and h values
    actions = actionsF(parentNode.state)
    if not actions:
        return ("failure", float('inf'))
    children = []
    for action in actions:
        (childState,stepCost) = takeActionF(parentNode.state, action)
        h = hF(childState)
        g = parentNode.g + stepCost
        f = max(h+g, parentNode.f)
        childNode = Node(state=childState, f=f, g=g, h=h)
        children.append(childNode)
    while True:
        # find best child
        children.sort(key = lambda n: n.f) # sort by f value
        bestChild = children[0]
        if bestChild.f > fmax:
            return ("failure",bestChild.f)
        # next lowest f value
        alternativef = children[1].f if len(children) > 1 else float('inf')
        # expand best child, reassign its f value to be returned value
        result,bestChild.f = aStarSearchHelper(bestChild, actionsF, takeActionF, goalTestF,
                                            hF, min(fmax,alternativef))
        if result is not "failure":
            result.insert(0,parentNode.state)
            return (result, bestChild.f)

def depthLimitedSearch(state, goalState, actionsF, takeActionF, depthLimit):
    """ base cases """
    if state == goalState:
        return []
    if depthLimit == 0:
        return 'cutoff'
    
    """ exploration """
    cutoffOccurred = False
    for action in actionsF(state):
        """ explore each child node """
        global globalNodes
        globalNodes += 1
        childState = takeActionF(state, action)[0]
        result = depthLimitedSearch(childState, goalState, actionsF,
                takeActionF, depthLimit-1)
        
        """ handle post exploration """
        if result == 'cutoff': # depth limit reached
            cutoffOccurred = True
        elif result != 'failure':
            return [childState] + result
    
    """ full tree traversal """
    if cutoffOccurred:
        return 'cutoff'
    else:
        return 'failure'

def iterativeDeepeningSearch(startState, goalState, actionsF, takeActionF,
        maxDepth):
    for depth in range(maxDepth):
        results = depthLimitedSearch(startState, goalState, actionsF,
                takeActionF, depth)
        if results == 'failure':
            return 'failure'
        elif results != 'cutoff':
            return [startState] + results
    return 'cutoff'

def ebf(nNodes, depth, precision=0.01):
    """Returns effective branching factor, given the number of nodes expanded
    and depth reached during a search.

    We can use bisection with an initial lower bound of 1 and an upper bound of
    nNodes.
    Here's the math:

    Let $N = nNodes$ and $d = depth$. 

    Define the effective branching factor be $x$ which solves $N = 1 + x + x^2
    + ... + x^d$.

    Then, $x$ is the zeros of $f(x) = -N + 1 + x + x^2 + ... + x^d$.

    The derivative of this expression is $1 + x + x^2 + ... + x^(d-1)$.

    $x > 0 \implies x^i > 0 \implies f'(x) > 0$; therefore, the expression
    above is strictly increasing (and continuous). 

    $f(1) = 1 + 1 + ... + 1 - N = d + 1 - N < 0$ since $d + 1 \leq N$. If $f(1)
    = 0$, then $x = 1$. Otherwise, we have a lower bound for bisection, 1.

    $f(N) = 1 + N + N^2 + ... + N^d - N = 1 + N^2 + ... + N^d$, so $N \geq 0$,
    which it is, $\implies f(N) > 0$. If $d = 0$, then $f(N) = 0$, and $x = N$.
    Otherwise, we have an upperbound. Since $f$ is continuous and strictly
    increasing, by intermediate value theorem, we have a zero somewhere in the
    interval $[1, N]$. We can use bisection to find it.
    """
    lower = 1
    upper = nNodes
    while (upper - lower)/2 >= precision:
        lower, upper = ebfHelper(lower, upper, nNodes, depth)
    return (lower + upper) / 2 # mid point

def ebfPowerFunction(b, nNodes, depth):
    return (1 - b**(depth+1)) / (1 - b) - nNodes

def ebfHelper(lower, upper, nNodes, depth):
    #print('{} {}'.format(lower, upper))
    mid = (lower + upper) / 2
    midF = ebfPowerFunction(mid, nNodes, depth)
    if midF < 0:
        return mid, upper
    elif midF > 0:
        return lower, mid
    else: # soln found
        return mid, mid

# Problem function definitions

def findBlank_8p(state):
    """return the row and column index for the location of the
    blank (the 0 value)
    """
    index = state.index(0)
    return index//3, index%3

def actionsF_8p(state):
    """returns a list of up to four valid actions that can be applied in state.
    Return them in the order left, right, up, down, though only if each one is
    a valid action.
    """
    ret_value = []
    r, c = findBlank_8p(state)
    if c != 0:
        ret_value.append(("left", 1))
    if c != 2:
        ret_value.append(("right", 1))
    if r != 0:
        ret_value.append(("up", 1))
    if r != 2:
        ret_value.append(("down", 1))
    return ret_value

def takeActionF_8p(state, action):
    """Return the state that results from applying action in state."""
    newState = state.copy()
    r, c = findBlank_8p(newState)
    zero_index = r*3 + c
    swap_index = zero_index
    cost = action[1]
    action = action[0]
    if action == "left":
        swap_index -= 1
    elif action == "right":
        swap_index += 1
    elif action == "up":
        swap_index -= 3
    elif action == "down":
        swap_index += 3
    newState[zero_index] = newState[swap_index]
    newState[swap_index] = 0
    return (newState, cost)

def goalTestF_8p(state, goal):
    return state==goal

# Heuristic Functions

def h1_8p(state, goal):
    """h(state,goal)=0, for all states state and all goal states goal,"""
    return 0

def h2_8p(state, goal):
    """h(state,goal)=m, where m is the Manhattan distance that the blank is
    from its goal position,
    """
    rs, cs = findBlank_8p(state)
    rg, cg = findBlank_8p(goal)
    return abs(rg - rs) + abs(cg - cs)

def h3_8p(state, goal):
    """The other two heuristics are pretty bad, so I'll make a good one. I read
    the book, so I don't want to copy those, but I'll do something similar.
    Instead of Manhattan distance, I'll use Euclidean. Triangle inequality says
    that the book's heuristic is greater than or equal to mine, and therefore
    dominates it. As such, mine is also admissible. This heuristic loses on
    functionality though being more complex.

    There's also a thing about not making heuristics map to the set of real
    numbers, but that doesn't apply here since the smallest non zero value
    my heuristic could return is 1, and the cost to goal state isn't infinite.
    """
    dist = 0
    for i in state:
        if i == 0: continue # including zero would violate admissible
        rs,cs = i//3, i%3
        index = goal.index(i)
        rg,cg = index//3, index%3
        #dist += sqrt(abs(rg - rs)**2 + abs(cg - cs)**2)
        dist += abs(rg - rs) + abs(cg - cs)
        return dist

# Comparison

def runExperimentRender(goalResults):
    dataMatrix = [[] for _ in range(len(goalResults[0]))]
    for goal in goalResults:
        for i, alg in enumerate(goal):
            nodes, depth, time = alg
            dataMatrix[i].extend([depth, nodes, ebf(nodes, depth), time])
    return dataMatrix

def runExperiment(goalState1, goalState2, goalState3, hFs,
        initialState = [1, 2, 3, 4, 0, 5, 6, 7, 8],
        maxDepth = 15):
    """Prints goal state as pandas DataFrame"""
    global globalNodes
    goalResults = []
    goalStates = [goalState1, goalState2, goalState3]
    for goalState in goalStates:
        globalNodes = 1
        start = time.time()
        idsRes = iterativeDeepeningSearch(initialState,
                goalState,
                actionsF_8p,
                takeActionF_8p,
                maxDepth)
        end = time.time()
        nodes = globalNodes
        if not isinstance(idsRes, str):
            depth = len(idsRes) - 1
        else:
            depth = 0
        results = [(nodes, depth, end-start)]
        aStarRes = []
        for hF in hFs:
            globalNodes = 0
            start = time.time()
            res = aStarSearch(initialState,
                    lambda s : actionsF_8p(s),
                    takeActionF_8p,
                    lambda s : goalTestF_8p(s, goalState),
                    lambda s : hF(s, goalState))
            end = time.time()
            nodes = globalNodes
            depth = res[1]
            aStarRes.append((nodes, depth, end-start))
        results.extend(aStarRes)
        goalResults.append(results)
        #goalResults = [goal[alg(path, time), alg...], goal...]
    dataMatrix = runExperimentRender(goalResults)
    algsStr = ['IDS']
    for i in range(1, len(hFs)+1):
        algsStr.append('A*h{}'.format(i))
    df = renderResultsAsPd(algsStr, goalStates, dataMatrix)
    print(df)

def renderResultsAsPd(algs, goalStates, dataMatrix):
    """Renders results of algorithms as pandas.DataFrame.
    
    Args:
        algs - a list of the names of algorithms as strings
        goalStates - a list of goal states as lists
        dataMatrix - the data matrix with algorthims as columns and
                depth/node/ebf/time as columns
    
    https://stackoverflow.com/questions/32370402/giving-a-column-multiple-indexes-headers
    This link helped me understand pandas enough to write this function.
    """
    goalStatesStr = [str(x) for x in goalStates]
    subHeader = ['Depth','Nodes','EBF', 'Time (s)']
    header = pd.MultiIndex.from_product([goalStatesStr, subHeader],
                                        names=['Initial State','Algorithm'])
    return pd.DataFrame(dataMatrix, index=algs, columns=header)
