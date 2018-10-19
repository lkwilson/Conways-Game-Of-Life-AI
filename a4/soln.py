from copy import copy, deepcopy
import numpy as np
import random

def printState(state):
    """prints the state in the form shown below"""
    stateCopy = deepcopy(state)
    maxLen = max([len(peg) for peg in stateCopy])
    for peg in stateCopy:
        while len(peg) < maxLen:
            peg.insert(0, ' ')
    for i in range(maxLen):
        for peg in stateCopy:
            print(peg[i], end=' ')
        print()
    print('-'*6)

def validMoves(state):
    """returns list of moves that are valid from state"""
    ret = []
    for i in range(len(state)):
        for j in range(len(state)):
            if i == j or len(state[i]) == 0:
                continue
            elif len(state[j]) == 0 or state[i][0] < state[j][0]:
                ret.append([i+1, j+1])
    return ret

def makeMove(state, move):
    """returns copy of state after move has been applied"""
    stateCopy = deepcopy(state)
    fromI = move[0]-1
    toI = move[1]-1
    stateCopy[toI].insert(0, stateCopy[fromI].pop(0))
    return stateCopy

def initHanoi(size):
    peg1 = list(range(1, size+1))
    return [peg1, [], []]

def greedyEpsilon(epsilon, Q, state, moves):
    if np.random.uniform() < epsilon:
        return random.choice(moves)
    else:
        Qs = np.array([Q.get(stateMoveTuple(state, move), 0) for move in moves])
        return moves[np.argmin(Qs)]

def isGoalState(state):
    return len(state[0]) == 0 and len(state[1]) == 0

def stateMoveTuple(state, move):
    tupleState = tuple(tuple(peg) for peg in state)
    return tuple((tupleState, tuple(move)))

def trainQ(nRepetitions, learningRate, epsilonDecayFactor, validMovesF,
        makeMoveF, size=3):
    """train the Q function for number of repetitions, decaying epsilon at
    start of each repetition. Returns Q and list or array of number of steps to
    reach goal for each repitition.
    """
    Q = {}
    stepsList = []
    epsilon = 1.0
    r = 1
    for i in range(nRepetitions):
        epsilon *= epsilonDecayFactor
        steps = 0
        state = initHanoi(size)
        done = False
        oldKey = None
        while not done:
            steps+=1
            moves = validMovesF(state)
            move = greedyEpsilon(epsilon, Q, state, moves)
            newState = makeMoveF(state, move)
            key = stateMoveTuple(state, move)

            if key not in Q:
                Q[key] = 0

            if isGoalState(newState):
                Q[key] = 1
                done = True

            if steps > 1:
                TDError = 1 + Q[key] - Q[oldKey]
                Q[oldKey] += learningRate * TDError

            state = newState
            oldKey = key
        stepsList.append(steps)
    return Q, np.array(stepsList)

def testQ(Q, maxSteps, validMovesF, makeMoveF, size=3):
    """without updating Q, use Q to find greedy action each step until goal is
    found. Return path of states.
    """
    state = initHanoi(size)
    path = [state]
    for i in range(maxSteps):
        move = greedyEpsilon(0, Q, state, validMovesF(state))
        state = makeMoveF(state, move)
        path.append(state)
        if isGoalState(state):
            return path
    return 'Goal not reached in {} steps'.format(maxSteps)

