#!/usr/bin/env python3

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

def depthLimitedSearch(state, goalState, actionsF, takeActionF,
        depthLimit):
    if state == goalState:
        return []
    if depthLimit == 0:
        return 'cutoff'
    cutoffOccurred = False
    for action in actionsF(state):
        childState = takeActionF(state, action)
        result = depthLimitedSearch(childState, goalState, actionsF,
                takeActionF, depthLimit-1)
        if result == 'cutoff':
            cutoffOccurred = True
        elif result != 'failure':
            return [childState] + result
    return 'cutoff' if cutoffOccurred else 'failure'

def findBlank_8p(state):
    """return the row and column index for the location of the blank (the 0
    value)
    """
    index = state.index(0)
    return index//3, index%3

def actionsF_8p(state):
    """returns a list of up to four valid actions that can be applied in state.
    Return them in the order left, right, up, down, though only if each one is a
    valid action.

    0 1 2
    3 4 5
    6 7 8
    """
    ret_value = []
    r, c = findBlank_8p(state)
    if c != 0:
        ret_value.append("left")
    if c != 2:
        ret_value.append("right")
    if r != 0:
        ret_value.append("up")
    if r != 2:
        ret_value.append("down")
    return ret_value

def takeActionF_8p(state, action):
    """Return the state that results from applying action in state."""
    newState = state.copy()
    r, c = findBlank_8p(newState)
    zero_index = r*3 + c
    swap_index = zero_index
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
    return newState

def printState_8p(state):
    pState = '{} {} {}\n{} {} {}\n{} {} {}'.format(*state).replace('0', '-')
    print(pState)

def printPath_8p(startState, goalState, path):
    """Print a solution path in a readable form. You choose the format."""
    print("Path from")
    print()
    printState_8p(startState)
    print()
    print("to")
    print()
    printState_8p(goalState)
    print()
    print("takes {} actions.".format(len(path)))
    print()
    print("~~~ path ~~~")
    print()
    printState_8p(startState)
    if startState == path[0]:
        newPath = path[1:]
    else:
        newPath = path
    for state in newPath:
        print()
        printState_8p(state)

if __name__ == '__main__':
    board = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    printPath_8p(board, board, [board, board, board])
