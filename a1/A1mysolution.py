#!/usr/bin/env python3

def removeVisitedNodes(children, expanded, unExpanded):
    """ Given a list of children, the expanded dictionary, and the unExpanded
    list of child-parent tuples, return a list of children not in expanded or
    unExpanded.

    Args:
        children(list):
        expanded(dict):
        unExpanded(list(tuple)):
    """
    # avoid building visited set if not needed
    if len(children) == 0:
        return children

    # build visited set
    visited = set(expanded.keys())
    for e in unExpanded:
        visited.add(e[0])

    # pythonic or nah?
    return list(filter(lambda child: child not in visited, children))

def treeSearch(startState, goalState, successorsf, breadthFirst):
    """
    Args:
        startState(object): initial state (can be any type)
        goalState(object): goal state (should have same type as startState)
        successorsf(function): action function
        breadthFirst(bool): breadth vs depth first toggle
    """
    expanded = {}
    unExpanded = [(startState, None)] # queue with end as top

    # trivial case
    if startState == goalState:
        return [startState]

    while unExpanded:
        state, parent = unExpanded.pop()
        children = successorsf(state)
        expanded[state] = parent
        children = removeVisitedNodes(children, expanded, unExpanded)
        if goalState in children:
            # this is backwards so I can push instead of shift an array
            # I guess I could use timeit to see if it's faster..
            solution = [goalState, state]
            while parent:
                solution.append(parent) # push
                parent = expanded[parent]
            solution.reverse()
            return solution
        children.sort()
        children.reverse()
        children = [(e, state) for e in children]
        if breadthFirst: # breadth first enqueues to bottom
            unExpanded = children + unExpanded
        else: # depth first pushes to top
            unExpanded = unExpanded + children

    return "Goal not found"

def breadthFirstSearch(startState, goalState, successorsf):
    return treeSearch(startState, goalState, successorsf, True)

def depthFirstSearch(startState, goalState, successorsf):
    return treeSearch(startState, goalState, successorsf, False)

def moveCamel(state, camel, space):
    state = list(state)
    tmp = state[camel]
    state[camel] = state[space]
    state[space] = tmp
    return tuple(state)

def camelSuccessorsf(state, S=' ', L='L', R='R'):
    """
    Args:
        state(list): a list of 'R', ' ', and 'L'. 'R' is a camel that can only
                move right, 'L' is a camel that can only move left, ' ' is an
                open space.
        S(str): an alias for space, default ' '
        L(str): an alias for left moving camel, default 'L'
        R(str): an alias for right moving camel, default 'R'

    Assumptions:
    * "...camels never go backwards...." => R can't move left, and L can't move
    right
    * "...camels will climb over each other, but only if there is a camel sized
    space on the other side." => all camels within one or two squares from ' '
    are able to move to the space as long as it doesn't violate the previous
    rule.

    Note: the problem doesn't require the camel be an L camel for them to climb
    over. This allows an L jump over an L, which is equivalent to both moving
    sequentially. Although equivalent, it raises those states in depth within
    the search tree, so it could yield interesting results. Since it's
    different than what the example assumes, I'll add a toggle to enable this
    assumption.
    """
    space = state.index(S)
    children = []
    if space-2 >= 0 and state[space-2] == R:
        children.append(moveCamel(state, space-2, space))
    if space-1 >= 0 and state[space-1] == R:
        children.append(moveCamel(state, space-1, space))
    if space+1 < len(state) and state[space+1] == L:
        children.append(moveCamel(state, space+1, space))
    if space+2 < len(state) and state[space+2] == L:
        children.append(moveCamel(state, space+2, space))
    return children
