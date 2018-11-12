#!/usr/bin/env python3

import random

### GIVEN ###
def min_conflicts(vars, domains, constraints, neighbors, max_steps=1000): 
    """Solve a CSP by stochastic hillclimbing on the number of conflicts."""
    # Generate a complete assignment for all vars (probably with conflicts)
    current = {}
    for var in vars:
        val = min_conflicts_value(var, current, domains, constraints, neighbors)
        current[var] = val
    # Now repeatedly choose a random conflicted variable and change it
    for i in range(max_steps):
        conflicted = conflicted_vars(current,vars,constraints,neighbors)
        if not conflicted:
            return (current,i)
        var = random.choice(conflicted)
        val = min_conflicts_value(var, current, domains, constraints, neighbors)
        current[var] = val
    return (None,None)

def min_conflicts_value(var, current, domains, constraints, neighbors):
    """Return the value that will give var the least number of conflicts.
    If there is a tie, choose at random."""
    return argmin_random_tie(domains[var],
                             lambda val: nconflicts(var, val, current, constraints, neighbors)) 

def conflicted_vars(current,vars,constraints,neighbors):
    "Return a list of variables in current assignment that are in conflict"
    return [var for var in vars
            if nconflicts(var, current[var], current, constraints, neighbors) > 0]

def nconflicts(var, val, assignment, constraints, neighbors):
    "Return the number of conflicts var=val has with other variables."
    # Subclasses may implement this more efficiently
    def conflict(var2):
        val2 = assignment.get(var2, None)
        return val2 != None and not constraints(var, val, var2, val2)
    return len(list(filter(conflict, neighbors[var])))

def argmin_random_tie(seq, fn):
    """Return an element with lowest fn(seq[i]) score; break ties at random.
    Thus, for all s,f: argmin_random_tie(s, f) in argmin_list(s, f)"""
    best_score = fn(seq[0]); n = 0
    for x in seq:
        x_score = fn(x)
        if x_score < best_score:
            best, best_score = x, x_score; n = 1
        elif x_score == best_score:
            n += 1
            if random.randrange(n) == 0:
                    best = x
    return best

### END OF GIVEN ###

def schedule(classes, times, rooms, max_steps):
    """
    Args:
        - classes: list of all class names, like 'CS410'
        - times: list of all start times, like '10 am' and ' 1 pm'
        - rooms: list of all rooms, like 'CSB 325'
        - max_steps: maximum number of assignments to try
    
    Returns:
        - assignments: dictionary of values assigned to variables, like
          {'CS410': ('CSB 425', '10 am'), ...}
        - steps: actual number of assignments tested before solution found
          assignments and steps will each be None if solution was not found.
    """
    domain = []
    for room in rooms:
        for time in times:
            domain.append((room, time))
    domains = {var: domain for var in classes}
    neighbors = {var: [v for v in classes if v != var] for var in classes}
    return min_conflicts(classes, domains, constraints_ok, neighbors, max_steps)

def constraints_ok(class_name_1, value_1, class_name_2, value_2):
    """
    Args:
        - class_name_1: as above, like 'CS410'
        - value_1: tuple containing room and time.
        - class_name_2: a second class name
        - value_2: another tuple containing a room and time
    
    Returns:
        - result: True of the assignment of value_1 to class_name 1 and value_2
          to class_name 2 does not violate any constraints.  False otherwise.

    Constraints:
        - Two classes cannot meet in the same room at the same time.
        - Classes with the same first digit cannot meet at the same time,
          because students might take a subset of these in one semester. There
          is one exception to this rule. CS163 and CS164 can meet at the same
          time.
    """
    c1, c2 = class_name_1, class_name_2
    r1, t1 = value_1
    r2, t2 = value_2
    if r1==r2 and t1==t2: # constraint 1
        return False
    elif c1[2] == c2[2] and t1==t2: # constraint 2
        # exception
        return (c1 == 'CS163' and c2=='CS164') or (c1 == 'CS164' and c2=='CS163')
    else:
        return True

def display(assignments, rooms, times):
    """
    Args:
        - assignments: returned from call to your schedule function
        - rooms: list of all rooms as above
        - times: list of all times as above pass
    """
    if assignments == None:
        print("No solution found!")
        return

    # Prep
    printMap = {} # times -> rooms -> classes
    timeSpace = len(times[0])
    for className in assignments:
        room, time = assignments[className]
        if time not in printMap:
            printMap[time] = {}
        printMap[time][room] = className
    
    # Print
    timeSpace = len(times[-1])
    print(' '*timeSpace, end='')
    for room in rooms:
        print('{:>9}'.format(room), end='')
    print()
    print('-'*(timeSpace+9*len(rooms)))
    for time in times:
        print(time, end='')
        for room in rooms:
            toPrint = printMap.get(time, {}).get(room, '')
            print('{:>9}'.format(toPrint), end='')
        print()

if __name__ == '__main__':
    classes = ['CS160', 'CS163', 'CS164',
           'CS220', 'CS270', 'CS253',
           'CS320', 'CS314', 'CS356', 'CS370',
           'CS410', 'CS414', 'CS420', 'CS430', 'CS440', 'CS445', 'CS453', 'CS464',
           'CS510', 'CS514', 'CS535', 'CS540', 'CS545']

    times = [' 9 am',
            '10 am',
            '11 am',
            '12 pm',
            ' 1 pm',
            ' 2 pm',
            ' 3 pm',
            ' 4 pm']

    rooms = ['CSB 130', 'CSB 325', 'CSB 425']

    max_steps = 100
    assignments, steps = schedule(classes, times, rooms, max_steps)
    print('Took', steps, 'steps')
    print(assignments)
    print("display call:")
    display(assignments, rooms, times)
