import os
import copy
import signal

# Code to limit running time of specific parts of code.
#  To use, do this for example...
#
#  signal.alarm(seconds)
#  try:
#    ... run this ...
#  except TimeoutException:
#     print(' 0/8 points. Your depthFirstSearch did not terminate in', seconds/60, 'minutes.')
# Exception to signal exceeding time limit.


# class TimeoutException(Exception):
#     def __init__(self, *args, **kwargs):
#         Exception.__init__(self, *args, **kwargs)


# def timeout(signum, frame):
#     raise TimeoutException

# seconds = 60 * 5

# Comment next line for Python2
# signal.signal(signal.SIGALRM, timeout)

import os
import numpy as np

print('\n======================= Code Execution =======================\n')

assignmentNumber = '4'

if False:
    runningInNotebook = False
    print('========================RUNNING INSTRUCTOR''S SOLUTION!')
    # import A2mysolution as useThisCode
    # train = useThisCode.train
    # trainSGD = useThisCode.trainSGD
    # use = useThisCode.use
    # rmse = useThisCode.rmse
else:
    import subprocess, glob, pathlib
    filename = next(glob.iglob('*-A{}.ipynb'.format(assignmentNumber)), None)
    print('Extracting python code from notebook named \'{}\' and storing in notebookcode.py'.format(filename))
    if not filename:
        raise Exception('Please rename your notebook file to <Your Last Name>-A{}.ipynb'.format(assignmentNumber))
    with open('notebookcode.py', 'w') as outputFile:
        subprocess.call(['jupyter', 'nbconvert', '--to', 'script',
                         '*-A{}.ipynb'.format(assignmentNumber), '--stdout'], stdout=outputFile)
    # from https://stackoverflow.com/questions/30133278/import-only-functions-from-a-python-file
    import sys
    import ast
    import types
    with open('notebookcode.py') as fp:
        tree = ast.parse(fp.read(), 'eval')
    print('Removing all statements that are not function or class defs or import statements.')
    for node in tree.body[:]:
        if (not isinstance(node, ast.FunctionDef) and
            not isinstance(node, ast.ClassDef) and
            not isinstance(node, ast.Import) and
            not isinstance(node, ast.ImportFrom)):
            tree.body.remove(node)
    # Now write remaining code to py file and import it
    module = types.ModuleType('notebookcodeStripped')
    code = compile(tree, 'notebookcodeStripped.py', 'exec')
    sys.modules['notebookcodeStripped'] = module
    exec(code, module.__dict__)
    # import notebookcodeStripped as useThisCode
    from notebookcodeStripped import *


g = 0

for func in ['printState', 'validMoves', 'makeMove', 'trainQ', 'testQ']:
    if func not in dir() or not callable(globals()[func]):
        print('CRITICAL ERROR: Function named \'{}\' is not defined'.format(func))
        print('  Check the spelling and capitalization of the function name.')


state = [[1], [2], [3]]
print('\nTesting validMoves({})'.format(state))
try:
    moves = validMoves(state)
    if len(moves) == 3 and [1, 2] in moves and [1, 3] in moves and [2, 3] in moves:
        g += 10
        print('\n--- 10/10 points. Correctly returned {}'.format(moves))
    else:
        print('\n---  0/10 points. Incorrect moves. Should have returned [[1, 2], [1, 3], [2, 3]].')
except Exception as ex:
    print('\n--- validMoves raised the exception\n {}'.format(ex))


state = [[], [], [1, 2, 3]]
print('\nTesting validMoves({})'.format(state))
try:
    moves = validMoves(state)
    if len(moves) == 2 and [3, 1] in moves and [3, 2] in moves:
        g += 10
        print('\n--- 10/10 points. Correctly returned {}'.format(moves))
    else:
        print('\n---  0/10 points. Incorrect moves. Should have returned [[3, 1], [3, 2]].')
except Exception as ex:
    print('\n--- validMoves raised the exception\n {}'.format(ex))

def equalNestedLists(a, b):
    if len(a) != len(b):
        return False
    for i, sub in enumerate(a):
        if sub != b[i]:
            return False
    return True

state = [[], [], [1, 2, 3]]
move = [3, 2]
print('\nTesting makeMove({}, {})'.format(state, move))
try:
    newstate = makeMove(state, move)
    correctanswer = [[], [1], [2, 3]]
    if equalNestedLists(newstate, correctanswer):
        g += 10
        print('\n--- 10/10 points. Correctly returned {}'.format(newstate))
    else:
        print('\n---  0/10 points. Incorrect. Returned {}. Should return {}'.
              format(newstate, correctanswer))
except Exception as ex:
    print('\n--- makeMove raised the exception\n {}'.format(ex))

state = [[2], [3], [1]]
move = [1, 2]
print('\nTesting makeMove({}, {})'.format(state, move))
try:
    newstate = makeMove(state, move)
    correctanswer = [[], [2, 3], [1]]
    if equalNestedLists(newstate, correctanswer):
        g += 10
        print('\n--- 10/10 points. Correctly returned {}'.format(newstate))
    else:
        print('\n---  0/10 points. Incorrect. Returned {}. Should return {}'.
              format(newstate, correctanswer))
except Exception as ex:
    print('\n--- makeMove raised the exception\n {}'.format(ex))



state = [[1, 2, 3], [], []]
print('\nTesting   Q, steps = trainQ(1000, 0.5, 0.7, validMoves, makeMove).')
try:
    Q, steps = trainQ(1000, 0.5, 0.7, validMoves, makeMove)
    if 70 < len(Q) < 80:
        g += 10
        print('\n--- 10/10 points. Q dictionary has correct number of entries.')
    else:
        print('\n---  0/10 points. Q dictionary should have close to 76 entries. Yours has {}'.format(len(Q)))

    mn = np.array(steps).mean()
    if mn < 10:
        g += 10
        print('\n--- 10/10 points. The mean of the number of steps of {} is correctly < 10.'.format(mn))
    else:
        print('\n---  0/10 points. The mean of the number of steps is incorrect.  Yours is {}.  It should be less than 10.'.format(mn))
except Exception as ex:
    print('\n--- trainQ raised the exception\n {}'.format(ex))

print('\nTesting   path = testQ(Q, 20, validMoves, makeMove).')
try:
    path = testQ(Q, 20, validMoves, makeMove)
    if len(path) < 10:
        g += 20
        print('\n--- 20/20 points. Correctly returns path of length {}, which is correctly less than 10.'.format(len(path)))
    else:
        print('\n---  0/20 points. Did not return correct path.  Length of your path is {}. It should be less than 10.'.format(len(path)))
except Exception as ex:
    print('\n--- testQ raised the exception\n {}'.format(ex))
    

    
name = os.getcwd().split('/')[-1]

print('\n{} Execution Grade is {} / 80'.format(name, g))

print('\n Remaining 20 points will be based on your text describing the trainQ and testQ functions.')

print('\n{} FINAL GRADE is   / 100'.format(name))

print('\n{} EXTRA CREDIT is   / 1'.format(name))

