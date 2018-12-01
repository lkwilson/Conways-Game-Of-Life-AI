import os
import numpy as np

print('\n======================= Code Execution =======================\n')

assignmentNumber = '2'

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

for func in ['iterativeDeepeningSearch', 'depthLimitedSearch',
             'findBlank_8p', 'actionsF_8p', 'takeActionF_8p', 'printPath_8p']:
    if func not in dir() or not callable(globals()[func]):
        print('CRITICAL ERROR: Function named \'{}\' is not defined'.format(func))
        print('  Check the spelling and capitalization of the function name.')


succs = {'a': ['b', 'z', 'd'], 'b':['a'], 'e':['z'], 'd':['y'], 'y':['z']}
print('\nSearching this graph:\n', succs)
def aF(state):
    import copy
    return copy.copy(succs.get(state,[]))
def tAF(state, action):
    return action
print('\nLooking for path from a to y with max depth of 1.')
path = iterativeDeepeningSearch('a', 'y', aF, tAF, 1)
if type(path) == str and path.lower() == 'cutoff':
    g += 5
    print(' 5/ 5 points. Your search correctly returned', path)
else:
    print(' 0/ 5 points. Your search should have returned ''cutoff''. You returned', path)

print('\nLooking for path from a to z with max depth of 5.')
path = iterativeDeepeningSearch('a', 'z', aF, tAF, 5)
if path == ['a', 'z']:
    g += 10
    print('10/10 points. Your search correctly returned', path)
else:
    print(' 0/10 points. Your search should have returned', ['a', 'z'])


print('\nTesting findBlank_8p([1, 2, 3, 4, 5, 6, 7, 0, 8])')
r, c = findBlank_8p([1, 2, 3, 4, 5, 6, 7, 0, 8])
if r == 2 and c == 1:
    g += 5
    print(' 5/ 5 points. Your findBlank_8p correctly returned', r, c)
else:
    print(' 0/ 5 points. Your findBlank_8p should have returned 2 1 but you returned', r, c)

print('\nTesting actionsF_8p([1, 2, 3, 4, 5, 6, 7, 0, 8])')
acts = list(actionsF_8p([1, 2, 3, 4, 5, 6, 7, 0, 8]))
correct = ['left', 'right', 'up']
if acts == correct:
    g += 10
    print('10/10 points. Your actionsF_8p correctly returned', acts)
else:
    print(' 0/10 points. Your actionsF_8p should have returned', correct, 'but you returned', acts)
    print('              Try ordering your action choices to match the correct answer.')
    
print('\nTesting takeActionF_8p([1, 2, 3, 4, 5, 6, 7, 0, 8],''up'')')
s = takeActionF_8p([1, 2, 3, 4, 5, 6, 7, 0, 8],'up')
correct = [1, 2, 3, 4, 0, 6, 7, 5, 8]
if s == correct:
    g += 10
    print('10/10 points. Your takeActionsF_8p correctly returned', s)
else:
    print(' 0/10 points. Your takeActionsF_8p should have returned', correct, 'but you returned', s)


print('\nTesting iterativeDeepeningSearch([1, 2, 3, 4, 5, 6, 7, 0, 8],[0, 2, 3, 1, 4,  6, 7, 5, 8], actionsF_8p, takeActionF_8p, 5)')
path = iterativeDeepeningSearch([1, 2, 3, 4, 5, 6, 7, 0, 8],[0, 2, 3, 1, 4,  6, 7, 5, 8], actionsF_8p, takeActionF_8p, 5)
correct = [[1, 2, 3, 4, 5, 6, 7, 0, 8], [1, 2, 3, 4, 0, 6, 7, 5, 8], [1, 2, 3, 0, 4, 6, 7, 5, 8], [0, 2, 3, 1, 4, 6, 7, 5, 8]]
if path == correct:
    g += 20
    print('20/20 points. Your search correctly returned', path)
else:
    print(' 0/20 points. Your search should have returned', correct, 'but you returned', path)

print('\nTesting iterativeDeepeningSearch([5, 2, 8, 0, 1, 4, 3, 7, 6], [0, 2, 3, 1, 4,  6, 7, 5, 8], actionsF_8p, takeActionF_8p, 10)')
path = iterativeDeepeningSearch([5, 2, 8, 0, 1, 4, 3, 7, 6],[0, 2, 3, 1, 4,  6, 7, 5, 8], actionsF_8p, takeActionF_8p, 10)
if type(path) == str and path.lower() == 'cutoff':
    g += 20
    print('20/20 points. Your search correctly returned', path)
else:
    print(' 0/20 points. Your search should have returned ''cutoff'', but you returned', path)


print('\n{} Grade is {}/80'.format(os.getcwd().split('/')[-1], g))
print('\nUp to 20 more points will be given based on the quality of your descriptions of the method and the results.')

