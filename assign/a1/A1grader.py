import os
import numpy as np

print('\n======================= Code Execution =======================\n')

assignmentNumber = '1'

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

for func in ['breadthFirstSearch', 'depthFirstSearch']:
    if func not in dir() or not callable(globals()[func]):
        print('CRITICAL ERROR: Function named \'{}\' is not defined'.format(func))
        print('  Check the spelling and capitalization of the function name.')

succs = {'a': ['b'], 'b':['c', 'd'], 'c':['e'], 'd':['f', 'i'], 'e':['g', 'h', 'i']}
print('Searching this graph:\n', succs)
def succsf(s):
    return copy.copy(succs.get(s,[]))

print('\nLooking for path from a to b.')
print('  Calling breadthFirstSearch(''a'', ''b'', successorsf)')
print('      and depthFirstSearch(''a'', ''b'', successorsf)')
bfsCorrect = ['a', 'b']
dfsCorrect = ['a', 'b']
bfs = breadthFirstSearch('a', 'b', succsf)
dfs = depthFirstSearch('a', 'b', succsf)

points = 10
if bfs == bfsCorrect:
    g += points
    print('{}/{} points. Your breadthFirstSearch found correct solution path of {}'.format(points,points,bfs))
else:
    print(' 0/{} points. Your breadthFirstSearch did not find correct solution path of {}'.format(points,bfsCorrect))

if dfs == dfsCorrect:
    g += points
    print('{}/{} points. Your depthFirstSearch found correct solution path of {}'.format(points,points,dfs))
else:
    print(' 0/{} points. Your depthFirstSearch did not find correct solution path of {}'.format(points,dfsCorrect))

print('\nLooking for path from a to i.')
print('  Calling breadthFirstSearch(''a'', ''i'', successorsf)')
print('      and depthFirstSearch(''a'', ''i'', successorsf)')
bfsCorrect = ['a', 'b', 'd', 'i']
dfsCorrect = ['a', 'b', 'c', 'e', 'i']
bfs = breadthFirstSearch('a', 'i', succsf)
dfs = depthFirstSearch('a', 'i', succsf)
points = 20
if bfs == bfsCorrect:
    g += points
    print('{}/{} points. Your breadthFirstSearch found correct solution path of {}'.format(points, points, bfs))
else:
    print(' 0/{} points. Your breadthFirstSearch did not find correct solution path of {}'.format(points, bfsCorrect))
if dfs == dfsCorrect:
    g += points
    print('{}/{} points. Your depthFirstSearch found correct solution path of {}'.format(points, points, dfs))
else:
    print(' 0/{} points. Your depthFirstSearch did not find correct solution path of {}'.format(points, dfsCorrect))

print('\nLooking for non-existant path from a to denver.')
print('  Calling breadthFirstSearch(''a'', ''denver'', successorsf)')
print('      and depthFirstSearch(''a'', ''denver'', successorsf)')

bfsCorrect = 'Goal not found'
dfsCorrect = 'Goal not found'
bfs = breadthFirstSearch('a', 'denver', succsf)
dfs = depthFirstSearch('a', 'denver', succsf)
points = 10
if bfs == bfsCorrect:
    g += points
    print('{}/{} points. Your breadthFirstSearch found correct solution path of {}'.format(points, points, bfs))
else:
    print(' 0/{} points. Your breadthFirstSearch did not find correct solution path of {}'.format(points, bfsCorrect))
if dfs == dfsCorrect:
    g += points
    print('{}/{} points. Your depthFirstSearch found correct solution path of {}'.format(points, points, dfs))
else:
    print(' 0/{} points. Your depthFirstSearch did not find correct solution path of {}'.format(points, dfsCorrect))


name = os.getcwd().split('/')[-1]

print('\n{} Execution Grade is {}/80'.format(name, g))

print('\n======================= Description of Code and Results =======================\n')

print('Up to 20 more points will be given based on the qualty of your descriptions of the method and the results.')
