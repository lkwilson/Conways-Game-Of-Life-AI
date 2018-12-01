import os
import numpy as np

print('\n======================= Code Execution =======================\n')

assignmentNumber = '3'

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

for func in ['aStarSearch', 'iterativeDeepeningSearch',
             'actionsF_8p', 'takeActionF_8p', 'goalTestF_8p',
             'runExperiment']:
    if func not in dir() or not callable(globals()[func]):
        print('CRITICAL ERROR: Function named \'{}\' is not defined'.format(func))
        print('  Check the spelling and capitalization of the function name.')


print('\nTesting actionsF_8p([1, 2, 3, 4, 5, 6, 7, 0, 8])')
acts = actionsF_8p([1, 2, 3, 4, 5, 6, 7, 0, 8])
acts = list(acts)
correct = [('left', 1), ('right', 1), ('up', 1)]
if acts == correct:
    g += 5
    print('\n--- 5/5 points. Your actionsF_8p correctly returned', acts)
else:
    print('\n--- 0/5 points. Your actionsF_8p should have returned', correct, 'but you returned', acts)

print('\nTesting takeActionF_8p([1, 2, 3, 4, 5, 6, 7, 0, 8], (''up'', 1))')
s = takeActionF_8p([1, 2, 3, 4, 5, 6, 7, 0, 8], ('up', 1))
correct = ([1, 2, 3, 4, 0, 6, 7, 5, 8], 1)
if s == correct:
    g += 5
    print('\n--- 5/5 points. Your takeActionsF_8p correctly returned', s)
else:
    print('\n--- 0/5 points. Your takeActionsF_8p should have returned', correct, 'but you returned', s)

print('\nTesting goalTestF_8p([1, 2, 3, 4, 5, 6, 7, 0, 8], [1, 2, 3, 4, 5, 6, 7, 0, 8])')
if goalTestF_8p([1, 2, 3, 4, 5, 6, 7, 0, 8], [1, 2, 3, 4, 5, 6, 7, 0, 8]):
    g += 5
    print('\n--- 5/5 points. Your goalTestF_8p correctly True')
else:
    print('\n--- 0/5 points. Your goalTestF_8p did not return True')


print('\nTesting aStarSearch(1, 2, 3, 4, 5, 6, 7, 0, 8],')
print('                     actionsF_8p, takeActionF_8p,')
print('                     lambda s: goalTestF_8p(s, [0, 2, 3, 1, 4,  6, 7, 5, 8]),')
print('                     lambda s: h1_8p(s, [0, 2, 3, 1, 4,  6, 7, 5, 8]))')

path = aStarSearch([1, 2, 3, 4, 5, 6, 7, 0, 8], actionsF_8p, takeActionF_8p,
                   lambda s: goalTestF_8p(s, [0, 2, 3, 1, 4,  6, 7, 5, 8]),
                   lambda s: h1_8p(s, [0, 2, 3, 1, 4,  6, 7, 5, 8]))

# print('nNodesExpanded =',nNodesExpanded)
                   

correct = ([[1, 2, 3, 4, 5, 6, 7, 0, 8], [1, 2, 3, 4, 0, 6, 7, 5, 8], [1, 2, 3, 0, 4, 6, 7, 5, 8], [0, 2, 3, 1, 4, 6, 7, 5, 8]], 3)
if path == correct:
    g += 20
    print('\n--- 20/20 points. Your search correctly returned', path)
else:
    print('\n---  0/20 points. Your search should have returned', correct, 'but you returned', path)



print('\nTesting iterativeDeepeningSearch([1, 2, 3, 4, 5, 6, 7, 0, 8], ')
print('                                 [0, 2, 3, 1, 4,  6, 7, 5, 8],')
print('                                 actionsF_8p, takeActionF_8p, 10)')
path = iterativeDeepeningSearch([1, 2, 3, 4, 5, 6, 7, 0, 8],[0, 2, 3, 1, 4,  6, 7, 5, 8], actionsF_8p, takeActionF_8p, 10)
if path == correct[0]:
    g += 15
    print('\n--- 15/15 points. Your search correctly returned', path)
else:
    print('\n---  0/15 points. Your search should have returned', correct[0], 'but you returned', path)


print('\nTesting iterativeDeepeningSearch([5, 2, 8, 0, 1, 4, 3, 7, 6], ')
print('                                 [0, 2, 3, 1, 4,  6, 7, 5, 8],')
print('                                 actionsF_8p, takeActionF_8p, 10)')
path = iterativeDeepeningSearch([5, 2, 8, 0, 1, 4, 3, 7, 6],[0, 2, 3, 1, 4,  6, 7, 5, 8], actionsF_8p, takeActionF_8p, 10)
if type(path) == str and path.lower() == 'cutoff':
    g += 15
    print('\n--- 15/15 points. Your search correctly returned', path)
else:
    print('\n---  0/15 points. Your search should have returned ''cutoff'', but you returned', path)



print('\nTesting ebf(200, 6, 0.1)')

b = ebf(200, 6, 0.1)
correctb = 2.18537
if abs(b - correctb) < 0.3:
    g += 15
    print('\n--- 15/15 points. Your call to ebf correctly returned', b)
else:
    print('\n---  0/15 points. Your call to ebf returned', b, 'but it should be search should have returned', correctb)


print('\n{} Grade is {}/80'.format(os.getcwd().split('/')[-1], g))

print('\nUp to 20 more points will be given based on the quality of your descriptions of the method and the results.')


