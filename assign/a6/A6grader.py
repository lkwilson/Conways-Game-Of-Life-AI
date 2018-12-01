import os
import numpy as np

print('\n======================= Code Execution =======================\n')

assignmentNumber = '6'

if False:
    runningInNotebook = False
    print('========================RUNNING INSTRUCTOR''S SOLUTION!')
    from A6mysolution import *
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

for func in ['schedule', 'constraints_ok', 'display']:
    if func not in dir() or not callable(globals()[func]):
        print('CRITICAL ERROR: Function named \'{}\' is not defined'.format(func))
        print('  Check the spelling and capitalization of the function name.')


print("\nTesting constraints_ok(\'CS410\', (\'CSB 130\', \' 9 am\'), \'CS510\', (\'CSB 130\', \' 9 am\'))")

result = constraints_ok('CS410', ('CSB 130', ' 9 am'), 'CS510', ('CSB 130', ' 9 am'))
if not result:
    g += 10
    print('\n--- 10/10 points. Your constraints_ok function correctly returned False')
else:
    print('\n---  0/10 points. Your constraints_ok function incorrectly returned True but it should return False.')


print("\nTesting constraints_ok(\'CS410\', (\'CSB 130\', \' 9 am\'), \'CS510\', (\'CSB 130\', \'10 am\'))")

result = constraints_ok('CS410', ('CSB 130', ' 9 am'), 'CS510', ('CSB 130', '10 am'))
if result:
    g += 10
    print('\n--- 10/10 points. Your constraints_ok function correctly returned True')
else:
    print('\n---  0/10 points. Your constraints_ok function incorrectly returned False but it should return True.')

print("\nTesting constraints_ok('CS410', ('CSB 130', '10 am'), 'CS430', ('CSB 425', '10 am')")
result = constraints_ok('CS410', ('CSB 130', '10 am'), 'CS430', ('CSB 425', '10 am'))
if not result:
    g += 10
    print('\n--- 10/10 points. Your constraints_ok function correctly returned False')
else:
    print('\n---  0/10 points. Your constraints_ok function incorrectly returned True but it should return False.')


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

print("\nTesting  result, nsteps = schedule(classes, times, rooms, 100)")
result, nsteps = schedule(classes, times, rooms, 1000)
valid = True
for a,b in result.items():
    for c,d in result.items():
        if c != a:
            valid = valid and constraints_ok(a, b, c, d)

if len(result) == 23 and nsteps < 101 and valid:
    g += 30
    print('\n--- 30/30 points. Your schedule function returned a valid schedule.')
else:
    print('\n---  0/30 points. Your schedule function did not return a valid schedule.')


print("\nTesting  call to schedule again with two more classes.")
classes += ['CS898', 'CS899']
result, nsteps = schedule(classes, times, rooms, 1000)

if result is None:
    g += 30
    print('\n--- 30/30 points. Your schedule function correctly returned None.')
else:
    print('\n---  0/30 points. Your schedule function did not correctly return None.')


print('\n{} Execution grade is {} / 90'.format(os.getcwd().split('/')[-1], g))

print('\n---   / 10 points for result of your display function.')

print('\n{} FINAL GRADE is   / 100'.format(os.getcwd().split('/')[-1]))

print('\n{} EXTRA CREDIT grade is  / 1'.format(os.getcwd().split('/')[-1], g))



