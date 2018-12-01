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

assignmentNumber = '5'

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

for func in ['trainNNs', 'summarize', 'bestNetwork']:
    if func not in dir() or not callable(globals()[func]):
        print('CRITICAL ERROR: Function named \'{}\' is not defined'.format(func))
        print('  Check the spelling and capitalization of the function name.')

def close(a, b, within=0.01):
    return abs(a-b) < within



print('\nTesting summarize([[[1,1], [1.2, 1.3, 1.4], [2.2, 2.3, 2.4], 0.5], [[2,2,2], [4.4, 4.3, 4.2], [6.5, 6.4, 6.3], 0.6]])')
try:
    summ = summarize([[[1,1], [1.2, 1.3, 1.4], [2.2, 2.3, 2.4], 0.5], [[2,2,2], [4.4, 4.3, 4.2], [6.5, 6.4, 6.3], 0.6]])
    correctAnswer = [[[1,1], 1.3, 2.3, 0.5], [[2,2,2], 4.3, 6.4, 0.6]]
    if (len(summ) == 2 and
        summ[0][0] == [1, 1] and close(summ[0][1], 1.3) and close(summ[0][2], 2.3) and summ[0][3] == 0.5 and
        summ[1][0] == [2, 2, 2] and close(summ[1][1], 4.3) and close(summ[1][2], 6.4) and summ[1][3] == 0.6):
        g += 10
        print('\n--- 10/10 points. Correctly returned {}'.format(summ))
    else:
        print('\n---  0/10 points. Incorrect. You returned {}, but correct answer is {}'.format(summ, correctAnswer))
except Exception as ex:
    print('\n--- 0/10 points. summarize raised the exception\n {}'.format(ex))

print('\nTesting bestNetwork([[[1, 1], 1.3, 2.3, 0.5], [[2, 2, 2], 4.3, 1.3, 0.6]])')
try:
    best = bestNetwork([[[1, 1], 1.3, 2.3, 0.5], [[2, 2, 2], 4.3, 1.3, 0.6]])
    correctAnswer = [[2, 2, 2], 4.3, 1.3, 0.6]
    if (len(best) == 4 and
        best[0] == [2, 2, 2] and close(best[1], 4.3) and close(best[2], 1.3) and best[3] == 0.6):
        g += 10
        print('\n--- 10/10 points. Correctly returned {}'.format(best))
    else:
        print('\n---  0/10 points. Incorrect. You returned {}, but correct answer is {}'.format(best, correctAnswer))

except Exception as ex:
    print('\n--- 0/10 points. bestNetwork raised the exception\n {}'.format(ex))


X = np.random.uniform(-1, 1, (100, 3))
T = np.hstack(((X**2 - 0.2*X**3).sum(axis=1,keepdims=True),
               (np.sin(X)).sum(axis=1,keepdims=True)))

print('''
X = np.random.uniform(-1, 1, (100, 3))
T = np.hstack(((X**2 - 0.2*X**3).sum(axis=1,keepdims=True),
               (np.sin(X)).sum(axis=1,keepdims=True)))
result = trainNNs(X, T, 0.7, [0, 5, 10, [20, 20]], 10, 100, False)''')
np.set_printoptions(precision=2)
try:
    result = trainNNs(X, T, 0.7, [0, 5, 10, [20, 20]], 10, 100, False)
    correctAnswer = [[0,
                      [0.38, 0.37, 0.39, 0.36, 0.33, 0.37, 0.38, 0.39, 0.38, 0.38],
                      [0.36, 0.39, 0.35, 0.42, 0.47, 0.40, 0.37, 0.34, 0.38, 0.38],
                      0.03],
                     [5,
                      [0.08, 0.07, 0.06, 0.08, 0.09, 0.05, 0.06, 0.08, 0.05, 0.09],
                      [0.11, 0.10, 0.06, 0.10, 0.13, 0.06, 0.12, 0.10, 0.07, 0.12],
                      0.84],
                     [10,
                      [0.05, 0.04, 0.05, 0.03, 0.04, 0.05, 0.04, 0.04, 0.04, 0.05],
                      [0.07, 0.05, 0.08, 0.08, 0.07, 0.09, 0.07, 0.07, 0.07, 0.08],
                      0.76],
                     [[20, 20],
                      [0.02, 0.02, 0.03, 0.02, 0.02, 0.02, 0.02, 0.03, 0.03, 0.03],
                      [0.06, 0.07, 0.05, 0.05, 0.04, 0.05, 0.07, 0.06, 0.06, 0.11],
                      1.52]]
    if (len(result) == 4 and
        [len(r) for r in result] == [4, 4, 4, 4] and
        [r[0] for r in result] == [0, 5, 10, [20, 20]] and
        [[len(r[1]), len(r[2])] for r in result] == [[10, 10], [10, 10], [10, 10], [10, 10]]) :
        g += 20
        print('\n--- 20/20 points. Correct.')
    else:
        print('\n---  0/20 points. Incorrect. You returned {}, but correct answer is {}'.format(result, correctAnswer))
except Exception as ex:
    print('\n--- 0/20 points. trainNNs raised the exception\n {}'.format(ex))


# for i in range(10):
#     print(bestNetwork(summarize(trainNNs(X, T, 0.7, [0, 5, 10, [20, 20]], 10, 100))))
    
print('\nTesting bestNetwork(summarize(result))')
try:
    best = bestNetwork(summarize(result))
    if len(best) == 4 and best[0] == [20, 20]:
        g += 20
        print('\n--- 20/20 points. You correctly found that network [20, 20] is best.')
    else:
        print('\n---  0/20 points. Network [20, 20] should be best.  You found {} as the best.'.format(best[0]))
except Exception as ex:
    print('\n--- 0/20 points. test raised the exception\n {}'.format(ex))


name = os.getcwd().split('/')[-1]

print('\n{} Execution Grade is {} / 60'.format(name, g))

print('\n======================= The regression data set =======================')
print('\n--- _ / 5 points. Read the data in energydata_complete.csv into variables Xenergy and Tenergy.')

print('\n--- _ / 5 points. Train some networks by calling the NeuralNetwork constructor and train method and plot the error trace to help you decide now many iterations might be needed.')

print('\n--- _ / 5 points. Try at least 10 different hidden layer structures using trainNNs.')

print('\n--- _ / 5 points. Train another network with your best hidden layer structure on 0.8 of the data and test it on remaining 0.2 of the data. Plot the predicted and actual Appliances energy use, and the predicted and actual lights energy use, in two separate plots. Discuss what you see.')

print('\n======================= Classification data set =======================')

print('\n--- _ / 5 points. Read the data in Frogs_MFCCs.csv into variables Xanuran and Tanuran.')

print('\n--- _ / 5 points. Train some networks by calling the NeuralNetwork constructor and train method and plot the error trace to help you decide now many iterations might be needed.')

print('\n--- _ / 5 points. Try at least 10 different hidden layer structures using trainNNs.')

print('\n--- _ / 5 points. Train another network with your best hidden layer structure on 0.8 of the data and test it on remaining 0.2 of the data. Plot the predicted and actual class labels. Discuss what you see.')

print('\n{} Notebook Grade is   / 40'.format(name))

print('\n{} FINAL GRADE is  / 100'.format(name))

print('\n\n{} EXTRA CREDIT is  / 1'.format(name))
