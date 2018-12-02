# cs440 - Introduction to Artificial Intelligence

In this repo are all the assignments I've worked on over the semester for CS
440.

The most notable content in this repo is my Conway's Game of Life and
Reinforcement Learning AI package--`cgolai`. This package was created to
assist with the final project.

## Final Project Details:
* Package: `src/cgolai`
* Analysis Files: `proj/run_report.py`
* Final Project Report: `Report.ipynb`
* References Summary (per Honors Option Requirement): `ReportWritingComponent.docx`
* See [Final Project](#final-project-reinforcement-learning-to-control-conways-game-of-life)

## Assignments:

* Ininformed Search
* Iterative-Deepening Search
* A\*, IDS and Effective Branching Factor
* Reinforcement Learning (Hanoi, Q-function, SARSA)
* Neural Networks
* Min-Conflicts

# Final Project

## Reinforcement Learning to control Conway's Game of Life

I created a reinforcement agent capable of learning to control a mutated form
of Conway's Game of Life (Cgol). My package, `cgolai`, contains a built Cgol
gui, PyTorch Nueral Network, and Deep RL agent with Neural Network as
Q-function. See Report.ipynb for more information.

In this context, the "actions" the RL agent performs is reviving or eliminating
cells within the grid at any iteration. This is different from the original in
that the original game's grid can only be changed at the start. Further, the
Cgol grid is actually a toroid (going off the left of the screen puts you on
the right side). This is to limit the complexity since infinite grids are not
efficient.

**Results:** The RL agent typically generated still-life and then avoided
interacting with the system.

## Using `cgolai`

`cgolai` can be installed, or you can copy the project directory from `src` to
wherever you need it (be sure to check dependencies in `setup.py` if you do
this. I recommend using a virtual environment.

Notable components are the following:

* `cgolai.ai.NNTorch` Neural Network class
* `cgolai.ai.ProblemNN` Problem definition class (used by RL agent)
* `cgolai.ai.RL` RL Agent with Neural Network as Q-function
* `cgolai.ai.RLQ` RL Agent with dictionary as Q-function
* `cgolai.cgol.Model` The headless Cgol Model 
* `cgolai.cgol.Cgol` The class for the Cgol GUI (See [Running Cgol AI GUI](#running-cgol-ai-gui))
* `cgolai.cgol.CgolProblem` The problem definition of Cgol (used by RL agent)

## Running Cgol AI GUI

`python3 -m cgolai.cgol [options]`

## Directory Layout

* Assignments are stored in the `assign` directory. Each has a Jupyter Notebook
  for easy viewing.
* Source code for my final project is stored in `src` and `proj`.
* `src` contains the package I built to assist with the experimentation
  (including the Neural Network, RL agent, Cgol model, and Cgol GUI).
* `proj` contains the testing and experimentation I did to generate the results
  summarized in the report.
* `MANIFEST.in`, `Makefile`, and `setup.py` are used for the package.

## Installing `cgolai`

`python3 -m pip install .`

### Options

* `-c` will print the controlls to control the GUI.
* `-v` will print information to the console (mostly debugging information)
* `-f {save file}` the save file will be loaded if it exists.
* `-w {grid width}` width of Cgol toroid grid
* `-h {grid height}` height of the Cgol toroid grid

## Testing

`python3 -m unittest discover -s src`
