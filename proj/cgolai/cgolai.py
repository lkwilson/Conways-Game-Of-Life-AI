import pygame

from .model import Model
from .view import View

class CgolAi:
    """ This class is the controller for the gui """
    def __init__(self, config):
        self.model = Model(config['boardSize'])
        self.view = View(self, config['size'], config['logo'], config['title'])

    def run(self):
        self.view.run()

    def tick(self):
        print('tick')

    def getModel(self):
        return self.model

    # Model facade (for UI)
    def XYToRowCol(self, coord):
        """ coord=(x,y) in a graphics plane, origin in upper left """
        return y,x

    def flipPos(self, pos):
        """ pos is (row,col) """
        self.model.flip(pos)

    def step(self):
        self.model.step()

    def back(self):
        self.model.back()

    def forward(self):
        self.model.forward()

    def save(self):
        self.model.save()

    def load(self):
        self.model.load() #TODO: UI or model gets file? probably UI

    def close(self):
        self.model.close()
