import os

import pygame

from .model import Model
from .view import View

class CgolAi:
    """ This class is the controller for the gui """
    def __init__(self, **config):
        """
        Model Config:
            boardSize
            verbose

        View Config:
            title
            logo
            windowSize
        """
        # model config
        boardSize = config.get('boardSize', (60, 80))
        self.verbose = config.get('verbose', False)

        # view config
        title = config.get('title', "Conway's Game of Life")
        logo = config.get('logo', os.path.join('res', 'logo.png'))
        scale = 10
        size = config.get('windowSize', (boardSize[1]*scale,boardSize[0]*scale))

        # ctrl config
        self.tickPeriod = 100 # in millis
        self.tickPeriodStep = 10
        self.ticking = False

        self.model = Model(
                size=boardSize,
                verbose=self.verbose,
                )
        self.view = View(
                self,
                title=title,
                logo=logo,
                size=size,
                verbose=self.verbose,
                )

    def run(self):
        self.view.run()

    def tick(self):
        if self.verbose: print('model tick')
        self.model.step()

    def getModel(self):
        return self.model

    # Model facade (for UI)
    def XYToRowCol(self, coord):
        """ coord=(x,y) in a graphics plane, origin in upper left """
        return y,x

    def flip(self, pos):
        """ pos is (row,col) """
        self.model.flip(pos)

    def clearFlip(self):
        self.model.clearFlip()

    def clear(self):
        self.model.clearBoard()

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

    def playPause(self):
        self.ticking = not self.ticking

    def close(self):
        self.model.close()

    def speedUp(self):
        self.tickPeriod-=self.tickPeriodStep
        if self.tickPeriod < 0:
            self.tickPeriod = 0
        if self.ticking:
            self.tick()

    def speedDown(self):
        self.tickPeriod+=self.tickPeriodStep
        if self.ticking:
            self.tick()
