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
        board_size = config.get('board_size', (60, 80))
        self.verbose = config.get('verbose', False)

        # view config
        title = config.get('title', "Conway's Game of Life")
        logo = config.get('logo', os.path.join('res', 'logo.png'))
        scale = 10
        size = config.get('windowSize', (board_size[1]*scale, board_size[0]*scale))

        # ctrl config
        self.tick_period = 100  # in millis
        self.tick_period_step = 10
        self.ticking = False

        self.model = Model(
            size=board_size,
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

    def log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def tick(self):
        self.log('model tick')
        self.step()

    def get_model(self):
        return self.model

    # model facade
    def flip(self, pos):
        """ pos is (row,col) """
        self.model.flip(pos)

    def clear_flip(self):
        self.model.clear_flip()

    def clear(self):
        self.model.clear_board()

    def step(self):
        self.model.step()

    def back(self):
        self.model.back()

    def forward(self):
        self.model.forward()

    def save(self, filename=None):
        self.model.save(filename)

    def load(self, filename=None):
        self.model.load(filename)

    def play_pause(self):
        self.ticking = not self.ticking

    def close(self):
        self.model.close()

    def speed_up(self):
        self.tick_period -= self.tick_period_step
        if self.tick_period <= 0:
            self.tick_period = self.tick_period_step
        if self.ticking:  # it looks better
            self.tick()

    def invert(self):
        self.model.invert()

    def speed_down(self):
        self.tick_period += self.tick_period_step
        if self.ticking:  # it looks better
            self.tick()
