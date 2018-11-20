import os

import pygame

from .model import Model
from .view import View

class CgolAi:
    """ This class is the controller for the gui """
    def __init__(self, **config):
        """
        Model Config:
            board_size
            verbose
            filename

        View Config:
            title
            verbose
            logo
            window_size
        """
        # variables
        self.clickPos = None
        self.clickDrawPoses = None

        # model config
        board_size = config.get('board_size', (60, 80))
        self.verbose = config.get('verbose', False)
        filename = config.get('filename', None)

        # view config
        title = config.get('title', "Conway's Game of Life")
        logo = config.get('logo', os.path.join('res', 'logo.png'))
        scale = 10
        size = config.get('window_size', (board_size[1]*scale, board_size[0]*scale))

        # ctrl config
        self.tick_period = 100  # in millis
        self.tick_period_step = 10
        self.ticking = False

        self.model = Model(
            size=board_size,
            verbose=self.verbose,
            filename=filename,
        )
        self.view = View(
            self,
            title=title,
            logo=logo,
            size=size,
            verbose=self.verbose,
        )
        if config.get('print_controls', False):
            self.view.print_controls()

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

    def invert(self):
        self.model.invert()

    def start_drag(self, pos):
        self.clickPos = pos
        self.flip(self.clickPos)
        self.clickDrawPoses = {pos}

    def drag(self, pos):
        if self.clickPos is not None and pos not in self.clickDrawPoses:
            self.clickDrawPoses.add(pos)

    def end_drag(self, pos):
        self.clickDrawPoses.discard(self.clickPos)
        for pos in self.clickDrawPoses:
            self.flip(pos)
        self.clickPos = None
        self.clickDrawPoses = None

    def speed_up(self):
        self.tick_period -= self.tick_period_step
        if self.tick_period <= 0:
            self.tick_period = self.tick_period_step
        if self.ticking:  # it looks better
            self.tick()

    def speed_down(self):
        self.tick_period += self.tick_period_step
        if self.ticking:  # it looks better
            self.tick()
