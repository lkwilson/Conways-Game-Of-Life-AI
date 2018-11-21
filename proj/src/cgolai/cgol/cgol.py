import os
import pkg_resources

from .model import Model
from .view import View


class Cgol:
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

        # ctrl config
        self.tick_period_min = 1  # in millis
        self.tick_period = 150
        self.tick_period_step = 25
        self.ticking = False

        # model config
        board_size = config.get('board_size', (60, 80))
        self.verbose = config.get('verbose', False)
        filename = config.get('filename', None)
        self.model = Model(
            size=board_size,
            verbose=self.verbose,
            filename=filename,
            load=True,
        )
        self.log('board_size', self.model.size)

        # view config
        title = config.get('title', "Conway's Game of Life")
        logo_path = os.path.join('res', 'logo.png')
        logo_path = pkg_resources.resource_filename('cgolai', logo_path)
        logo = config.get('logo', logo_path)
        scale = 10
        default = (self.model.size[1]*scale, self.model.size[0]*scale)
        self.log('default', default)
        size = config.get('window_size', self.fit_model_size(default=default))
        self.log('view size', size)
        self.view = View(
            self,
            title=title,
            logo=logo,
            size=size,
            verbose=self.verbose,
        )

        if config.get('print_controls', False):
            self.view.print_controls()

    def fit_model_size(self, default=(800, 600)):
        """
        resize the window to better fit the model

        :param default: (width, height)
        :return:
        """
        width = default[0] - default[0] % self.model.size[1]
        height = default[1] - default[1] % self.model.size[0]
        return width, height

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

    def save(self):
        self.model.save()

    def load(self):
        self.model.load()
        # TODO: handle model size change

    def play_pause(self):
        self.ticking = not self.ticking

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
        self.clickDrawPoses.add(pos)
        self.clickDrawPoses.discard(self.clickPos)
        for pos in self.clickDrawPoses:
            self.flip(pos)
        self.clickPos = None
        self.clickDrawPoses = None

    def speed_up(self):
        self.tick_period -= self.tick_period_step
        if self.tick_period < self.tick_period_min:
            self.tick_period = self.tick_period_min
        if self.ticking:  # it looks better
            self.tick()
        self.log('tick_rate:', self.tick_period)

    def speed_down(self):
        if self.tick_period == self.tick_period_min:
            self.tick_period = 0
        self.tick_period += self.tick_period_step
        if self.ticking:  # it looks better
            self.tick()
        self.log('tick_rate:', self.tick_period)
