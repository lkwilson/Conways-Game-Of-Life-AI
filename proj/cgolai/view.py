import pygame


class View:
    def __init__(self, ctrl, size, logo, title, fps=60, verbose=False):
        # variables
        self.verbose = verbose
        self.running = True
        self.bg = None
        self.TICK_EVENT = None
        self.key_map = None
        self.event_map = None
        self.fps = None
        self.size = None
        self.width = None
        self.height = None
        self.model_update = None
        self.screen = None
        self.DEAD_COLOR = None
        self.ALIVE_COLOR = None
        self.tile_width = None
        self.tile_height = None
        self.ctrl = None

        # init
        self.init_controller(ctrl)
        self.init_display(fps, logo, size, title)
        self.init_events()
        self.init_screen()

    def init_controller(self, ctrl):
        # set controller
        self.ctrl = ctrl
        self.ctrl.get_model().watch(self)

    def init_display(self, fps, logo, size, title):
        self.fps = fps
        self.size = size
        self.width, self.height = self.size
        self.model_update = True
        pygame.init()
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_icon(pygame.image.load(logo))
        pygame.display.set_caption(title)

        # colors
        self.DEAD_COLOR = self.color(0x67, 0xB4, 0xFF)
        self.ALIVE_COLOR = self.color(0x67, 0xFF, 0xB2)
        # self.FLIP_COLOR = self.color()

        # tiles
        model_height, model_width = self.ctrl.get_model().size
        self.tile_width = self.width // model_width
        self.tile_height = self.height // model_height

    def init_events(self):
        self.TICK_EVENT = pygame.USEREVENT + 1
        self.set_tick_period()
        self.print_controls()
        self.key_map = {
            pygame.K_SPACE: self.play_pause,
            pygame.K_LEFT: self.left, pygame.K_h: self.left,
            pygame.K_RIGHT: self.right, pygame.K_l: self.right,
            pygame.K_UP: self.speed_up, pygame.K_k: self.speed_up,
            pygame.K_DOWN: self.speed_down, pygame.K_j: self.speed_down,
            pygame.K_c: self.clear,
            pygame.K_r: self.reset,
            pygame.K_n: self.step,
            pygame.K_s: self.save,
            pygame.K_o: self.open,
            pygame.K_i: self.invert,
        }
        self.event_map = {
            pygame.QUIT: self.quit,
            pygame.MOUSEBUTTONDOWN: self.click,
            pygame.KEYDOWN: self.key_down,
            self.TICK_EVENT: self.tick,
        }

    def print_controls(self):
        print('space - play/pause')
        print('left/h - iterate back')
        print('right/l - iterate right')
        print('up/k - speed up iteration')
        print('down/j - speed down iteration')
        print('c - clear cells')
        print('r - reset / undo flips')
        print('n - generate next iteration')
        print('s - save iterations')
        print('o - open iterations')
        print('i - invert cells')

    def init_screen(self):
        self.bg = pygame.Surface(self.screen.get_size())
        self.bg = self.bg.convert()
        self.bg.fill(self.DEAD_COLOR)
        self.render()

    @staticmethod
    def color(r=0, g=0, b=0, a=None):
        """ A color building method

        arg types are integer: 0-256
        """
        if a is None:
            return r, g, b
        else:
            return r, g, b, a

    def run(self):
        """ Starts the gui and blocks, so be careful calling this one """
        while self.running:
            event = pygame.event.wait()
            self.log('event:', event)
            self.handle_event(event)
            for event in pygame.event.get():
                self.handle_event(event)
                self.log('overflow event:', event)
            if self.model_update:
                self.render()
                self.log('render')

    def log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def set_tick_period(self):
        if self.ctrl.ticking:
            pygame.time.set_timer(self.TICK_EVENT, self.ctrl.tick_period)
        else:
            pygame.time.set_timer(self.TICK_EVENT, 0)
        self.model_update = True

    def tick(self, event):
        self.ctrl.tick()

    def update(self):
        """ The model has been updated, and game needs to reflect that """
        self.model_update = True

    def handle_event(self, event):
        """ handle events """
        if event.type in self.event_map:
            self.event_map[event.type](event)

    def key_down(self, event):
        self.key_map.get(event.key, self.unhandled_key)(event)

    def unhandled_key(self, event):
        self.log('unhandled key:', event)

    # RENDER
    def render(self):
        """ render """
        self.screen.blit(self.bg, (0, 0))
        self.render_model()
        pygame.display.update()
        self.model_update = False

    def render_model(self):
        """ only this function reads from the model """
        model = self.ctrl.get_model()
        for row in range(model.size[0]):
            for col in range(model.size[1]):
                if model.get_cell(row, col):
                    xpos = col*self.tile_width
                    ypos = (model.size[0]-1-row)*self.tile_height
                    pygame.draw.rect(self.screen,
                                     self.ALIVE_COLOR,
                                     [xpos, ypos, self.tile_width, self.tile_height])

    # EVENT HANDLERS
    def speed_up(self, event):
        self.ctrl.speed_up()
        self.set_tick_period()

    def speed_down(self, event):
        self.ctrl.speed_down()
        self.set_tick_period()

    def play_pause(self, event):
        self.ctrl.play_pause()
        self.set_tick_period()

    def clear(self, event):
        self.ctrl.clear()

    def quit(self, event):
        self.running = False
        self.ctrl.save()

    def right(self, event):
        self.ctrl.forward()

    def left(self, event):
        self.ctrl.back()

    def click(self, event):
        model = self.ctrl.get_model()
        col, row = event.pos
        row //= self.tile_height
        col //= self.tile_width
        pos = (model.size[0]-1-row, col)
        self.ctrl.flip(pos)

    def step(self, event):
        self.ctrl.step()

    def reset(self, event):
        self.ctrl.clear_flip()

    def invert(self, event):
        self.ctrl.invert()

    def save(self, event):
        # TODO: get filename
        self.ctrl.save()

    def open(self, event):
        # TODO: get filename
        self.ctrl.load()
