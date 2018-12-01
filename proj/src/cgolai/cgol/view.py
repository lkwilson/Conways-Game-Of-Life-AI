import contextlib
with contextlib.redirect_stdout(None):
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
        self.FLIP_COLOR = None
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
        if logo:
            pygame.display.set_icon(pygame.image.load(logo))
        pygame.display.set_caption(title)

        # colors (lower are ones I like)
        # hexes = "4edcff,4e83ff,4effca".split(',')  # colors near blue
        # hexes = "ffff89,c489ff,89c4ff".split(',')  # yellow purple blue
        # hexes = "67b4ff,67ffb2,ffffff".split(',')  # blue green black
        # hexes = "f0f0f0,a1a6ff,d0f8d5".split(',')  # white blue green
        hexes = "f0f0f0,0f0f0f,909090".split(',')  # white black grey
        colors = [self.color(hex_val=hex_val) for hex_val in hexes]
        self.DEAD_COLOR, self.ALIVE_COLOR, self.FLIP_COLOR = colors

        # tiles
        self.fit_model_size()

    def fit_model_size(self):
        model_height, model_width = self.ctrl.get_model().size
        self.tile_width = self.width // model_width
        self.tile_height = self.height // model_height

    def init_events(self):
        self.TICK_EVENT = pygame.USEREVENT + 1
        self.set_tick_period()
        self.key_map = {
            pygame.K_SPACE: self.play_pause,
            pygame.K_LEFT: self.left, pygame.K_h: self.left,
            pygame.K_RIGHT: self.right, pygame.K_l: self.right,
            pygame.K_UP: self.speed_up, pygame.K_k: self.speed_up,
            pygame.K_DOWN: self.speed_down, pygame.K_j: self.speed_down,
            pygame.K_q: self.quit,
            pygame.K_c: self.clear,
            pygame.K_r: self.reset,
            pygame.K_n: self.step,
            pygame.K_s: self.save,
            pygame.K_o: self.open,
            pygame.K_i: self.invert,
        }
        self.event_map = {
            pygame.QUIT: self.quit,
            pygame.MOUSEBUTTONUP: self.click_up,
            pygame.MOUSEBUTTONDOWN: self.click_down,
            pygame.MOUSEMOTION: self.drag,
            pygame.KEYDOWN: self.key_down,
            pygame.ACTIVEEVENT: self.active,
            self.TICK_EVENT: self.tick,
        }

    @staticmethod
    def print_controls():
        print('click/drag - revive/kill cells')
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
        # TODO: RAND

    @staticmethod
    def color(r=0, g=0, b=0, a=None, hex_val=None):
        """ A color building method

        arg types are integer: 0-256
        """
        if hex_val is not None:
            r, g, b = [int(x, 16) for x in (hex_val[:2], hex_val[2:4], hex_val[4:])]
        if a is None:
            return r, g, b
        else:
            return r, g, b, a

    def init_screen(self):
        self.bg = pygame.Surface(self.screen.get_size())
        self.bg = self.bg.convert()
        self.bg.fill(self.DEAD_COLOR)
        self.render()

    def run(self):
        """ Starts the gui and blocks, so be careful calling this one """
        while self.running:
            event = pygame.event.wait()
            if not self.handle_event(event):
                self.log('event:', event)
            for event in pygame.event.get():
                if not self.handle_event(event):
                    self.log('overflow event:', event)
            if self.model_update:
                self.render()

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
        """ handle events

        event handling doesn't have an un caught event handler function because
        there wasn't really a way to pass whether it was an overflow event or
        not.
        """
        if event.type in self.event_map:
            self.event_map[event.type](event)
            return True
        else:
            return False

    def active(self, event):
        if event.gain:
            self.render()

    def key_down(self, event):
        self.key_map.get(event.key, self.unhandled_key)(event)

    def unhandled_key(self, event):
        self.log('unhandled key:', event)

    def get_pos(self, pos):
        model = self.ctrl.get_model()
        col, row = pos
        row //= self.tile_height
        col //= self.tile_width
        if row < 0:
            row = 0
        elif row > self.ctrl.get_model().size[0]:
            row = self.ctrl.get_model().size[0]
        if col < 0:
            col = 0
        elif col > self.ctrl.get_model().size[1]:
            col = self.ctrl.get_model().size[1]
        return model.size[0]-1-row, col

    # RENDER
    def render(self):
        """ render """
        self.screen.blit(self.bg, (0, 0))
        self.render_model()
        pygame.display.update()
        self.model_update = False
        self.log('render')

    def render_model(self):
        """ only this function reads from the model """
        self.render_cells()

    def render_cells(self):
        model = self.ctrl.get_model()
        flip_map = model.get_flip_map()
        for row in range(model.size[0]):
            for col in range(model.size[1]):
                self.draw_cell(row, col, flip_map, model)

    def draw_cell(self, row, col, flip_map, model):
        xpos = col * self.tile_width
        ypos = (model.size[0] - 1 - row) * self.tile_height
        if model.get_cell(row, col):
            rect = [xpos, ypos, self.tile_width, self.tile_height]
            pygame.draw.rect(self.screen, self.ALIVE_COLOR, rect)
        if flip_map[row, col]:
            x_flip_pos = xpos + self.tile_width/4
            y_flip_pos = ypos + self.tile_height/4
            rect = [x_flip_pos, y_flip_pos, self.tile_width/2, self.tile_height/2]
            pygame.draw.ellipse(self.screen, self.FLIP_COLOR, rect)


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

    def right(self, event):
        self.ctrl.forward()

    def left(self, event):
        self.ctrl.back()

    def click_down(self, event):
        pos = self.get_pos(event.pos)
        self.ctrl.start_drag(pos)

    def drag(self, event):
        pos = self.get_pos(event.pos)
        self.ctrl.drag(pos)

    def click_up(self, event):
        self.ctrl.end_drag(self.get_pos(event.pos))

    def step(self, event):
        self.ctrl.step()

    def reset(self, event):
        self.ctrl.clear_flip()

    def invert(self, event):
        self.ctrl.invert()

    def save(self, event):
        self.ctrl.save()

    def open(self, event):
        self.ctrl.load()
        self.fit_model_size()
