import pygame

class View:
    def __init__(self, ctrl, size, logo, title, fps=60, verbose=False):
        pygame.init()
        self.verbose = verbose

        # set controller
        self.ctrl = ctrl
        self.ctrl.getModel().watch(self)

        # build display
        self.fps = fps
        self.size = size
        self.width, self.height = self.size
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_icon(pygame.image.load(logo))
        pygame.display.set_caption(title)
        self.buildDisplayComponents()

        # build events
        self.TICK_EVENT = pygame.USEREVENT + 1
        allowedEvents = [pygame.QUIT, pygame.KEYDOWN,
                self.TICK_EVENT, pygame.MOUSEBUTTONDOWN]
        #pygame.event.set_allowed(None)
        pygame.event.set_allowed(allowedEvents)
        self.setTickPeriod()
        self.buildKeyMap()
        self.buildEventMap()

        # create background
        self.bg = pygame.Surface(self.screen.get_size())
        self.bg = self.bg.convert()
        self.bg.fill(self.DEAD_COLOR)

        # render
        self.render()

    def buildDisplayComponents(self):
        # colors
        self.DEAD_COLOR = self.color(0x67, 0xB4, 0xFF)
        self.ALIVE_COLOR = self.color(0x67, 0xFF, 0xB2)
        #self.FLIP_COLOR = self.color()

        # tiles
        modelHeight,modelWidth = self.ctrl.getModel().size
        self.tileWidth = self.width//modelWidth
        self.tileHeight = self.height//modelHeight

    def buildKeyMap(self):
        # keys
        space = 32
        h = 104
        j = 106
        k = 107
        l = 108
        left = 276
        up = 273
        down = 274
        right = 275
        c = 99
        f = 114
        r = -1
        n = -1
        s = -1
        o = -1
        i = -1

        # building key map
        self.keymap = {
            space:self.playPause,
            left:self.left, h:self.left,
            right:self.right, l:self.right,
            up:self.speedUp, k:self.speedUp,
            down:self.speedDown, j:self.speedDown,
            c:self.clear,
            r:self.reset,
            n:self.next,
            s:self.save,
            o:self.open,
            i:self.invert,
        }

    def buildEventMap(self):
        self.eventMap = {
            pygame.QUIT:self.quit,
            pygame.MOUSEBUTTONDOWN:self.click,
            pygame.KEYDOWN:self.keyDown,
            self.TICK_EVENT:self.tick,
        }

    def color(self, r=0, g=0, b=0, a=None):
        ''' A color building method

        arg types are integer: 0-256
        '''
        if a:
            return (r, g, b, a)
        else:
            return (r, g, b)

    def run(self):
        ''' Starts the gui and blocks, so be careful calling this one '''
        self.running = True
        while self.running:
            event = pygame.event.wait()
            self.log('event:',event)
            self.handleEvent(event)
            for event in pygame.event.get(): # handle overflow
                self.handleEvent(event)
                self.log('overflow event:',event)
            if self.rerender:
                self.render()
                self.log('render')

    def setTickPeriod(self):
        if self.ctrl.ticking:
            pygame.time.set_timer(self.TICK_EVENT, self.ctrl.tickPeriod)
        else:
            pygame.time.set_timer(self.TICK_EVENT, 0)
        self.rerender = True

    # RENDER
    def tick(self, event):
        self.ctrl.tick()

    def render(self):
        ''' render '''
        self.screen.blit(self.bg, (0,0))
        self.renderModel()
        pygame.display.update()

    def renderModel(self):
        """ only this function reads from the model """
        model = self.ctrl.getModel()
        for row in range(model.size[0]):
            for col in range(model.size[1]):
                if model.getCell(row, col):
                    xpos = col*self.tileWidth
                    ypos = (model.size[0]-1-row)*self.tileHeight
                    pygame.draw.rect(self.screen,
                            self.ALIVE_COLOR,
                            [xpos, ypos, self.tileWidth, self.tileHeight])

    # EVENT HANDLERS
    def update(self):
        ''' The model has been updated, and game needs to reflect that '''
        self.rerender = True

    def log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def handleEvent(self, event):
        ''' handle events '''
        if event.type in self.eventMap:
            self.eventMap[event.type](event)

    def keyDown(self, event):
        # handle event
        key = event.key
        if key in keymap:
            self.keymap[key](event)
        else:
            self.log('key:',key)

    def speedUp(self, event):
        self.ctrl.speedUp()
        self.setTickPeriod()

    def speedDown(self, event):
        self.ctrl.speedDown()
        self.setTickPeriod()

    def playPause(self, event):
        self.ctrl.playPause()
        self.setTickPeriod()

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
        model = self.ctrl.getModel()
        col,row = event.pos
        row //= self.tileHeight
        col //= self.tileWidth
        pos = (model.size[0]-1-row, col)
        self.ctrl.flip(pos)

    def step(self, event):
        self.ctrl.step()

    def reset(self, event):
        self.ctrl.clearFlip()

    def save(self, event):
        pass

    def open(self, event):
        pass

    def invert(self, event):
        pass

    def next(self, event):
        pass
