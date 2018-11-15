import pygame

class View:
    def __init__(self, ctrl, size, logo, title):
        pygame.init()

        # set controller
        self.ctrl = ctrl
        self.ctrl.getModel().watch(self)

        # build display
        self.size = size
        self.width, self.height = self.size
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_icon(pygame.image.load(logo))
        pygame.display.set_caption(title)
        self.buildDisplayComponents()

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

        # tiles
        modelHeight,modelWidth = self.ctrl.getModel().size
        self.tileWidth = self.width//modelWidth
        self.tileHeight = self.height//modelHeight

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
        clock = pygame.time.Clock()
        while self.running:
            clock.tick(self.ctrl.tickRate)
            self.ctrl.tick()
            for event in pygame.event.get():
                self.handleEvent(event)
            self.render()

    # RENDER
    def render(self):
        ''' render '''
        self.screen.blit(self.bg, (0,0))
        self.renderModel()
        pygame.display.flip()

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
        #self.render()

    def handleEvent(self, event):
        ''' handle events '''
        if event.type == pygame.QUIT:
            self.quit(event)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            self.click(event)
        elif event.type == pygame.KEYDOWN:
            self.keyDown(event)

    def keyDown(self, event):
        key = event.key

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

        # switch
        if key == space:
            self.playPause(event)
        elif key == left or key == h:
            self.left(event)
        elif key == right or key == l:
            self.right(event)
        elif key == up or key == k:
            self.speedUp(event)
        elif key == down or key == j:
            self.speedDown(event)
        elif key == c:
            self.clear(event)

    def speedUp(self, event):
        self.ctrl.speedUp()

    def speedDown(self, event):
        self.ctrl.speedDown()

    def playPause(self, event):
        self.ctrl.playPause()

    def clear(self, event):
        self.ctrl.clearFlip()

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
