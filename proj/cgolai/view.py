import pygame

class View:
    def __init__(self, ctrl, size, logo, title):
        pygame.init()

        # set controller
        self.ctrl = ctrl

        # build display
        self.size = size
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_icon(pygame.image.load(logo))
        pygame.display.set_caption(title)

        # create background
        self.bg = pygame.Surface(self.screen.get_size())
        self.bg = self.bg.convert()
        self.bg.fill(self.color(250, 250, 250))

        # render
        self.render()

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
            #clock.tick(60)
            #self.ctrl.tick()
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
        model = self.ctrl.model
        # TODO: render model

    # EVENT HANDLERS
    def handleEvent(self, event):
        ''' handle events '''
        if event.type == pygame.QUIT:
            self.quit(event)
        # TODO: add more

    def quit(self, event):
        self.running = False
        self.ctrl.save()

    def right(self, event):
        pass # TODO

    def left(self, event):
        pass # TODO

    def click(self, event):
        pass # TODO

    def space(self, event):
        pass # TODO
