import myappkit
from pygame.draw import line, rect
from pygame import KEYDOWN, K_g
from pygame.key import get_pressed


class Grid(myappkit.Item):
    def __init__(self, dim: tuple, block_size: int, padding_size=0, padding_value=0, padding_hidden=True,
                 origin=(0, 0)):
        super().__init__()

        self.tag = 'grid'
        self.color = (200, 200, 180)
        self.lines_color = (190, 190, 170)

        self.dim = tuple([d + 2*padding_size for d in dim])
        self.block_size = block_size
        self.padding_size = padding_size
        self.padding_value = padding_value
        self.padding_hidden = padding_hidden
        self.origin = origin
        self.rect = None
        self.x_lines = self.dim[0] + 1
        self.x_line_len = self.block_size * self.dim[1]
        self.y_lines = self.dim[1] + 1
        self.y_line_len = self.block_size * self.dim[0]
        self.shown = True

        if self.padding_hidden:
            self.rect = (self.origin[0] + self.block_size * self.padding_size,
                         self.origin[1] + self.block_size * self.padding_size,
                         (self.dim[0] - 2 * self.padding_size) * self.block_size,
                         (self.dim[1] - 2 * self.padding_size) * self.block_size
                         )
            self.x_lines = self.x_lines - 2*self.padding_size
            self.x_line_len = self.x_line_len - 2*self.padding_size*self.block_size
            self.y_lines = self.y_lines - 2*self.padding_size
            self.y_line_len = self.y_line_len - 2*self.padding_size*self.block_size

        else:
            self.rect = (self.origin[0], self.origin[1], self.dim[0]*self.block_size, self.dim[1]*self.block_size)

    def render(self, window):
        line(window, self.lines_color, (self.origin[0] - 1 + self.block_size * self.padding_size, 0),
             (self.origin[0] - 1 + self.block_size * self.padding_size, 800), 2)

        if self.shown:
            rect(window, self.color, self.rect)
            for i in range(self.x_lines):
                p1 = (self.origin[0] + i * self.block_size + self.block_size * self.padding_size * int(self.padding_hidden),
                      self.origin[1] + self.block_size * self.padding_size * int(self.padding_hidden))
                p2 = (self.origin[0] + i * self.block_size + self.block_size * self.padding_size * int(self.padding_hidden),
                      self.origin[1] + self.y_line_len + self.block_size * self.padding_size * int(self.padding_hidden))
                line(window, self.lines_color, p1, p2, 1)

            for i in range(self.y_lines):
                p1 = (self.origin[0] + self.block_size * self.padding_size * int(self.padding_hidden),
                      self.origin[1] + i * self.block_size + self.block_size * self.padding_size * int(self.padding_hidden))
                p2 = (self.origin[0] + self.x_line_len + self.block_size * self.padding_size * int(self.padding_hidden),
                      self.origin[1] + i * self.block_size + self.block_size * self.padding_size * int(self.padding_hidden))
                line(window, self.lines_color, p1, p2, 1)

    def handleEvent(self, event):
        if not event.type == KEYDOWN:
            return

        pressed_keys = get_pressed()
        if pressed_keys[K_g]:
            self.shown = not self.shown
