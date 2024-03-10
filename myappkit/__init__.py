import pygame, pkg_resources

class Item:
    def __init__(self, tag = None):
        self.tag = tag

    def handleEvent(self, event):
        pass

    def update(self):
        pass

    def render(self, window):
        pass

def coords_within_rect(coordinates, rect) -> bool:
    x, y = coordinates[0], coordinates[1]
    if x >= rect[0] and x < rect[0] + rect[2] and y > rect[1] and y < rect[1] + rect[3]:
        return True

class Button(Item):
    def __init__(self,
                 rect = (0, 0, 100, 50),
                 border_radii = (3, 3, 3, 3),
                 width=0,
                 default_color = (255, 255, 255),
                 hover_color = (200, 200, 200),
                 focus_color = (100, 100, 100),
                 text = None,
                 text_size = 20,
                 text_color = (0, 0, 0),
                 font_path = None,
                 click_action = 'press',
                 onClick = None,
                 onUnfocus = None,
                 active = True):
        
        super().__init__(tag = 'button')
        
        self.rect = rect
        self.border_radii = border_radii
        self.width = width
        self.default_color = default_color
        self.hover_color = hover_color
        self.focus_color = focus_color
        
        self.text = text
        self.text_color = text_color
        if font_path is None:
            font_path = pkg_resources.resource_filename('myappkit', 'Courier Prime Code.ttf')
            
        self.font = pygame.font.Font(font_path, text_size)
        
        self.hovered = False
        self.focused = False
        self.click_action = click_action # 'press' or 'focus'
        self.onClick = onClick
        self.onUnfocus = onUnfocus
        self.active = active
        
        
    def render(self, window):
        if not self.active:
            return

        if self.focused:
            color = self.focus_color
        elif self.hovered:
            color = self.hover_color
        else:
            color = self.default_color
            
        pygame.draw.rect(window, color, self.rect, width=self.width,
                         border_top_left_radius=self.border_radii[0],
                         border_top_right_radius=self.border_radii[1],
                         border_bottom_right_radius=self.border_radii[2],
                         border_bottom_left_radius=self.border_radii[3])
        
        text_render = self.font.render(self.text, False, self.text_color)
        text_rect = text_render.get_rect()
        text_x = self.rect[0] + 0.5*self.rect[2] - 0.5*text_rect.w
        text_y = self.rect[1] + 0.5*self.rect[3] - 0.5*text_rect.h
        window.blit(text_render, (text_x, text_y))

    def handleEvent(self, event):
        if not self.active:
            return

        if event.type == pygame.MOUSEMOTION:
            x, y = pygame.mouse.get_pos()
            if coords_within_rect((x, y), self.rect):
                self.hovered = True
            else:
                self.hovered = False
                
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.hovered:
                if self.click_action == 'focus':
                    self.focused = not self.focused
                    if self.focused:
                        if self.onClick is not None:
                            self.onClick()
                    elif self.onUnfocus is not None:
                        self.onUnfocus()
                        
                elif self.click_action == 'press':
                    self.focused = True
                
        elif event.type == pygame.MOUSEBUTTONUP and self.click_action == 'press':
            if self.focused and self.hovered and self.onClick is not None:
                    self.onClick()

            self.focused = False

class Activity:
    def __init__(self, bgColor=(255, 255, 255), frame_rate = 60):
        self.bgColor = pygame.Color(bgColor)
        self.color = pygame.Color((0, 0, 0))
        self.frame_rate = frame_rate
        self.items = []

    def setFrameRate(self, frame_rate):
        self.frame_rate = frame_rate

    def handleEvent(self, event):
        for item in self.items:
            item.handleEvent(event)

    def update(self):
        for item in self.items:
            item.update()

    def render(self, window):
        window.fill(self.bgColor)

        for item in self.items:
            item.render(window)

        pygame.display.update()

    def addItem(self, item):
        self.items.append(item)

class App:
    def __init__(self,
                 w = 600,
                 h = 400,
                 title = 'app'):
        
        if not pygame.get_init():
            pygame.init()
            
        self.window = pygame.display.set_mode((w, h))
        self.activities = {'home': Activity()} # default activity
        self.current_activity = 'home'
        self.FPS = pygame.time.Clock()
        self.setTitle(title)
        self.running = False

    def addActivity(self, name: str, activity: Activity):
        self.activities[name] = activity

    def resize(self, w, h):
        self.window = pygame.display.set_mode((w, h))

    def setCurrentActivity(self, activity_name):
        self.current_activity = activity_name

    def handleEvent(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            else:
                self.activities[self.current_activity].handleEvent(event)

    def update(self):
        self.activities[self.current_activity].update()

    def render(self):
        self.activities[self.current_activity].render(self.window)

    def run(self):
        self.running = True
        while self.running:
            self.handleEvent()
            self.update()
            self.render()

            self.FPS.tick(self.activities[self.current_activity].frame_rate)

        pygame.quit()
    
    def setTitle(self, title):
        pygame.display.set_caption(title)

    def setColor(self, color):
        self.color = pygame.Color(color)
