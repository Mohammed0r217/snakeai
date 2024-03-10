import myappkit
from random import shuffle, randint
import pygame


class Apple(myappkit.Item):
    all_places = [[]]
    ready = False
    color = (255, 0, 0)

    def __init__(self):
        super().__init__()

        self.tag = 'apple'
        self.size = 20
        self.grid_size = (0, 0)
        self.origin = (0, 0)

        self.coordinates = [-1, -1]
        self.coor_index = -1
        self.spawn_random = True

        self.rect = pygame.Rect(0, 0, 0, 0)

    def setSpawnRandomness(self, _type: bool):
        self.spawn_random = _type

    @staticmethod
    def shuffle():
        if Apple.ready:
            shuffle(Apple.all_places)

    def setGrid(self, grid):
        self.grid_size = grid.dim
        self.size = grid.block_size
        self.origin = grid.origin

        if not Apple.ready:
            Apple.all_places = [[x, y] for x in range(self.grid_size[0]) for y in range(self.grid_size[1])]
            Apple.ready = True
            self.shuffle()

    def respawn(self):
        if not self.spawn_random:

            self.coor_index += 1
            if self.coor_index >= len(Apple.all_places):
                self.coor_index = 0

            self.coordinates = Apple.all_places[self.coor_index]

        else:
            new_coordinates = [randint(0, self.grid_size[0] - 1), randint(0, self.grid_size[1] - 1)]
            if new_coordinates == self.coordinates:
                self.respawn()

            self.coordinates = new_coordinates

        self.rect = (1 + self.origin[0] + self.coordinates[0] * self.size,
                     1 + self.origin[1] + self.coordinates[1] * self.size,
                     self.size - 1, self.size - 1)

        return self.coordinates

    def render(self, window):
        pygame.draw.rect(window, Apple.color, self.rect)
