import myappkit

import numpy as np
import pygame
from decimal import Decimal
from random import randint
from statistics import stdev

from numpy import zeros

direction = {'up': [0, 0],
             'right': [0, 1],
             'down': [1, 0],
             'left': [1, 1]}


class Snake(myappkit.Item):
    living_snakes = 0
    color = (0, 0, 130)

    def __init__(self, apple, init_lenght=2):
        super().__init__()

        self.tag = 'snake'
        Snake.living_snakes += 1
        self.ai = None
        self.ai_controlled = False
        self.environment = None
        self.confidence = 0
        self.moves = 0
        self.base_lifespan = 0
        self.lifespan = 0
        self.RL_input = []
        self.RL_output = []

        self.steps_per_apple = []
        self.aaaaaaa = 0
        self.completion = 0
        self.avg_path_len = 0

        self.grid_dim = (0, 0)
        self.origin = (0, 0)
        self.grid = None
        self.padding_size = 0
        self.padding_value = 0

        self.size = 20
        self.init_length = init_lenght
        self.length = self.init_length  # segments
        self.steps = 4  # steps per block
        self.currentStep = 0
        self.speed = Decimal(str(1. / self.steps))  # blocks per frame
        self.lost = False
        self.won = False

        self.coordinates = None
        self.direction = [direction['right']] * (self.length + 1)  # up, right, down, left
        self.nextTurn = direction['right']

        self.apple = apple
        self.apple.size = self.size

    def __lt__(self, other):
        return self.ai.fitness < other.ai.fitness

    def calcFitness(self):
        weights_start = 6.6
        self.completion = 100 * float(self.length - self.init_length) / float(
            (self.grid_dim[0] - 2 * self.padding_size) * (
                    self.grid_dim[1] - 2 * self.padding_size) - self.init_length)

        if len(self.steps_per_apple) == 0:
            self.avg_path_len = 100

        else:
            paths = [100 * i / float(
                    (self.grid_dim[0] - 2 * self.padding_size) * (self.grid_dim[1] - 2 * self.padding_size)
                ) for i in self.steps_per_apple]
            weights = np.arange(weights_start, 1, -(weights_start-1) / len(paths))
            if len(weights) > len(paths):
                weights = np.delete(weights, -1)
            elif len(weights) < len(paths):
                weights = np.append(weights, 1)
                
            self.avg_path_len = np.average(paths, weights=weights)

        self.avg_path_len = max(self.avg_path_len, 10)
        return self.ai.calcFitness(self.completion, self.avg_path_len)

    def end(self):
        Snake.living_snakes -= 1
        if self.ai_controlled:
            self.calcFitness()
            if self.lifespan > 0 and self.lost:
                self.punish()

    def setGrid(self, grid):
        self.grid_dim = grid.dim
        self.padding_size = grid.padding_size
        self.padding_value = grid.padding_value
        self.size = grid.block_size
        self.origin = grid.origin
        self.coordinates = [[(self.grid_dim[0] // 2) - i, self.grid_dim[1] // 2] for i in range(self.length)]
        self.base_lifespan = (self.grid_dim[0] - 2 * self.padding_size) * 10
        self.lifespan = self.base_lifespan

        self.apple.setGrid(grid)
        self.updateGrid()
        self.respawnApple()

    def updateGrid(self):
        self.grid = zeros(self.grid_dim)

        if self.padding_size > 0:
            for i in range(self.grid_dim[0]):
                for j in range(self.grid_dim[1]):
                    if (i < self.padding_size or i > self.grid_dim[0] - 1 - self.padding_size or
                            j < self.padding_size or j > self.grid_dim[0] - 1 - self.padding_size):
                        self.grid[i][j] = self.padding_value

        for i in range(self.length):
            self.grid[int(self.coordinates[i][0])][int(self.coordinates[i][1])] = i + 1

    def setSteps(self, s):
        self.steps = s  # steps per block
        self.speed = Decimal(str(1. / self.steps))  # blocks per frame

    def linkAI(self, ai, active):
        self.ai = ai
        self.ai_controlled = active

    def scanEnvironment(self):
        env = []

        x, y = int(self.coordinates[0][0]), int(self.coordinates[0][1])

        field_of_view = self.grid[x-2: x+3, y-2: y+3]

        filter_ = np.array([[0, 0, 1, 0, 0],
                            [0, 1, 1, 1, 0],
                            [1, 1, 0, 1, 1],
                            [0, 1, 1, 1, 0],
                            [0, 0, 1, 0, 0]])

        mask = filter_ == 1
        field_of_view = field_of_view[mask]

        for i in field_of_view:
            if i > 0:
                env.append(0.5)
            elif i == self.padding_value:
                env.append(1.)
            else:
                env.append(0.)

        ################################################

        # apple detection
        apple_x, apple_y = self.apple.coordinates[0], self.apple.coordinates[1]

        if x == apple_x:
            env.append(0)
        elif x > apple_x:
            env.append(1)
        else:
            env.append(-1)

        if y == apple_y:
            env.append(0)
        elif y > apple_y:
            env.append(1)
        else:
            env.append(-1)
        """if x == apple_x:
            if y > apple_y:
                env.append(1)
            else:
                env.append(-1)
        else:
            env.append(0)

        # horizontal
        if y == apple_y:
            if apple_x > x:
                env.append(1)
            else:
                env.append(-1)
        else:
            env.append(0)"""

        return env

    def handleEvent(self, event):

        if event.type == pygame.KEYDOWN:
            if not self.ai_controlled:
                if event.key == pygame.K_w and self.direction[0] != direction['down']:
                    self.nextTurn = direction['up']

                elif event.key == pygame.K_d and self.direction[0] != direction['left']:
                    self.nextTurn = direction['right']

                elif event.key == pygame.K_s and self.direction[0] != direction['up']:
                    self.nextTurn = direction['down']

                elif event.key == pygame.K_a and self.direction[0] != direction['right']:
                    self.nextTurn = direction['left']

            if event.key == pygame.K_TAB:
                self.ai_controlled = not self.ai_controlled

            if event.key == pygame.K_r:
                self.reset()

    def reset(self):
        Snake.living_snakes += 1
        self.length = self.init_length  # segments
        self.lost = False
        self.won = False
        self.currentStep = 0
        self.moves = 0
        self.lifespan = self.base_lifespan
        self.environment = None
        self.RL_input = []
        self.RL_output = []
        self.steps_per_apple = []
        self.aaaaaaa = 0
        self.completion = 0
        self.avg_path_len = 0

        self.coordinates = [[(self.grid_dim[0] // 2) - i, self.grid_dim[1] // 2] for i in range(self.length)]
        self.direction = [direction['right']] * (self.length + 1)  # up, right, down, left
        self.nextTurn = direction['right']
        self.updateGrid()

        self.apple.coor_index = -1
        self.respawnApple()

    def reward(self):
        y = [0, 0, 0, 0]
        if self.direction[0] == direction['up']:
            y[0] = 1

        if self.direction[0] == direction['right']:
            y[1] = 1

        if self.direction[0] == direction['down']:
            y[2] = 1

        if self.direction[0] == direction['left']:
            y[3] = 1

        self.RL_input.append(self.environment[0])
        self.RL_output.append(y)

    def safeDecisions(self):
        x, y = int(self.coordinates[1][0]), int(self.coordinates[1][1])
        decision_matrix = [1, 1, 1, 1]

        if (self.length > self.grid[x][y - 1] > 4) or self.grid[x][y - 1] == self.padding_value:
            decision_matrix[0] = 0

        if (self.length > self.grid[x + 1][y] > 4) or self.grid[x + 1][y] == self.padding_value:
            decision_matrix[1] = 0

        if (self.length > self.grid[x][y + 1] > 4) or self.grid[x][y + 1] == self.padding_value:
            decision_matrix[2] = 0

        if (self.length > self.grid[x - 1][y] > 4) or self.grid[x - 1][y] == self.padding_value:
            decision_matrix[3] = 0

        return decision_matrix

    def punish(self):
        decision = self.safeDecisions()
        n = decision.count(1)

        if n == 0:
            return

        else:
            decision = [element / n for element in decision]
            self.ai.fit([self.environment], [decision], 1, 1)

    def checkFood(self):
        return self.coordinates[0] == self.apple.coordinates

    def grow(self):
        self.direction.append(self.direction[-1])

        if self.direction[-1] == direction['up']:
            self.coordinates.append([self.coordinates[-1][0], self.coordinates[-1][1] + 1])

        if self.direction[-1] == direction['right']:
            self.coordinates.append([self.coordinates[-1][0] - 1, self.coordinates[-1][1]])

        if self.direction[-1] == direction['down']:
            self.coordinates.append([self.coordinates[-1][0], self.coordinates[-1][1] - 1])

        if self.direction[-1] == direction['left']:
            self.coordinates.append([self.coordinates[-1][0] + 1, self.coordinates[-1][1]])

        self.length += 1

    def updateLifespan(self):  # fix
        self.lifespan = (self.grid_dim[0] - 2 * self.padding_size) * (self.grid_dim[1] - 2 * self.padding_size)

    def checkCollision(self):
        x, y = int(self.coordinates[0][0]), int(self.coordinates[0][1])

        if (x not in range(self.padding_size, self.grid_dim[0] - self.padding_size)
                or y not in range(self.padding_size, self.grid_dim[1] - self.padding_size)
                or (3 < self.grid[x][y] < self.length)):
            return True

    def decide(self):
        self.environment = self.scanEnvironment()
        y = self.ai.predict([self.environment]).tolist()[0]

        self.confidence = round(100 * stdev(y) / stdev([1, 0, 0, 0]), 2)
        y = [round(a * 1000) for a in y]

        decided = False
        attempt = 0
        while not decided:
            attempt += 1
            prob = randint(1, 1000)

            if prob in range(0, y[0]) and self.direction[0] != direction['down']:
                self.nextTurn = direction['up']
                decided = True

            elif prob in range(y[0], y[0] + y[1]) and self.direction[0] != direction['left']:
                self.nextTurn = direction['right']
                decided = True

            elif prob in range(y[0] + y[1], y[0] + y[1] + y[2]) and self.direction[0] != direction['up']:
                self.nextTurn = direction['down']
                decided = True

            elif prob in range(y[0] + y[1] + y[2],
                               y[0] + y[1] + y[2] + y[3]) and self.direction[0] != direction['right']:
                self.nextTurn = direction['left']
                decided = True

            elif attempt == 10:
                decided = True

    def move(self):
        for i in range(len(self.coordinates)):
            x, y = Decimal(str(self.coordinates[i][0])), Decimal(str(self.coordinates[i][1]))

            if self.direction[i] == direction['up']:
                y -= self.speed
                self.coordinates[i][1] = float(y)

            if self.direction[i] == direction['right']:
                x += self.speed
                self.coordinates[i][0] = float(x)

            if self.direction[i] == direction['down']:
                y += self.speed
                self.coordinates[i][1] = float(y)

            if self.direction[i] == direction['left']:
                x -= self.speed
                self.coordinates[i][0] = float(x)

        self.currentStep += 1
        if self.currentStep == self.steps:
            self.currentStep = 0

    def checkLifespan(self):
        return self.lifespan <= 0

    def updateDirection(self):
        for i in range(1, len(self.direction)):
            self.direction[-i] = self.direction[-i - 1]
        self.direction[0] = self.nextTurn

    def checkWinCondition(self):
        return self.length >= (self.grid_dim[0] - 2 * self.padding_size) * (self.grid_dim[1] - 2 * self.padding_size)

    def respawnApple(self):
        apple = self.apple.respawn()
        while self.grid[apple[0]][apple[1]]:
            apple = self.apple.respawn()

        self.grid[apple[0]][apple[1]] = -1

    def update(self):
        if self.lost or self.won:
            return

        if self.currentStep == 0:
            if self.ai_controlled:
                self.decide()

            self.updateDirection()

            if self.checkFood():
                self.grow()
                self.steps_per_apple.append(self.moves - self.aaaaaaa)
                self.aaaaaaa = self.moves
                self.updateLifespan()
                # self.calcFitness()

                if self.checkWinCondition():
                    self.won = True
                    self.end()
                    return

                else:
                    self.respawnApple()

        self.move()

        if self.currentStep == 0:

            self.moves += 1
            self.lifespan -= 1

            if self.checkCollision() or self.checkLifespan():
                self.lost = True
                self.end()
                return

            self.updateGrid()

    def render(self, window):
        if not self.won:
            self.apple.render(window)

        self.ai.render(window)

        for i in self.coordinates:
            x, y = int(i[0]), int(i[1])
            x_margin, y_margin = 1, 1
            if x < self.grid_dim[0] and y < self.grid_dim[1]:
                if x < self.grid_dim[0] - 1:
                    if abs(self.grid[x][y] - self.grid[x + 1][y]) == 1 and self.grid[x + 1][y]:
                        x_margin = 0

                if y < self.grid_dim[1] - 1:
                    if abs(self.grid[x][y] - self.grid[x][y + 1]) == 1 and self.grid[x][y + 1]:
                        y_margin = 0

            rect = (1 + self.origin[0] + x * self.size,
                    1 + self.origin[1] + y * self.size,
                    self.size - x_margin, self.size - y_margin)
            pygame.draw.rect(window, Snake.color, rect)
