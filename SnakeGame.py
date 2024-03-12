import myappkit
import random
from time import time
from random import seed
from copy import deepcopy

import numpy as np
import pygame
from os import mkdir as create_folder
from pathlib import Path

import AI
import Grid
import Snake
import settings

seed(0)


class GameplayActivity(myappkit.Activity):
    def __init__(self):
        super().__init__((210, 210, 190))

        self.probabilities = None
        self.efficiency_mode = False
        self._1st_snake = None
        self.grid = None
        self.start_time = time()
        self.n_parents = 0
        self.parents = None
        self.time_spent_training = 0
        self.currentSnake = 0
        self.highest_score = 0
        self.highest_fitness = 0
        self.avg_fitness = 0
        self.all_time_highest = 0
        self.population = 500
        self.remaining = self.population
        self.mutation_rate = 0
        self.x_vector = []
        self.y_vector = []
        self.sped_up_x = 1
        self.model_loaded = False
        self.paused = False

    def setGrid(self, grid: Grid.Grid):
        self.grid = grid
        self.grid.color = self.bgColor
        self.addItem(grid)

    def save(self):
        if not Path('data\\G' + str(self.items[self._1st_snake].ai.generation)).is_dir():
            create_folder('data\\G' + str(self.items[self._1st_snake].ai.generation))

        for i in range(self.n_parents):
            path = 'data\\G' + str(self.parents[i].generation) + '\\' + str(i) + '.pt'
            self.parents[i].save_model(path)

        path = 'data\\G' + str(self.items[self._1st_snake].ai.generation) + '\\details.txt'
        with open(path, 'w') as details:
            lines = [
                str(AI.GAModel.generation) + '\n',
                str(self.time_spent_training) + '\n',
                str(self.n_parents) + '\n'
            ]

            for p in self.probabilities:
                lines.append(str(p) + '\n')

            lines.append(str(self.all_time_highest) + '\n')
            lines.append(str(self.highest_fitness) + '\n')
            lines.append(str(self.avg_fitness) + '\n')

            details.writelines(lines)
            
        ####
        
        if not Path('load'):
            create_folder('load')

        for i in range(self.n_parents):
            path = 'load\\' + str(i) + '.pt'
            self.parents[i].save_model(path)

        path = 'load\\details.txt'
        with open(path, 'w') as details:
            lines = [
                str(AI.GAModel.generation) + '\n',
                str(self.time_spent_training) + '\n',
                str(self.n_parents) + '\n'
            ]

            for p in self.probabilities:
                lines.append(str(p) + '\n')

            lines.append(str(self.all_time_highest) + '\n')
            lines.append(str(self.highest_fitness) + '\n')
            lines.append(str(self.avg_fitness) + '\n')

            details.writelines(lines)

    def load_model(self):
        with open('load\\details.txt', 'r') as details_file:
            AI.GAModel.generation = eval(details_file.readline().strip())
            self.time_spent_training = eval(details_file.readline().strip())
            self.n_parents = eval(details_file.readline().strip())

            self.probabilities = []
            for i in range(self.n_parents):
                self.probabilities.append(eval(details_file.readline().strip()))

            self.all_time_highest = eval(details_file.readline().strip())
            self.highest_fitness = eval(details_file.readline().strip())
            self.avg_fitness = eval(details_file.readline().strip())

        self.model_loaded = True

    def add_snake(self, snake: Snake.Snake):
        if self._1st_snake is None:
            self._1st_snake = len(self.items)

        snake.setGrid(self.grid)
        if self.model_loaded:
            p_index = np.random.choice([i for i in range(self.n_parents)], p=self.probabilities)
            path = 'load\\' + str(p_index) + '.pt'
            snake.ai.load_model(path)
        self.addItem(snake)

    def handleEvent(self, event):
        for item in self.items:
            item.handleEvent(event)

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT:
                while self.currentSnake < self.population - 1:
                    self.currentSnake += 1
                    if not self.items[self.currentSnake + self._1st_snake].lost:
                        break

            if event.key == pygame.K_LEFT:
                while self.currentSnake > 0:
                    self.currentSnake -= 1
                    if not self.items[self.currentSnake + self._1st_snake].lost:
                        break

            if event.key == pygame.K_UP and self.sped_up_x < 5:
                self.sped_up_x += 1

            if event.key == pygame.K_DOWN and self.sped_up_x > 1:
                self.sped_up_x -= 1

    def render(self, window):
        if self.efficiency_mode:
            return

        window.fill(self.bgColor)

        if self._1st_snake > 0:
            for item in self.items[: self._1st_snake]:
                item.render(window)

        self.items[self.currentSnake + self._1st_snake].render(window)

        self.display_stats(window)

        pygame.display.update()

    def update_time(self):
        self.time_spent_training += time() - self.start_time
        self.start_time = time()

    def update_current_snake(self):
        if self.items[self.currentSnake + self._1st_snake].lost:
            highest = 0
            for i in range(self.population):
                if self.items[i + self._1st_snake].length > highest and not self.items[i + self._1st_snake].lost:
                    highest = self.items[i + self._1st_snake].length
                    self.currentSnake = i

    def update_score(self):
        self.highest_score = max([a.length - a.init_length for a in self.items[
                                                                    self._1st_snake: self.population + self._1st_snake
                                                                    ]])
        self.all_time_highest = max([self.all_time_highest, self.highest_score])

    def update_xy_vectors(self):
        self.x_vector = []
        self.y_vector = []
        for s in self.items[self._1st_snake: self.population + self._1st_snake]:
            for index in range(len(s.RL_input)):
                self.x_vector.append(s.RL_input[index])
                self.y_vector.append(s.RL_output[index])

    def RouletteSelection(self, n):
        self.n_parents = n
        self.parents = self.items[self._1st_snake: self.population + self._1st_snake]
        self.parents.sort(reverse=True)
        self.parents = deepcopy([p.ai for p in self.parents[:self.n_parents]])

        fitnesses = [max([p.fitness, 0]) for p in self.parents]
        sum_fitness = sum(fitnesses)
        self.avg_fitness = np.average(fitnesses)

        self.highest_fitness = max([self.avg_fitness, self.highest_fitness])

        self.probabilities = [f / sum_fitness for f in fitnesses]

    def reproduction(self):
        for snake in self.items[self._1st_snake: self.population + self._1st_snake]:
            parent = np.random.choice(self.parents, p=self.probabilities)
            snake.ai.copy(parent)

    def update_mutation_rate(self):
        self.mutation_rate = 0.1 - 0.095 * (
                self.all_time_highest / (
                    (self.grid.dim[0] - 2 * self.grid.padding_size) *
                    (self.grid.dim[1] - 2 * self.grid.padding_size) -
                    self.items[self._1st_snake].init_length
                )
        ) ** 0.1

    def mutate(self, chance=1.):
        if self.mutation_rate == 0:
            self.mutation_rate = 0.03

        for s in self.items[self._1st_snake: self.population + self._1st_snake]:
            if random.randint(1, 1000) <= chance*1000:
                s.ai.mutate(self.mutation_rate)

    def fit(self):
        for s in self.items[self._1st_snake: self.population + self._1st_snake]:
            s.ai.fit(self.x_vector, self.y_vector, 32, 1)

    def reset(self):
        for s in self.items[self._1st_snake: self.population + self._1st_snake]:
            s.reset()

    def update(self):
        if self.paused:
            return

        self.update_time()
        for i in range(self.sped_up_x):
            self.update_current_snake()
            self.update_score()

            super().update()

            if Snake.Snake.living_snakes == 0:
                # self.update_xy_vectors()
                self.update_mutation_rate()
                self.RouletteSelection(10)
                self.save()
                self.reproduction()
                self.mutate(chance=settings.mutation_chance)
                # self.fit()
                self.reset()
                AI.GAModel.generation += 1


    def display_stats(self, window):
        font = pygame.font.Font('C:\\Users\\cvemo\\Snake AI\\Courier Prime Code.ttf', 26)

        stats = 'Generation #' + str(self.items[self._1st_snake].ai.generation)
        text = font.render(stats, False, self.color)
        window.blit(text, (10, 10))

        stats = 'Remaining Snakes = ' + str(Snake.Snake.living_snakes) + '/' + str(self.population)
        text = font.render(stats, False, self.color)
        window.blit(text, (10, 10 + 35 * 1))

        stats = 'Score = ' + str(self.items[self.currentSnake + self._1st_snake].length -
                                 self.items[self.currentSnake + self._1st_snake].init_length) + ' (Highest = ' + str(
            self.highest_score) + ')'
        text = font.render(stats, False, self.color)
        window.blit(text, (10, 10 + 35 * 2))

        stats = 'All-time Highest = ' + str(self.all_time_highest)
        text = font.render(stats, False, self.color)
        window.blit(text, (10, 10 + 35 * 3))

        confidence = self.items[self.currentSnake + self._1st_snake].confidence
        stats = 'Confidence = ' + str(confidence) + '%'
        text = font.render(stats, False, self.color)
        window.blit(text, (10, 10 + 35 * 4))

        """stats = 'Lifespan = ' + str(self.items[self.currentSnake + self._1st_snake].lifespan)
        text = font.render(stats, False, self.color)
        window.blit(text, (10, 10 + 35 * 5))"""

        if self.items[self._1st_snake].ai.type == 'GA':

            stats = 'PGAF = ' + str(round(self.avg_fitness, 2))
            if self.avg_fitness < self.highest_fitness:
                stats += (' (-%' +
                          str(round(100 * (self.highest_fitness - self.avg_fitness) / self.highest_fitness, 1))) + ')'

            text = font.render(stats, False, self.color)
            window.blit(text, (10, 10 + 35 * 5))

            stats = 'Mutation Rate = ' + str(round(100 * self.mutation_rate, 2)) + '%'
            text = font.render(stats, False, self.color)
            window.blit(text, (10, 10 + 35 * 6))

            """stats = 'Weighted avg. path = ' + str(round(self.items[self._1st_snake + self.currentSnake].avg_path_len, 2))
            text = font.render(stats, False, self.color)
            window.blit(text, (10, 10 + 35 * 8))"""

            hours, minutes, secs = 0, 0, 0
            t = self.time_spent_training
            if t >= 3600:
                hours = int(t // 3600)

            if t - hours * 3600 >= 60:
                minutes = int((t - hours * 3600) // 60)

            secs = int(t - hours * 3600 - minutes * 60)

            stats = "{:02d}:".format(hours) + "{:02d}:".format(minutes) + "{:02d}".format(secs)
            text = font.render(stats, False, self.color)
            window.blit(text, (355, 10))

        stats = 'Generation #' + str(self.items[self._1st_snake].ai.generation)
        text = font.render(stats, False, self.color)
        window.blit(text, (10, 10))
