import importlib
import time
import random
import shelve

import pdb

import cellular
# reload(cellular)  # Python 2
importlib.reload(cellular)
import qlearn_mod_random as qlearn # to use the alternative exploration method
#import qlearn # to use standard exploration method

# reload(qlearn)  # Python 2
importlib.reload(cellular)  # Python 3

directions = 8

lookdist = 2
lookcells = []
for i in range(-lookdist,lookdist+1):
    for j in range(-lookdist,lookdist+1):
        if (abs(i) + abs(j) <= lookdist) and (i != 0 or j != 0):
            lookcells.append((i,j))

def pickRandomLocation():
    while 1:
        x = random.randrange(world.width)
        y = random.randrange(world.height)
        cell = world.getCell(x, y)
        if not (cell.wall or len(cell.agents) > 0):
            return cell


class Cell(cellular.Cell):
    wall = False

    def colour(self):
        if self.wall:
            return 'black'
        else:
            return 'white'

    def load(self, data):
        if data == 'X':
            self.wall = True
        else:
            self.wall = False


class Cat(cellular.Agent):
    cell = None
    score = 0
    colour = 'orange'

    def update(self):
        cell = self.cell
        if cell != mouse.cell:
            self.goTowards(mouse.cell)
            while cell == self.cell:
                self.goInDirection(random.randrange(directions))


class Cheese(cellular.Agent):
    colour = 'yellow'

    def update(self):
        pass


class Mouse(cellular.Agent):
    colour = 'gray'

    def __init__(self):
        self.ai = None
        self.eaten = 0
        self.fed = 0

    def update(self):
        # observe the reward and update the Q-value
        if self.cell == cat.cell:
            self.eaten += 1
            self.cell = pickRandomLocation()
            return

        if self.cell == cheese.cell:
            self.fed += 1
            cheese.cell = pickRandomLocation()

mouse = Mouse()
cat = Cat()
cheese = Cheese()

world = cellular.World(Cell, directions=directions, filename='../worlds/waco.txt')
world.age = 0

world.addAgent(cheese, cell=pickRandomLocation())
world.addAgent(cat, cell = pickRandomLocation())
world.addAgent(mouse, cell = pickRandomLocation())

world.update()

world.display.activate(size=30)
world.display.delay = 1

while 1:
    world.update(mouse.fed, mouse.eaten)