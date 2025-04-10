import numpy as np
from prey.utils import *
step = None

def static(robots, preys, p, grid):
    if len(legal_steps(robots, preys, p, grid)) == 0:
        return None
    return preys[p]
    
def random(robots, preys, p, grid):
    steps = legal_steps(robots, preys, p, grid)
    if len(steps) == 0:
        return None
    step = steps[np.random.randint(0, len(steps))]
    return preys[p] + step

def smartLinear(robots, preys, p, grid):
    global step
    if step is None:
        step = [None] * len(preys)
    steps = legal_steps(robots, preys, p, grid)
    if len(steps) == 0:
        return None
    if step[p] is None:
        target = random(robots, preys, p, grid)
        step[p] = target - preys[p]
        return target
    if not np.any(np.all(steps == step[p], axis=1)):
        angle0 = np.arctan2(step[p][1], step[p][0])
        angles = np.arctan2(steps[:, 1], steps[:, 0])
        diff = (angles - angle0 + np.pi) % (2 * np.pi) - np.pi
        step[p] = steps[np.argmin(np.abs(diff))]
    return preys[p] + step[p]