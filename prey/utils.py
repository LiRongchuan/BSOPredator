import numpy as np

SN = np.array([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]])

def collide(robots, preys, target):
    return np.any(np.all(target == robots, axis=1)) or np.any(np.all(target == preys, axis=1))

def legal_steps(robots, preys, p, grid):
    steps = np.array([], dtype=np.int32)
    for step in SN[0::2]:
        target = preys[p] + step
        if target[0] >= 0 and target[0] <= grid and\
            target[1] >= 0 and target[1] <= grid and\
            not collide(robots, preys, target):
            steps = np.append(steps, step, axis=0)
    for step in SN[1::2]:
        target = preys[p] + step
        hori = np.array([step[0], 0])
        vert = np.array([0, step[1]])
        if target[0] >= 0 and target[0] <= grid and\
            target[1] >= 0 and target[1] <= grid and\
            not collide(robots, preys, target) and not collide(robots, preys, preys[p] + hori) and not collide(robots, preys, preys[p] + vert):
            steps = np.append(steps, step, axis=0)
    return steps.reshape(-1, 2)