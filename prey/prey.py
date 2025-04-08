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