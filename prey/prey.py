import numpy as np

SN = np.array([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]])

def collide(robots, prey):
    for robot in robots:
        if np.all(prey == robot):
            return True
    return False

def legal_steps(robots, prey, grid):
    steps = np.array([], dtype=np.int32)
    for step in SN[0::2]:
        target = prey + step
        if target[0] >= 0 and target[0] <= grid and\
            target[1] >= 0 and target[1] <= grid and\
            not collide(robots, target):
            steps = np.append(steps, step, axis=0)
    for step in SN[1::2]:
        target = prey + step
        hori = np.array([step[0], 0])
        vert = np.array([0, step[1]])
        if target[0] >= 0 and target[0] <= grid and\
            target[1] >= 0 and target[1] <= grid and\
            (not collide(robots, target)) and not (collide(robots, prey + hori) and collide(robots, prey + vert)):
            steps = np.append(steps, step, axis=0)
    return steps.reshape(-1, 2)

step = [0, 0]

def static(robots, prey, grid):
    if len(legal_steps(robots, prey, grid)) == 0:
        return None
    return prey
    
def random(robots, prey, grid):
    global step
    steps = legal_steps(robots, prey, grid)
    if len(steps) == 0:
        return None
    step = steps[np.random.randint(0, len(steps))]
    return prey + step

def smartLinear(robots, prey, grid):
    global step
    steps = legal_steps(robots, prey, grid)
    if len(steps) == 0:
        return None
    if np.all(step == [0, 0], axis=0):
        return random(robots, prey, grid)
    if not np.any(np.all(steps == step, axis=1)):
        angle0 = np.arctan2(step[1], step[0])
        angles = np.arctan2(steps[:, 1], steps[:, 0])
        diff = (angles - angle0 + np.pi) % (2 * np.pi) - np.pi
        step = steps[np.argmin(np.abs(diff))]
    return prey + step