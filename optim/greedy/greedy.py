import numpy as np

def greedy(p_robots, preys, grid):
    """
    Naive algorithm for optimization.
    
    Parameters:
    p_robots (np.ndarray): Positions of the robots, shape (Ns, Np, 2). Fisrst column represents the real robots.
    prey (np.ndarray): Position of the prey, shape (2,).
    grid (int): Size of the grid.
    fit (function): Fitness function to be optimized.
    
    This function modifies the p_robots.    
    """
    Ns, _, _ = p_robots.shape # Robot number, virtual robot number, dimension
    SN = np.array([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]])
    
    # Helper functions
    def collide(pos):
        return np.any(np.all(pos == p_robots[:, 0], axis=1)) or np.any(np.all(pos == preys, axis=1))
    
    def legal_steps(pos):
        steps = np.array([[0, 0]], dtype=np.int32)
        for step in SN[0::2]:
            target = pos + step
            if target[0] >= 0 and target[0] <= grid and\
               target[1] >= 0 and target[1] <= grid and\
               not collide(target):
                steps = np.append(steps, [step], axis=0)
        for step in SN[1::2]:
            target = pos + step
            hori = np.array([step[0], 0])
            vert = np.array([0, step[1]])
            if target[0] >= 0 and target[0] <= grid and\
               target[1] >= 0 and target[1] <= grid and\
               (not collide(target)) and not (collide(pos + hori) and collide(pos + vert)):
                steps = np.append(steps, [step], axis=0)
        return steps
    
    for i in range(Ns):
        steps = legal_steps(p_robots[i, 0])
        targets = p_robots[i, 0] + steps
        distances = np.linalg.norm(targets[:, None, :] - preys[None, :, :], axis=2)
        step = steps[distances.min(axis=1).argmin()]
        p_robots[i, 0] += step