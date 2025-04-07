import numpy as np
history = None # History of prey position

def naive(p_robots, prey, grid):
    global history
    """
    Naive algorithm for optimization.
    
    Parameters:
    p_robots (np.ndarray): Positions of the robots, shape (Ns, Np, 2). Fisrst column represents the real robots.
    prey (np.ndarray): Position of the prey, shape (2,).
    grid (int): Size of the grid.
    fit (function): Fitness function to be optimized.
    
    This function modifies the p_robots.    
    """
    Ns, Np, _ = p_robots.shape # Robot number, virtual robot number, dimension
    SN = np.array([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]])
    
    # Helper functions
    def collide(pos):
        for robot in p_robots[:, 0]:
            if np.all(pos == robot):
                return True
        return np.all(pos == prey)
    
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
        step = steps[np.argmin(np.linalg.norm(p_robots[i, 0] + steps - prey, axis=1))]
        p_robots[i, 0] += step
            
    history = prey