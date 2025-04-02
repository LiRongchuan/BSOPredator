import numpy as np
from optim.BSO.fitness import fitness

history = None # History of prey position

def bso(p_robots, prey, grid, fit=fitness):
    global history
    """
    BSO algorithm for optimization.
    
    Parameters:
    p_robots (np.ndarray): Positions of the robots, shape (Ns, Np, 2). Fisrst column represents the real robots.
    prey (np.ndarray): Position of the prey, shape (2,).
    grid (int): Size of the grid.
    fit (function): Fitness function to be optimized.
    
    This function modifies the p_robots.    
    """
    Ns, Np, _ = p_robots.shape # Robot number, virtual robot number, dimension
    f_robots = np.zeros((Ns, Np)) # Fitness of each robot
    p_group = np.zeros((Ns, 2), dtype=np.int32) # Best position of each group
    f_group = np.full(Ns, np.inf) # Best fitness of each group
    SN = np.array([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]])
    bound_angle = np.arctan2(SN[:, 1], SN[:, 0])
    
    # Hyperparameters
    vicinity = 5 # Neighborhood size
    singal_prob = 0.5
    keep_prob = 0.6
    
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
        # Evaluate each position
        for j in range(Np):
            f_robots[i, j] = fit(p_robots[:, 0], i, p_robots[i, j], prey, grid)
            if f_robots[i, j] < f_group[i]:
                f_group[i] = f_robots[i, j]
                p_group[i] = p_robots[i, j]
        
        # Mutate virtual robots
        for j in range(1, Np):
            if np.random.rand() < singal_prob:
                if np.random.rand() < keep_prob:
                    p_ij = p_robots[i, 0].copy()
                else:
                    p_ij = p_robots[np.random.randint(0, Ns), 0].copy()
            else:
                parent1 = p_robots[np.random.randint(0, Ns), 0]
                parent2 = p_robots[np.random.randint(0, Ns), 0]
                split = np.random.rand()
                p_ij = split * parent1 + (1 - split) * parent2
                p_ij = p_ij.astype(np.int32)
            p_ij += SN[np.random.randint(0, len(SN))] # Random noise
            if np.linalg.norm(p_ij - p_robots[i, 0]) > vicinity:
                real_angle = np.arctan2(p_ij[1] - p_robots[i, 0, 1], p_ij[0] - p_robots[i, 0, 0])
                diff = (bound_angle - real_angle + np.pi) % (2 * np.pi) - np.pi
                p_ij = p_robots[i, 0] + vicinity * SN[np.argmin(np.abs(diff))]
            f_ij = fit(p_robots[:, 0], i, p_ij, prey, grid)
            if f_ij <= f_robots[i, j]:
                f_robots[i, j] = f_ij
                p_robots[i, j] = p_ij

        # Update real robots
        if history is not None and np.linalg.norm(p_robots[i, 0] - history) <= 1.5:
            step = prey - history
            if np.any(np.all(legal_steps(p_robots[i, 0]) == step, axis=1)):
                p_robots[i, 0] += step
        else:
            steps = legal_steps(p_robots[i, 0])
            step = steps[np.argmin(np.linalg.norm(p_robots[i, 0] + steps - prey, axis=1))]
            p_robots[i, 0] += step
        # else:
        #     target = prey if np.linalg.norm(p_robots[i, 0] - prey) >= 10 else p_group[i]
        #     v = target - p_robots[i, 0]
        #     steps = legal_steps(p_robots[i, 0])
        #     dot_products = np.dot(steps, v) / (np.linalg.norm(steps, axis=1) + 1e-6)
        #     index = np.argmax(dot_products)
        #     step = steps[index]
        #     p_robots[i, 0] += step
            
    history = prey