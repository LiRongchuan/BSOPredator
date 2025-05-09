import numpy as np

from scipy.spatial import distance

def naive(p_robots, preys, grid):
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



def clustered_naive(p_robots, preys, grid):
    Ns = p_robots.shape[0]
    N_prey = preys.shape[0]
    real_robots = p_robots[:, 0]
    
    dist_matrix = distance.cdist(real_robots, preys, 'euclidean')
    assignments = np.zeros(Ns, dtype=int)
    prey_status = np.zeros(N_prey, dtype=bool)

    if Ns < 2 * N_prey:
        robots_per_prey = 2
        groups = []
        remaining_robots = list(range(Ns))
        
        while len(remaining_robots) >= 2:
            group = remaining_robots[:2]
            groups.append(group)
            remaining_robots = remaining_robots[2:]
        if remaining_robots:
            groups[-1].append(remaining_robots[0])

        # 为每组分配最近的未捕获猎物
        for group_idx, group in enumerate(groups):
            if np.all(prey_status):
                break
                
            valid_preys = np.where(~prey_status)[0]
            group_distances = dist_matrix[group][:, valid_preys]
            closest_prey = valid_preys[np.argmin(group_distances.min(axis=0))]
            
            for robot_idx in group:
                assignments[robot_idx] = closest_prey
            
            # 检查猎物是否被捕获（组内所有捕食者到达）
            group_positions = real_robots[group]
            has_captured = all(np.array_equal(pos, preys[closest_prey]) for pos in group_positions)
            if has_captured:
                prey_status[closest_prey] = True
                for robot_idx in group:
                    assignments[robot_idx] = -1
            # group_positions = real_robots[group]
            # if np.all(group_positions == preys[closest_prey], axis=1):
            #     prey_status[closest_prey] = True  # 标记为已捕获
            #     # 释放该组成员重新分配
            #     for robot_idx in group:
            #         assignments[robot_idx] = -1 

    elif Ns >= 4 * N_prey:
        robots_per_prey = 4
    
    elif Ns >= 3 * N_prey:
        robots_per_prey = 3
    
    elif Ns >= 2 * N_prey:
        robots_per_prey = 2
    
    else:
        naive(p_robots, preys, grid)
        return p_robots

    assigned_robots = set()
    for prey_idx in range(N_prey):
        sorted_robots = np.argsort(dist_matrix[:, prey_idx])
        
        assigned = 0
        for robot_idx in sorted_robots:
            if robot_idx not in assigned_robots:
                assignments[robot_idx] = prey_idx
                assigned_robots.add(robot_idx)
                assigned += 1
                if assigned == robots_per_prey:
                    break
    
    for robot_idx in range(Ns):
        if robot_idx not in assigned_robots:
            closest_prey = np.argmin(dist_matrix[robot_idx])
            assignments[robot_idx] = closest_prey
    
    SN = np.array([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]])
    
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
    
    # for i in range(Ns):
    #     steps = legal_steps(p_robots[i, 0])
    #     targets = p_robots[i, 0] + steps
    #     # Only consider the assigned prey for this robot
    #     distances = np.linalg.norm(targets - preys[assignments[i]], axis=1)
    #     step = steps[np.argmin(distances)]
    #     p_robots[i, 0] += step

    for i in range(Ns):
        current_prey = assignments[i]
        if current_prey == -1:  # 情况4中的待分配捕食者
            # 寻找最近未捕获猎物
            valid_preys = np.where(~prey_status)[0]
            if len(valid_preys) > 0:
                closest = valid_preys[np.argmin(dist_matrix[i][valid_preys])]
                assignments[i] = closest
        
        steps = legal_steps(p_robots[i, 0])
        targets = p_robots[i, 0] + steps
        if assignments[i] >= 0:  # 有明确目标
            distances = np.linalg.norm(targets - preys[assignments[i]], axis=1)
            step = steps[np.argmin(distances)]
            p_robots[i, 0] += step
    
    return p_robots