import numpy as np
from scipy.spatial import ConvexHull
from matplotlib.path import Path

def fitness_repel(robots, idx, pos, Dmin=1):
    distances = np.linalg.norm(robots - pos, axis=1)
    distances[idx] = np.inf
    nnd = np.min(distances)
    return np.exp(2 * (Dmin - nnd)) if nnd < Dmin else 1

def is_point_on_segment(point, p1, p2):
    v1 = point - p1
    v2 = point - p2
    if np.cross(np.append(v1, [0]), np.append(v2, [0]))[-1] == 0:
        if v1[0] * v2[0] <= 0 and v1[1] * v2[1] <= 0:
            return True
    return False
   
def fitness_closure(robots, idx, pos, prey):
    robots[idx] = pos
    try:
        hull = ConvexHull(robots)
    except Exception:
        return 1
    hull_vertices = robots[hull.vertices]
    hull_path = Path(hull_vertices)
    if hull_path.contains_point(prey):
        return 0
    for i in range(len(hull_vertices)):
        p1 = hull_vertices[i]
        p2 = hull_vertices[(i+1) % len(hull_vertices)]
        if is_point_on_segment(prey, p1, p2):
            return 0.5
    return 1

def fitness_expanse(robots, idx, pos, prey):
    # robots[idx] = pos
    # return np.sum(np.linalg.norm(robots - prey, axis=1)) / len(robots)
    return np.linalg.norm(pos - prey)

def fitness_uniformity(robots, idx, pos, prey):
    robots[idx] = pos
    mask11 = (robots[:, 0] < prey[0]) & (robots[:, 1] > prey[1])
    mask12 = (robots[:, 0] == prey[0]) & (robots[:, 1] > prey[1])
    mask13 = (robots[:, 0] > prey[0]) & (robots[:, 1] > prey[1])
    mask21 = (robots[:, 0] < prey[0]) & (robots[:, 1] == prey[1])
    mask22 = (robots[:, 0] == prey[0]) & (robots[:, 1] == prey[1])
    mask23 = (robots[:, 0] > prey[0]) & (robots[:, 1] == prey[1])
    mask31 = (robots[:, 0] < prey[0]) & (robots[:, 1] < prey[1])
    mask32 = (robots[:, 0] == prey[0]) & (robots[:, 1] < prey[1])
    mask33 = (robots[:, 0] > prey[0]) & (robots[:, 1] < prey[1])
    std = np.std(np.array([sum(mask11), sum(mask12), sum(mask13), sum(mask21), sum(mask22), sum(mask23), sum(mask31), sum(mask32), sum(mask33)]))
    return std

def fitness(robots, idx, pos, prey, Dmin=1):
    f_repel = fitness_repel(robots.copy(), idx, pos, Dmin)
    f_closure = fitness_closure(robots.copy(), idx, pos, prey)
    f_expanse = fitness_expanse(robots.copy(), idx, pos, prey)
    f_uniformity = fitness_uniformity(robots.copy(), idx, pos, prey)
    return f_repel * (f_closure + f_expanse + f_uniformity) / (np.linalg.norm(pos - robots[idx]) + 1)

def greedy_fitness(robots, pos, preys, grid):
    """
    Example fitness function
    
    Parameters:
    p_robots (np.ndarray): Positions of real robots, shape (Ns, 2). Each represents a real robots.
    pos (np.ndarray): Position to evaluate, shape (2,)
    preys (np.ndarray): Position of preys, shape (n_prey, 2).
    grid (int): Size of the grid.
    
    This function should not modify p_robots or preys.  
    """
    Ns = robots.shape[0]
    n_prey = preys.shape[0]

    # Step 1: Bind each p_robot to the nearest prey
    bindings = [[] for _ in range(n_prey)]  # bindings[i] stores distances of p_robots bound to prey i

    for robot in robots:
        dist = np.linalg.norm(preys - robot, axis=1)
        target = np.argmin(dist)
        bindings[target].append(dist[target])

    # Step 2: Compute pos distance to each prey, and rank among binding distances
    fitness_values = []

    for i in range(n_prey):
        prey = preys[i]
        dist = np.linalg.norm(pos - prey)

        if bindings[i]:
            # Rank the pos distance among binding distances
            sorted_dists = sorted(bindings[i] + [dist])
            rank = sorted_dists.index(dist)  # 0-based ranking
        else:
            # No robot bound to this prey, assign highest penalty
            rank = 0
        coef = 1 - rank / (4 * np.exp(rank / 4))
        fitness = dist * coef
        fitness_values.append(fitness)

    return min(fitness_values)