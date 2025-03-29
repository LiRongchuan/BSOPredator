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
    robots[idx] = pos
    return np.sum(np.linalg.norm(robots - prey, axis=1)) / len(robots)

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