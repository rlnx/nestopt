import numpy as np

def compute_2d(problem, x: np.ndarray, y: np.ndarray):
    assert problem.dimension == 2
    def compute_scalar(x, y):
        return problem.compute([x, y])
    compute_vec = np.vectorize(compute_scalar)
    return compute_vec(x, y)

def contour_2d(problem, n_points=100):
    assert problem.dimension == 2
    bbox = problem.domain.box
    xx = np.linspace(bbox.a[0], bbox.b[0], n_points)
    yy = np.linspace(bbox.a[1], bbox.b[1], n_points)
    X, Y = np.meshgrid(xx, yy)
    Z = compute_2d(problem, X, Y)
    return xx, yy, Z

def intersect_sphere_with_axis(center, radius, x, axis):
    x = np.asanyarray(x)
    center = np.asanyarray(center)
    assert len(center) == len(x)
    squares = (center - x) ** 2
    d = squares[axis] - np.sum(squares) + radius ** 2
    if d < 0: return None
    d = np.sqrt(d)
    y_1 = center[axis] - d
    y_2 = center[axis] + d
    return (y_1, y_2)

def unite_intervals(intervals, epsilon=1e-3):
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    new_intervals = [ sorted_intervals[0] ]
    for i in range(1, len(intervals)):
        p_beg, p_end = new_intervals[-1]
        c_beg, c_end = sorted_intervals[i]
        if c_beg - p_end <= epsilon:
            new = (p_beg, max(p_end, c_end))
            new_intervals[-1] = new
        else:
            new_intervals.append((c_beg, c_end))
    return new_intervals
