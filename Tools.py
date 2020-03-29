import math
import numpy as np

def interpolate(field, x, y, ax):
    """
    Interpolates field values at arbitrary position (x,y).
    ax = 0, 1 gives the x- and y-component respectively.
    """

    x1 = math.floor(x)
    y1 = math.floor(y)
    x2 = x1 + 1
    y2 = y1 + 1

    v1 = (x2 - x) * field[ax][x1][y1] + (x - x1) * field[ax][x2][y1]
    v2 = (x2 - x) * field[ax][x1][y2] + (x - x1) * field[ax][x2][y2]

    return (y2 - y) * v1 + (y - y1) * v2

def gaussian(x, mu, sigma):
    """
    Standard normal distribution
        mu = center
        sigma = std dev
        x = range
    """
    sigma2 = sigma * sigma
    c = 1.0/(2*math.pi*sigma2)**0.5
    gaussian_ = c * np.exp(-0.5*(x-mu)*(x-mu)/sigma2)
    return gaussian_

def outside(x, y, dim):
    """
    Returns True if point (x,y) is outside scope (0, dim-1).
    """

    outside_ = (x < 0) or (y < 0) or (x > (dim-1)) or (y > (dim-1))

    return outside_

def distance(p1, p2):
    """
    Takes either

        A) two points p1, p2 and returns
           distance between them

        B) one array p1 and a point p2 and returns
           smallest distance between point and array
    """

    if p1.shape == p2.shape:
        p = p1 - p2
        distance_ = (p[0]*p[0] + p[1]*p[1])**0.5

    else:
        p = p1.T - p2
        length = p.shape[0]
        distance_ = np.zeros(length)
        for i in range(length):
            distance_[i] = ( p[i,0]*p[i,0] + p[i,1]*p[i,1] )**0.5

        distance_ = np.min(distance_)

    return distance_
