import numpy as np
import Tools

def fwEuler(field, x0, y0, ds, L):
    """
    Integrates one field line of length L using
    Euler's forward method, starting at
    point (x0, y0), with steplength ds.
    """

    iterations = int(L / ds)

    x = np.full(iterations, None)
    y = np.full(iterations, None)

    dim = field[0][0].shape[0]

    mid = int(iterations/2.0)

    x[mid] = x0
    y[mid] = y0

    #integrate in forward direction
    for i in range(mid-1):

        xf = x[mid + i]
        yf = y[mid + i]

        #stop if point reaches edge
        if Tools.outside(xf, yf, dim):
            break

        fx = Tools.interpolate(field, xf, yf, 0)
        fy = Tools.interpolate(field, xf, yf, 1)

        norm = (fx * fx + fy * fy)**(0.5)

        #stop if field very small
        if norm < 1e-15:
            break

        fx /= norm
        fy /= norm

        x[mid + i + 1] = xf +  fx * ds
        y[mid + i + 1] = yf +  fy * ds


    #integrate in backward direction
    for j in range(mid-1):

        xb = x[mid - j]
        yb = y[mid - j]

        #stop if point reaches edge
        if Tools.outside(xb, yb, dim):
            break

        fx = Tools.interpolate(field, xb, yb, 0)
        fy = Tools.interpolate(field, xb, yb, 1)

        norm = (fx*fx + fy*fy)**(0.5)

        #stop if field very small
        if norm < 1e-15:
            break

        fx /= norm
        fy /= norm

        x[mid - (j+1)] = xb -  fx * ds
        y[mid - (j+1)] = yb -  fy * ds


    x = x[mid - j : mid + i]
    y = y[mid - j : mid + i]

    return np.array([x, y])

def RK4(field, x0, y0, ds, L):
    """
    Integrates one field line of length L using
    Runge-Kutta 4, starting at
    point (x0, y0), with steplength ds.
    """

    iterations = int(L / ds)

    x = np.full(iterations, None)
    y = np.full(iterations, None)

    dim = field[0][0].shape[0]

    mid = int(iterations/2.0)

    #set seed value in the middle of the line
    x[mid] = x0
    y[mid] = y0

    #integrate in forward direction
    for i in range(mid-1):

        xf = x[mid + i]
        yf = y[mid + i]

        #stop if point reaches edge
        if Tools.outside(xf, yf, dim):
            break

        k1x = Tools.interpolate(field, xf, yf, 0)
        k1y = Tools.interpolate(field, xf, yf, 1)

        xf += 0.5*ds*k1x
        yf += 0.5*ds*k1y

        #stop if point reaches edge
        if Tools.outside(xf, yf, dim):
            break

        k2x = Tools.interpolate(field, xf, yf, 0)
        k2y = Tools.interpolate(field, xf, yf, 1)

        xf += 0.5*ds*k2x
        yf += 0.5*ds*k2y

        #stop if point reaches edge
        if Tools.outside(xf, yf, dim):
            break

        k3x = Tools.interpolate(field, xf, yf, 0)
        k3y = Tools.interpolate(field, xf, yf, 1)

        xf += ds*k3x
        yf += ds*k3y

        #stop if point reaches edge
        if Tools.outside(xf, yf, dim):
            break

        k4x = Tools.interpolate(field, xf, yf, 0)
        k4y = Tools.interpolate(field, xf, yf, 1)

        fx = 1.0/6.0 * (k1x + 2*k2x + 2*k3x + k4x)
        fy = 1.0/6.0 * (k1y + 2*k2y + 2*k3y + k4y)

        norm = (fx * fx + fy * fy)**(0.5)

        #stop if field very small
        if norm < 1e-15:
            break

        fx /= norm
        fy /= norm

        x[mid + i + 1] = x[mid + i] + fx * ds
        y[mid + i + 1] = y[mid + i] + fy * ds

    #integrate in backward direction
    for j in range(mid-1):

        xb = x[mid - j]
        yb = y[mid - j]

        #stop if point reaches edge
        if Tools.outside(xb, yb, dim):
            break

        k1x = Tools.interpolate(field, xb, yb, 0)
        k1y = Tools.interpolate(field, xb, yb, 1)

        xb += 0.5*ds*k1x
        yb += 0.5*ds*k1y

        #stop if point reaches edge
        if Tools.outside(xb, yb, dim):
            break

        k2x = Tools.interpolate(field, xb, yb, 0)
        k2y = Tools.interpolate(field, xb, yb, 1)

        xb += 0.5*ds*k2x
        yb += 0.5*ds*k2y

        #stop if point reaches edge
        if Tools.outside(xb, yb, dim):
            break

        k3x = Tools.interpolate(field, xb, yb, 0)
        k3y = Tools.interpolate(field, xb, yb, 1)

        xb += ds*k3x
        yb += ds*k3y

        #stop if point reaches edge
        if Tools.outside(xb, yb, dim):
            break

        k4x = Tools.interpolate(field, xb, yb, 0)
        k4y = Tools.interpolate(field, xb, yb, 1)

        fx = 1.0/6.0 * (k1x + 2*k2x + 2*k3x + k4x)
        fy = 1.0/6.0 * (k1y + 2*k2y + 2*k3y + k4y)

        norm = (fx * fx + fy * fy)**(0.5)

        #stop if field very small
        if norm < 1e-15:
            break

        fx /= norm
        fy /= norm

        x[mid - (j + 1)] = x[mid - j] - fx * ds
        y[mid - (j + 1)] = y[mid - j] - fy * ds

    #slice array to remove None values
    x = x[mid - j : mid + i]
    y = y[mid - j : mid + i]

    return np.array([x,y])
