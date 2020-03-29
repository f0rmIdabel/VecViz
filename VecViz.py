import h5py
import numpy as np
import math
import matplotlib.pyplot as plt
import Integrate
import Tools
import time

class VecViz:

    def __init__(self, dataset):
        self.dataset = dataset

    def read_field(self):
        """
        Reads the fields such that field.shape = (2, xdim, ydim),
            field_xdim = field[0]
            field_ydim = field[1]

        The different field configs found by trial and error.
        """

        f = h5py.File(self.dataset, 'r')

        field_x = np.asarray(f['Velocity']['X-comp'])
        field_y = np.asarray(f['Velocity']['Y-comp'])

        f.close()

        if self.dataset == 'Data/metsim1_2d.h5':

            field_x = field_x.T
            field_y = field_y.T

        if self.dataset == 'Data/isabel_2d.h5':

            field_x_ = np.rot90(field_y, 3)
            field_y = -np.rot90(field_x, 3)
            field_x = field_x_

        return np.array([field_x, field_y])

    def glyphs(self, n=1, show = False):
        """
        Plots 2D glyphs for every nth field point,
        using matplotlib's quiver function.
        """

        field = self.read_field()

        fx = field[0].T
        fy = field[1].T

        dim = fx.shape[0]

        x = np.arange(dim)
        y = np.arange(dim)

        x,y = np.meshgrid(x,y)

        fig, ax = plt.subplots()
        ax.quiver(x[::n,::n],y[::n,::n],fx[::n,::n],fy[::n,::n])

        if show:
            plt.ylim(0,dim)
            plt.xlim(0,dim)
            plt.show()

        return None

    def geometric_fieldlines_reg(self, ds, L, method, n = 1, show = False):
        """
        Plots field lines of length L starting from every
        nth point of the field, using
        """
        field = self.read_field()

        dim = field.shape[1]

        #pick every nth data point as seed point
        x_seeds = np.arange(dim)[::n]
        y_seeds = np.arange(dim)[::n]

        lines = []

        #go through every seed point
        for x_seed in x_seeds:
            for y_seed in y_seeds:

                #calculate line starting at seed point
                line = eval('Integrate.' + method + '(field, x_seed, y_seed, ds, L)')

                #go to next seed, if zero-length line
                length = line.shape[1]
                if length == 0:
                    continue

                #accept line of non-zero length
                lines.append(line)

        #plot every field line
        for line in lines:
            plt.plot(line[0], line[1], 'k-')

        if show:
            plt.show()

        return None

    def geometric_fieldlines_even(self, ds, L, method, line_separation, x0, y0, max_lines, show = False):
        """
        Plots field lines of length L,
        with minimum line separation.
        First line starts from (x0, y0).
        """

        field = self.read_field()
        dim = field.shape[1]

        #integrates an initial field line starting at (x0, y0) and add it to the queue
        line0 = eval('Integrate.' + method + '(field, x0, y0, ds, L)')
        queue = [line0]

        total_lines = 1
        queue_number = -1

        while total_lines < max_lines:

            queue_number += 1

            #stop if no more lines in queue_number
            if queue_number > (total_lines - 1):
                break

            #picks the next line in the queue
            current_line = queue[queue_number]
            current_line_length = current_line.shape[1]

            i = 0

            #goes through the line twice, first checks for points to the left, the to the right
            while i < ((current_line_length - 1)*2) and total_lines < max_lines:


                #finds point a distance line_separation perpendicular to the left of line
                if i < (current_line_length - 1):

                    current_line_element = current_line[:,i]
                    next_line_element = current_line[:,i+1]
                    tangent =  next_line_element - current_line_element
                    new_point = current_line_element + np.asarray([-tangent[1], tangent[0]]) * line_separation/ds*2.0

                #finds point a distance line_separation perpendicular to the right of line
                else:
                    j = i - (current_line_length - 1)
                    current_line_element = current_line[:,j]
                    next_line_element = current_line[:,j+1]

                    tangent =  next_line_element - current_line_element

                    new_point = current_line_element + np.asarray([tangent[1], -tangent[0]]) * line_separation/ds*2.0

                #if new_point outside range, go to the next point on current_line
                if Tools.outside(new_point[0], new_point[1], dim):
                   i += 1
                   continue

                #if new_point too close to existing line, go to the next point on current_line
                pick_new_point = False
                for line in range(total_lines):
                    if Tools.distance(queue[line], new_point) < line_separation:
                        pick_new_point = True
                        break

                if pick_new_point:
                    i += 1
                    continue

                #if new_point has passed all tests, calculate new_line
                new_line = eval('Integrate.' + method + '(field, new_point[0], new_point[1], ds, L)')
                new_line_length = new_line.shape[1]

                #exclude very short lines and jump ahead
                if new_line_length < 5:
                    i += 5
                    continue

                #check if any point on new_line is too close to existing lines and slice the line
                slice_line = False
                for j in range(new_line_length):
                    for line in range(total_lines):
                        if Tools.distance( queue[line], new_line[:,j] ) < line_separation:

                            new_line = new_line[:,:j]
                            new_line_length = new_line.shape[1]

                            slice_line = True
                            break

                    if slice_line:
                        break

                #if the resulting line is not empty, append it to the queue
                if new_line.shape[1] > 0:
                    queue.append(new_line)
                    total_lines += 1
                    print(total_lines)

                i+= 5

        #plot every line in the queue
        for line in queue:
            plt.plot(line[0], line[1], 'k-')

        if show:
            plt.show()

        return None

    def texture_fieldlines_LIC(self, ds, L, method, pix, show = False):

        field = self.read_field()

        dim = field.shape[1]
        ratio = dim/pix

        #create white noise input texture and initialize output texture
        input_texture = np.random.rand(pix, pix)
        output_texture = np.zeros((pix, pix))

        #go through every pixel
        for y in range(pix):
            for x in range(pix):

                #calculate field line starting at center of pixel
                line = eval('Integrate.' + method + '(field, (x + 0.5)*ratio , (y + 0.5)*ratio, ds, L)')
                line_length = line.shape[1]

                #if line
                if line_length == 0:
                    continue

                #find line center index
                mid = np.where(line[0] == (x + 0.5)*ratio)[0]

                #added to avoid strange bug
                if mid.shape[0] == 0:
                    continue

                #Calculate weights
                k = Tools.gaussian(np.arange(line_length), mid[0], L*0.5)
                k /= np.sum(k)

                for l in range(line_length):

                    #find pixel corresponding to position
                    x_pix = math.floor(line[0,l]/ratio)
                    y_pix = math.floor(line[1,l]/ratio)

                    #add weighted input pixel value to output pixel
                    output_texture[y, x] += input_texture[y_pix, x_pix] *k[l]


        plt.imshow(np.rot90(output_texture.T,1), cmap='Greys')
        if show:
            plt.show()

        return None


if __name__ == '__main__':

    metsim = 'Data/metsim1_2d.h5'
    isabel = 'Data/isabel_2d.h5'

    f = 0

    while f not in ['1', '2']:

        f = input("\nChoose between datasets:\n\n    1. metsim\n    2. isabel\n\nChoice: ")

        if f == '1':
            field = VecViz2D(metsim)

        elif f == '2':
            field = VecViz2D(isabel)

        else:
            print("\nNo dataset chosen. Try again.")

    technique = 0

    while technique not in ['1', '2', '3']:

        technique = input("\nChoose technique:\n\n    1. Geometric: regular seeding\n    2. Geometric: evenly spaced lines\n    3. Texture: LIC\n\nChoice: ")

        if technique == '1':

            integration = 0

            while integration not in ['1', '2']:

                integration = input("\nChoose technique:\n\n    1. Forward Euler \n    2. Runge-Kutta 4\n\nChoice: ")

                if integration == '1':
                    int_technique = 'fwEuler'

                elif integration == '2':
                    int_technique = 'RK4'
                else:
                    print("\nNo integration technique chosen. Try again.")

            L  = float(input("Choose field line length, L = "))
            ds = float(input("Choose integration step size, ds = "))
            n = int(input("Sample step size, n = "))

            start = time.time()

            field.geometric_fieldlines_reg(ds, L, int_technique, n, False)
            total_time = time.time() - start
            print("Time usage: %f" %(total_time))
            plt.show()

        elif technique == '2':

            integration = 0

            while integration not in ['1', '2']:

                integration = input("\nChoose technique:\n\n    1. Forward Euler \n    2. Runge-Kutta 4\n\nChoice: ")

                if integration == '1':
                    int_technique = 'fwEuler'

                elif integration == '2':
                    int_technique = 'RK4'
                else:
                    print("\nNo integration technique chosen. Try again.")

            L  = float(input("Choose field line length, L = "))
            ds = float(input("Choose integration step size, ds = "))
            d_sep = float(input("Choose line separation, d_sep = "))
            max_lines = int(input("Choose maximum number of lines, max_lines = "))

            x0 = float(input("Choose starting point.\n    x0 = "))
            y0 = float(input("    y0 = "))

            time_start = time.time()
            field.geometric_fieldlines_even(ds, L, int_technique, d_sep, x0, y0, max_lines, False)
            total_time = time.time() - time_start
            print("Time usage: %f" %(total_time))
            plt.show()

        elif technique == '3':

            integration = 0

            while integration not in ['1', '2']:

                integration = input("\nChoose technique:\n\n    1. Forward Euler \n    2. Runge-Kutta 4\n\nChoice: ")

                if integration == '1':
                    int_technique = 'fwEuler'

                elif integration == '2':
                    int_technique = 'RK4'
                else:
                    print("\nNo integration technique chosen. Try again.")

            L  = float(input("Choose field line length, L = "))
            ds = float(input("Choose integration step size, ds = "))
            pix = int(input("Choose resolution of input texture, pix = "))

            time_start = time.time()
            field.texture_fieldlines_LIC(ds, L, int_technique, pix, False)
            total_time = time.time() - time_start
            print("Time usage: %f" %(total_time))
            plt.show()
        else:
            print("\nNo technique chosen. Try again.")
