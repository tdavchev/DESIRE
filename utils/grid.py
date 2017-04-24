'''
Helper functions to compute the masks relevant to social grid

Author : Anirudh Vemula
Date : 29th October 2016
'''
from math import *

import numpy as np

def get_grid_mask(frame, dimensions, neighborhood_size, grid_size):
    '''
    This function computes the binary mask that represents the
    occupancy of each ped in the other's grid
    params:
    frame : This will be a MNP x 3 matrix with each row being [pedID, x, y]
    dimensions : This will be a list [width, height]
    neighborhood_size : Scalar value representing the size of neighborhood considered
    grid_size : Scalar value representing the size of the grid discretization
    '''

    dmin = 1
    dmax = 40
    radial_bin_discr = np.logspace(
        start=log(dmin, 2),
        stop=log(dmax, 2),
        num=6,
        base=2,
        endpoint=True)
    # Maximum number of objects
    mno = frame.shape[0]
    frame_mask = np.zeros((mnp, mnp, 6, 6))

    # For each ped in the frame (existent and non-existent)
    for pedindex in range(mno):
        # If pedID is zero, then non-existent ped
        if frame[pedindex, 0] == 0:
            # Binary mask should be zero for non-existent ped
            continue

        # Get x and y of the current ped
        current_x, current_y = frame[pedindex, 1], frame[pedindex, 2]

        # For all the other peds
        for otherpedindex in range(mno):
            # If other pedID is zero, then non-existent ped
            if frame[otherpedindex, 0] == 0:
                # Binary mask should be zero
                continue

            # If the other pedID is the same as current pedID
            if frame[otherpedindex, 0] == frame[pedindex, 0]:
                # The ped cannot be counted in his own grid
                continue

            # Get x and y of the other ped
            other_x, other_y = frame[otherpedindex, 1], frame[otherpedindex, 2]
            coordinate = 1

            # Calculate the radius and angle for the log-polar coordinate
            radius = log(np.sqrt(
                (other_x - current_x)^2 + (other_y - current_y)^2
            ))
            theta = np.arctan2(current_x - other_x, current_y - other_y)
            if other_x - current_x < 0:
                if other_y - current_y < 0:
                    # coordinate = 3
                    theta += 180
                else:
                    # coordinate = 2
                    theta += 90
            else:
                if other_y - current_x < 0:
                    # coordinate = 4
                    theta = np.abs(theta) + 270
            if radius > dmax and radius < dmin:
                # Ped not in surrounding, so binary mask should be zero
                continue

            # If in surrounding, calculate the grid cell
            for i in xrange(radial_bin_discr):
                if radius <= radial_bin_discr[i]:
                    radial_bin = i
                    angular_bin = theta / (360/60)
                    break

            # Other ped is in the corresponding grid cell of current ped
            frame_mask[pedindex, otherpedindex, radial_bin + angular_bin*6] = 1
        frame_mask[pedindex] = \
            frame_mask[pedindex]/np.sum(frame_mask[pedindex], axis=0, dtype=np.float32)
    return frame_mask


def get_location_mask(frame, dimensions):
    '''
    This function computes the binary mask that represents the
    occupancy of each ped in the other's grid
    params:
    frame : This will be a MNP x 3 matrix with each row being [pedID, x, y]
    dimensions : This will be a list [width, height]
    neighborhood_size : Scalar value representing the size of neighborhood considered
    grid_size : Scalar value representing the size of the grid discretization
    '''
    mno = frame.shape[0]
    width, height = dimensions[0], dimensions[1]

    frame_mask = np.zeros((mno, dimensions[0] / 2, dimensions[1] / 2))

    # For each ped in the frame (existent and non-existent)
    for pedindex in range(mno):
        # If pedID is zero, then non-existent ped
        if frame[pedindex, 0] == 0:
            # Binary mask should be zero for non-existent ped
            continue

        # Get x and y of the current ped
        # TODO:check if these are x and y exactly !
        current_x, current_y = frame[pedindex, 1], frame[pedindex, 2]

        # Other ped is in the corresponding grid cell of current ped
        frame_mask[pedindex, current_x / 2, current_y / 2] = [[1], [1]]

    return frame_mask


def get_sequence_location_mask(sequence, dimensions):
    '''
    Get the position masks for all the frames in the sequence
    params:
    sequence : A numpy matrix of shape SL x MNP x 3
    dimensions : This will be a list [width, height]
    neighborhood_size : Scalar value representing the size of neighborhood considered
    grid_size : Scalar value representing the size of the grid discretization
    '''
    _sl = sequence.shape[0]
    mno = sequence.shape[1]
    sequence_mask = np.zeros((_sl, mno, dimensions[0] / 2, dimensions[1] / 2))

    for i in range(_sl):
        sequence_mask[i, :, :, :] = \
            get_location_mask(sequence[i, :, :], dimensions)

    return sequence_mask


def get_sequence_grid_mask(sequence, dimensions, neighborhood_size, grid_size):
    '''
    Get the grid masks for all the frames in the sequence
    params:
    sequence : A numpy matrix of shape SL x MNO x 3
    dimensions : This will be a list [width, height]
    neighborhood_size : Scalar value representing the size of neighborhood considered
    grid_size : Scalar value representing the size of the grid discretization
    '''
    _sl = sequence.shape[0]
    mno = sequence.shape[1]
    sequence_mask = np.zeros((_sl, mno, mno, 6, 6))

    for i in range(_sl):
        sequence_mask[i, :, :, :] = \
            get_grid_mask(sequence[i, :, :], dimensions,
                          neighborhood_size, grid_size)

    return sequence_mask
