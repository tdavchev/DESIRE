'''
Helper functions to compute the masks relevant to social grid

Author : Anirudh Vemula
Date : 29th October 2016
'''
from math import *

import numpy as np


def get_grid_mask(frame, frame_ids, dmin=1, dmax=40):
    '''
    This function computes the binary mask that represents the
    occupancy of each ped in the other's grid
    params:
    frame : This will be a MNP x 3 matrix with each row being [pedID, x, y]
    grid_size : Scalar value representing the size of the grid discretization
    '''

    radial_bin_discr = np.logspace(
        start=log(dmin, 2),
        stop=log(dmax, 2),
        num=6,
        base=2,
        endpoint=True)
    rbd_tensor = [tf.constant(radial_bin_discr[i], dtype=tf.float32) for i in xrange(6)]
    # Maximum number of objects
    mno = len(frame)#frame.shape[0]
    ktraj = len(frame[1])#frame.shape[1]
    frame = [[tf.split(0, 2, frame[o][k][0]) for k in xrange(ktraj)] for o in xrange(mno)]
    cost = [[[tf.zeros(1) for i in xrange(36)] for i in xrange(ktraj)] for y in xrange(mno)]
    frame_mask = [
        [
            [
                [
                    [tf.zeros(1) for r in xrange(36)]
                    for y in xrange(ktraj)]
                for z in xrange(mno)]
            for y in xrange(ktraj)]
        for i in xrange(mno)]
    # For each ped in the frame (existent and non-existent)
    for pedindex in range(mno):
        for k_traj in range(ktraj):
            # Get x and y of the current ped
            current_x, current_y = frame[pedindex][k_traj][0], frame[pedindex][k_traj][1]
            # For all the other peds
            for otherpedindex in range(mno):
                # If the other pedID is the same as current pedID

                for k in range(ktraj):
                    angular_bin = tf.split(0, 6, tf.zeros([6]))
                    radial_bin = tf.split(0, 6, tf.zeros([6]))
                    # Get x and y of the other ped
                    other_x, other_y = \
                        frame[otherpedindex][k_traj][0], frame[otherpedindex][k_traj][1]

                    # Calculate the radius and angle for the log-polar coordinate
                    radius = tf.log(
                        tf.sqrt(
                            tf.add(
                                tf.square(
                                    tf.sub(other_x, current_x)),
                                tf.square(
                                    tf.sub(other_y, current_y)))))
                    theta = tf.atan(
                        tf.divide(
                            tf.sub(other_x, current_x),
                            tf.sub(other_y, current_y)))
                    angular_bin[0] = tf.select(
                        tf.logical_and(
                            tf.logical_and(
                                tf.greater(
                                    tf.sub(other_x, current_x),
                                    tf.constant(0.0)
                                ),
                                tf.greater(
                                    tf.sub(other_y, current_y),
                                    tf.constant(0.0)
                                )),
                            tf.less(
                                tf.divide(tf.abs(theta), 2.0), tf.constant(45.0)
                            )),
                        [tf.add(angular_bin[0], tf.constant(1.0))],
                        [angular_bin[0]])
                    angular_bin[1] = tf.select(
                        tf.logical_and(
                            tf.logical_and(
                                tf.greater(
                                    tf.sub(other_x, current_x),
                                    tf.constant(0.0)
                                ),
                                tf.greater(
                                    tf.sub(other_y, current_y),
                                    tf.constant(0.0)
                                )),
                            tf.greater_equal(
                                tf.divide(tf.abs(theta), 2.0), tf.constant(45.0)
                            )),
                        [tf.add(angular_bin[1], tf.constant(1.0))],
                        [angular_bin[1]])
                    angular_bin[2] = tf.select(
                        tf.logical_and(
                            tf.logical_and(
                                tf.less(
                                    tf.sub(other_x, current_x),
                                    tf.constant(0.0)
                                ),
                                tf.greater(
                                    tf.sub(other_y, current_y),
                                    tf.constant(0.0)
                                )),
                            tf.less(
                                tf.divide(tf.abs(theta), 2.0), tf.constant(45.0)
                            )),
                        [tf.add(angular_bin[2], tf.constant(1.0))],
                        [angular_bin[2]])
                    angular_bin[3] = tf.select(
                        tf.logical_and(
                            tf.logical_and(
                                tf.less(
                                    tf.sub(other_x, current_x),
                                    tf.constant(0.0)
                                ),
                                tf.greater(
                                    tf.sub(other_y, current_y),
                                    tf.constant(0.0)
                                )),
                            tf.greater_equal(
                                tf.divide(tf.abs(theta), 2.0), tf.constant(45.0)
                            )),
                        [tf.add(angular_bin[3], tf.constant(1.0))],
                        [angular_bin[3]])
                    angular_bin[4] = tf.select(
                        tf.logical_and(
                            tf.logical_and(
                                tf.less(
                                    tf.sub(other_x, current_x),
                                    tf.constant(0.0)
                                ),
                                tf.less(
                                    tf.sub(other_y, current_y),
                                    tf.constant(0.0)
                                )),
                            tf.less(
                                tf.divide(tf.abs(theta), 2.0), tf.constant(45.0)
                            )),
                        [tf.add(angular_bin[4], tf.constant(1.0))],
                        [angular_bin[4]])
                    angular_bin[5] = tf.select(
                        tf.logical_and(
                            tf.logical_and(
                                tf.greater(
                                    tf.sub(other_x, current_x),
                                    tf.constant(0.0)
                                ),
                                tf.less(
                                    tf.sub(other_y, current_y),
                                    tf.constant(0.0)
                                )),
                            tf.greater_equal(
                                tf.divide(tf.abs(theta), 2.0), tf.constant(45.0)
                            )),
                        [tf.add(angular_bin[5], tf.constant(1.0))],
                        [angular_bin[5]])
                    radius = tf.select(
                        tf.logical_and(
                            tf.greater(radius, tf.constant(dmax, dtype=tf.float32)),
                            tf.less(radius, tf.constant(dmin, dtype=tf.float32))),
                        [tf.constant(0.0)], [radius[0]])


                    # If in surrounding, calculate the grid cell
                    for i in xrange(1, len(radial_bin_discr)):
                        radial_bin[i] = \
                            tf.select(
                                tf.logical_and(
                                    tf.less_equal(radius, rbd_tensor[i]),
                                    tf.greater_equal(radius, rbd_tensor[i-1])),
                                [tf.add(radial_bin[i], tf.constant(1.0))],
                                [radial_bin[i]])

                    # Then I will know which otherpeds are in a given cell i.e. 23
                    # This won't work, radial bin and angular bin are converted to tensors ..
                    # frame_mask[pedindex][ktraj][otherpedindex] = [angular_bin[0], radial_bin[0]]
                    for rb in xrange(0, len(radial_bin_discr)):
                        for ab in xrange(0, 6):
                            # radial_bin + angular_bin*6
                            frame_mask[pedindex][k_traj][otherpedindex][k][rb + ab*6] = \
                                tf.select(
                                    tf.logical_and(
                                        tf.equal(radial_bin[rb], tf.constant(1.0)),
                                        tf.equal(angular_bin[ab], tf.constant(1.0))),
                                    [tf.add(
                                        frame_mask[pedindex][k_traj][otherpedindex][k][rb + ab*6],
                                        tf.constant(1.0))],
                                    [frame_mask[pedindex][k_traj][otherpedindex][k][rb + ab*6]])
                            cost[pedindex][k_traj][rb+ab*6] = \
                                tf.select(
                                    tf.logical_and(
                                        tf.equal(radial_bin[rb], tf.constant(1.0)),
                                        tf.equal(angular_bin[ab], tf.constant(1.0))),
                                    [tf.add(
                                        cost[pedindex][k_traj][rb+ab*6],
                                        tf.constant(1.0))],
                                    [cost[pedindex][k_traj][rb+ab*6]])
                            cost[pedindex][k_traj][rb+ab*6] = tf.squeeze(cost[pedindex][k_traj][rb+ab*6], [0])
                            frame_mask[pedindex][k_traj][otherpedindex][k][rb + ab*6] = \
                                tf.squeeze(
                                    frame_mask[pedindex][k_traj][otherpedindex][k][rb + ab*6],
                                    [0])

    return frame_mask, cost


# def get_grid_mask(frame, dmin=1, dmax=40):
#     '''
#     This function computes the binary mask that represents the
#     occupancy of each ped in the other's grid
#     params:
#     frame : This will be a MNP x 3 matrix with each row being [pedID, x, y]
#     grid_size : Scalar value representing the size of the grid discretization
#     '''

#     radial_bin_discr = np.logspace(
#         start=log(dmin, 2),
#         stop=log(dmax, 2),
#         num=6,
#         base=2,
#         endpoint=True)
#     # Maximum number of objects
#     mno = frame.shape[0]
#     frame_mask = np.zeros((mnp, mnp, 6, 6))

#     # For each ped in the frame (existent and non-existent)
#     for pedindex in range(mno):
#         # If pedID is zero, then non-existent ped
#         if frame[pedindex, 0] == 0:
#             # Binary mask should be zero for non-existent ped
#             continue

#         # Get x and y of the current ped
#         current_x, current_y = frame[pedindex, 1], frame[pedindex, 2]

#         # For all the other peds
#         for otherpedindex in range(mno):
#             # If other pedID is zero, then non-existent ped
#             if frame[otherpedindex, 0] == 0:
#                 # Binary mask should be zero
#                 continue

#             # If the other pedID is the same as current pedID
#             if frame[otherpedindex, 0] == frame[pedindex, 0]:
#                 # The ped cannot be counted in his own grid
#                 continue

#             # Get x and y of the other ped
#             other_x, other_y = frame[otherpedindex, 1], frame[otherpedindex, 2]
#             coordinate = 1

#             # Calculate the radius and angle for the log-polar coordinate
#             radius = log(np.sqrt(
#                 (other_x - current_x)^2 + (other_y - current_y)^2
#             ))
#             theta = np.arctan2(current_x - other_x, current_y - other_y)
#             if other_x - current_x < 0:
#                 if other_y - current_y < 0:
#                     # coordinate = 3
#                     theta += 180
#                 else:
#                     # coordinate = 2
#                     theta += 90
#             else:
#                 if other_y - current_x < 0:
#                     # coordinate = 4
#                     theta = np.abs(theta) + 270
#             if radius > dmax and radius < dmin:
#                 # Ped not in surrounding, so binary mask should be zero
#                 continue

#             # If in surrounding, calculate the grid cell
#             for i in xrange(radial_bin_discr):
#                 if radius <= radial_bin_discr[i]:
#                     radial_bin = i
#                     angular_bin = theta / (360/60)
#                     break

#             # Other ped is in the corresponding grid cell of current ped
#             frame_mask[pedindex, otherpedindex, radial_bin + angular_bin*6] = 1
#         frame_mask[pedindex] = \
#             frame_mask[pedindex]/np.sum(frame_mask[pedindex], axis=0, dtype=np.float32)
#     return frame_mask


def get_location_mask(frame, dimensions):
    '''
    This function computes the binary mask that represents the
    location of each agent
    params:
    frame : This will be a MNO x 3 matrix with each row being [objID, x, y]
    dimensions : This will be a list [width, height]
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
    Get the sequence locations mask
    params:
    sequence : A numpy matrix of shape SL x MNO x 3
    dimensions : This will be a list [width, height]
    '''
    _sl = sequence.shape[0]
    mno = sequence.shape[1]
    sequence_mask = np.zeros((_sl, mno, dimensions[0] / 2, dimensions[1] / 2))

    for i in range(_sl):
        sequence_mask[i, :, :, :] = \
            get_location_mask(sequence[i, :, :], dimensions)

    return sequence_mask

def get_sequence_grid_mask(sequence):
    '''
    Get the grid masks for all the frames in the sequence
    params:
    sequence : A numpy matrix of shape SL x MNO x 3
    '''
    _sl = sequence.shape[0]
    mno = sequence.shape[1]
    sequence_mask = np.zeros((_sl, mno, mno, 6, 6))

    for i in range(_sl):
        sequence_mask[i, :, :, :] = \
            get_grid_mask(sequence[i, :, :])

    return sequence_mask
