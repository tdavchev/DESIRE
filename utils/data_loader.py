'''
Utils script for the DESIRE implementation
Handles processing the input and target data in batches and sequences

Author : Todor Davchev
Date : 13th February 2017
'''

import os
import pickle
import random
import sys
import numpy as np
import struct
import imghdr
from scipy import misc

sys.path.append("/home/s1045064/deep-learning/DESIRE")

# The data loader class that loads data from the datasets considering
# each frame as a datapoint and a sequence of consecutive frames as the
# sequence.
class DataLoader(object):
    """All frobnication, all the time."""
    # Eight is reasonable in this case.

    def __init__(self,
                 batch_size=50, seq_length=5, max_num_obj=40, leave_dataset=1, preprocess=False):
        '''
        Initialiser function for the DataLoader class
        params:
        batch_size : Size of the mini-batch
        grid_size : Size of the social grid constructed
        forcePreProcess : Flag to forcefully preprocess the data again from csv files
        '''
        # Number of datasets
        self.leave_dataset = leave_dataset

        # Data directory where the pre-processed pickle file resides
        self.data_dir = 'data/'

        # Set the frame pointer to zero for the current dataset
        self.frame_pointer = 0
        self.dataset_pointer = 0

        # Maximum number of obj in a single frame (Number obtained by checking the datasets)
        self.max_num_obj = max_num_obj

        # Store the arguments
        self.batch_size = batch_size
        self.seq_length = seq_length

        # Define the path in which the process data would be stored
        data_file = os.path.join(self.data_dir, "trajectories.cpkl")

        # If the file doesn't exist or forcePreProcess is true
        if not os.path.exists(data_file) or preprocess:
            print "Creating pre-processed data from raw data"
            sys.stdout.flush()
            # Preprocess the data from the csv files of the datasets
            # Note that this data is processed in frames
            self.frame_preprocess(data_file)

        # Load the processed data from the pickle file
        self.load_preprocessed(data_file)
        # Reset all the data pointers of the dataloader object
        self.reset_batch_pointer()

    def frame_preprocess(self, data_file):
        '''
        Function that will pre-process the annotations_processed.csv files of each dataset
        into data with occupancy grid that can be used
        params:
        data_file : The file into which all the pre-processed data needs to be stored
        '''

        # all_frame_data would be a list of numpy arrays corresponding to each dataset
        # Each numpy array would be of size (numFrames, max_num_obj, 3) where each objestrian's
        # obj_id, x, y , in each frame is stored
        all_frame_data = []
        # frame_list_data would be a list of lists corresponding to each dataset
        # Each list would contain the frameIds of all the frames in the dataset
        frame_list_data = []
        # numobj_data would be a list of lists corresponding to each dataset
        # Each list would contain the number of objestrians in each frame in the dataset
        num_obj_data = []
        reference_filenames = []
        # Index of the current dataset
        dataset_index = -1
        tmp_dir = ""
        # For each dataset
        for subdir, dirs, files in os.walk(self.data_dir):
            for file in files:
                # if dataset_index != leave_dataset and \
                if dataset_index < self.leave_dataset and \
                    file == 'reference.jpg':
                    # filenames = ['im_01.jpg', 'im_02.jpg', 'im_03.jpg', 'im_04.jpg']
                    reference_filenames.append([])
                    # step 2
                    file_path = os.path.join(subdir, file)
                    width, height = self.get_image_size(file_path)
                    img = misc.imread(file_path,  mode="CMYK")
                    # img = self.rgb_to_cmyk(img[:, :, 0], img[:, :, 1], img[:, :, 2])
                    # filename_queue = tf.train.string_input_producer(file_path)

                    # # read, decode and resize images
                    # reader = tf.WholeFileReader()
                    # filename, content = reader.read(filename_queue)
                    # image = tf.image.decode_jpeg(content, channels=4)
                    # image = tf.cast(image, tf.float32)
                    # resized_image = tf.image.resize_images(image, [height, width])
                    reference_filenames[dataset_index] = [img, height, width]

                if dataset_index < self.leave_dataset and \
                    file == 'annotations_processed.csv':

                    # Define path of the csv file of the current dataset
                    file_path = os.path.join(subdir, file)

                    # Load the data from the csv file
                    data = np.genfromtxt(file_path, delimiter=',')

                    # Frame IDs of the frames in the current dataset
                    frame_list = np.unique(data[0, :]).tolist()

                    # Number of frames
                    num_frames = len(frame_list)

                    # Add the list of frame IDs to the frame_list_data
                    frame_list_data.append(frame_list)

                    # Initialize the list of num_obj for the current dataset
                    num_obj_data.append([])

                    # Initialize the numpy array for the current dataset
                    all_frame_data.append((np.zeros((num_frames, self.max_num_obj, 3))))

                    # index to maintain the current frame
                    curr_frame = 0
                    for frame in frame_list:
                        # Extract all objestrians in current frame
                        obj_in_frame = data[:, data[0, :] == frame]

                        # Extract obj list
                        obj_list = obj_in_frame[1, :].tolist()

                        # Add number of obj in the current frame to the stored data
                        num_obj_data[dataset_index].append(len(obj_list))

                        # Initialize the row of the numpy array
                        obj_with_pos = []

                        # For each obj in the current frame
                        for obj in obj_list:
                            # Extract their x and y positions
                            current_x = obj_in_frame[2, obj_in_frame[1, :] == obj][0]
                            current_y = obj_in_frame[3, obj_in_frame[1, :] == obj][0]

                            # Add their obj_id, x, y to the row of the numpy array
                            obj_with_pos.append([obj, current_x, current_y])

                        # Add the details of all the obj in the current frame to all_frame_data
                        all_frame_data[dataset_index][curr_frame, 0:len(obj_list), :] = \
                            np.array(obj_with_pos)

                        # Increment the frame index
                        curr_frame += 1
            if subdir != "data/" and subdir != tmp_dir:
                # Increment the dataset index
                dataset_index += 1
                tmp_dir = subdir

        # Save the tuple (all_frame_data, frame_list_data, num_obj_data) in the pickle file
        file = open(data_file, "wb")
        pickle.dump((all_frame_data, frame_list_data, num_obj_data, reference_filenames),
                    file, protocol=2)
        file.close()

    def load_preprocessed(self, data_file):
        '''
        Function to load the pre-processed data into the DataLoader object
        params:
        data_file : the path to the pickled data file
        '''
        # Load data from the pickled file
        file = open(data_file, 'rb')
        self.raw_data = pickle.load(file)
        file.close()

        # Get all the data from the pickle file
        self.data = self.raw_data[0]
        self.frame_list = self.raw_data[1]
        self.num_obj_list = self.raw_data[2]
        self.reference_filenames = self.raw_data[3]
        counter = 0

        # For each dataset
        for dataset, _data in enumerate(self.data):
            # get the frame data for the current dataset
            all_frame_data = self.data[dataset]
            print len(all_frame_data)
            sys.stdout.flush()
            # Increment the counter with the number of sequences in the current dataset
            counter += int(len(all_frame_data) / (self.seq_length+2))

        # Calculate the number of batches
        self.num_batches = int(counter/self.batch_size)
        # On an average, we need twice the number of batches to cover the data
        # due to randomization introduced
        self.num_batches = self.num_batches * 2

    def next_batch(self, random_update=True):
        '''
        Function to get the next batch of points
        '''
        semantic_context_data = []
        # Source data
        x_batch = []
        # Target data
        y_batch = []
        # Dataset data
        dval = []
        # Iteration index
        i = 0
        while i < self.batch_size:
            # Extract the frame data of the current dataset
            current_data = self.data[self.dataset_pointer]
            current_ref_filenames = self.reference_filenames[self.dataset_pointer]
            # Get the frame pointer for the current dataset
            idx = self.frame_pointer
            # While there is still seq_length number of frames left in the current dataset
            if idx + self.seq_length < current_data.shape[0]:
                # All the data in this sequence
                seq_frame_data = current_data[idx:idx+self.seq_length+1, :]
                seq_source_frame_data = current_data[idx:idx+self.seq_length, :]
                seq_target_frame_data = current_data[idx+1:idx+self.seq_length+1, :]
                # Number of unique obj in this sequence of frames
                obj_id_list = np.unique(seq_frame_data[:, :, 0])
                num_unique_obj = obj_id_list.shape[0]

                source_data = np.zeros((self.seq_length, self.max_num_obj, 3))
                target_data = np.zeros((self.seq_length, self.max_num_obj, 3))

                for seq in range(self.seq_length):
                    sseq_frame_data = seq_source_frame_data[seq, :]
                    tseq_frame_data = seq_target_frame_data[seq, :]
                    for obj in range(num_unique_obj):
                        obj_id = obj_id_list[obj]

                        if obj_id == 0:
                            continue
                        else:
                            sobj = sseq_frame_data[sseq_frame_data[:, 0] == obj_id, :]
                            tobj = np.squeeze(tseq_frame_data[tseq_frame_data[:, 0] == obj_id, :])
                            if sobj.size != 0:
                                source_data[seq, obj, :] = sobj
                            if tobj.size != 0:
                                target_data[seq, obj, :] = tobj

                x_batch.append(source_data)
                y_batch.append(target_data)

                # Advance the frame pointer to a random point
                if random_update:
                    self.frame_pointer += random.randint(1, self.seq_length)
                else:
                    self.frame_pointer += self.seq_length

                dval.append(self.dataset_pointer)
                i += 1
            else:
                # Not enough frames left
                # Increment the dataset pointer and set the frame_pointer to zero
                self.tick_batch_pointer()

        return x_batch, y_batch, dval, current_ref_filenames

    def tick_batch_pointer(self):
        '''
        Advance the dataset pointer
        '''
        # Go to the next dataset
        self.dataset_pointer += 1
        self.frame_pointer = 0
        # If all datasets are done, then go to the first one again
        if self.dataset_pointer >= len(self.data):
            self.dataset_pointer = 0

    def reset_batch_pointer(self):
        '''
        Reset all pointers
        '''
        # Go to the first frame of the first dataset
        self.dataset_pointer = 0
        self.frame_pointer = 0

    def get_image_size(self, fname):
        '''Determine the image type of fhandle and return its size.
        from draco'''
        with open(fname, 'rb') as fhandle:
            head = fhandle.read(24)
            if len(head) != 24:
                return
            if imghdr.what(fname) == 'png':
                check = struct.unpack('>i', head[4:8])[0]
                if check != 0x0d0a1a0a:
                    return
                width, height = struct.unpack('>ii', head[16:24])
            elif imghdr.what(fname) == 'gif':
                width, height = struct.unpack('<HH', head[6:10])
            elif imghdr.what(fname) == 'jpeg':
                try:
                    fhandle.seek(0) # Read 0xff next
                    size = 2
                    ftype = 0
                    while not 0xc0 <= ftype <= 0xcf:
                        fhandle.seek(size, 1)
                        byte = fhandle.read(1)
                        while ord(byte) == 0xff:
                            byte = fhandle.read(1)
                        ftype = ord(byte)
                        size = struct.unpack('>H', fhandle.read(2))[0] - 2
                    # We are at a SOFn block
                    fhandle.seek(1, 1)  # Skip `precision' byte.
                    height, width = struct.unpack('>HH', fhandle.read(4))
                except Exception: #IGNORE:W0703
                    return
            else:
                return
            return width, height
