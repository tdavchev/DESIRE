'''
Utils script for the DESIRE implementation
Handles processing the input and target data in batches and sequences

Author : Todor Davchev
Date : 13th February 2017
'''

import os
import pickle
import numpy as np

# The data loader class that loads data from the datasets considering
# each frame as a datapoint and a sequence of consecutive frames as the
# sequence.
class DataLoader(object):
    """All frobnication, all the time."""
    # Eight is reasonable in this case.

    def __init__(self,
                 batch_size=50, seq_length=5, max_num_peds=40, leave_dataset=1):
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

        # Maximum number of peds in a single frame (Number obtained by checking the datasets)
        self.max_num_peds = max_num_peds

        # Store the arguments
        self.batch_size = batch_size
        self.seq_length = seq_length

        # Define the path in which the process data would be stored
        data_file = os.path.join(data_dir, "trajectories.cpkl")

        # If the file doesn't exist or forcePreProcess is true
        if not os.path.exists(data_file):
            print "Creating pre-processed data from raw data"
            # Preprocess the data from the csv files of the datasets
            # Note that this data is processed in frames
            self.frame_preprocess(self.data_dir, data_file)

        # Load the processed data from the pickle file
        self.load_preprocessed(data_file)
        # Reset all the data pointers of the dataloader object
        self.reset_batch_pointer()

    def frame_preprocess(self, data_dir, data_file):
        '''
        Function that will pre-process the annotations_processed.csv files of each dataset
        into data with occupancy grid that can be used
        params:
        data_file : The file into which all the pre-processed data needs to be stored
        '''

        # all_frame_data would be a list of numpy arrays corresponding to each dataset
        # Each numpy array would be of size (numFrames, maxNumPeds, 3) where each pedestrian's
        # ped_id, x, y , in each frame is stored
        all_frame_data = []
        # frame_list_data would be a list of lists corresponding to each dataset
        # Each list would contain the frameIds of all the frames in the dataset
        frame_list_data = []
        # numPeds_data would be a list of lists corresponding to each dataset
        # Each list would contain the number of pedestrians in each frame in the dataset
        num_peds_data = []
        # Index of the current dataset
        dataset_index = 0

        # For each dataset
        for subdir, dirs, files in os.walk(data_dir):
            for file in files:
                if dataset_index != leave_dataset and \
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

                    # Initialize the list of num_peds for the current dataset
                    num_peds_data.append([])

                    # Initialize the numpy array for the current dataset
                    all_frame_data.append((np.zeros((num_frames, self.max_num_peds, 3))))

                    # index to maintain the current frame
                    curr_frame = 0
                    for frame in frame_list:
                        # Extract all pedestrians in current frame
                        peds_in_frame = data[:, data[0, :] == frame]

                        # Extract peds list
                        peds_list = peds_in_frame[1, :].tolist()

                        # Add number of peds in the current frame to the stored data
                        num_peds_data[dataset_index].append(len(peds_list))

                        # Initialize the row of the numpy array
                        peds_with_pos = []

                        # For each ped in the current frame
                        for ped in peds_list:
                            # Extract their x and y positions
                            current_x = peds_in_frame[2, peds_in_frame[1, :] == ped][0]
                            current_y = peds_in_frame[3, peds_in_frame[1, :] == ped][0]

                            # Add their ped_id, x, y to the row of the numpy array
                            peds_with_pos.append([ped, current_x, current_y])

                        # Add the details of all the peds in the current frame to all_frame_data
                        all_frame_data[dataset_index][curr_frame, 0:len(peds_list), :] = \
                            np.array(peds_with_pos)

                        # Increment the frame index
                        curr_frame += 1
                    # Increment the dataset index
                    dataset_index += 1

        # Save the tuple (all_frame_data, frame_list_data, num_peds_data) in the pickle file
        file = open(data_file, "wb")
        pickle.dump((all_frame_data, frame_list_data, num_peds_data), file, protocol=2)
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
        self.num_peds_list = self.raw_data[2]
        counter = 0

        # For each dataset
        for dataset, _data in enumerate(self.data):
            # get the frame data for the current dataset
            all_frame_data = self.data[dataset]
            print len(all_frame_data)
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
        # Source data
        x_batch = []
        # Target data
        y_batch = []
        # Iteration index
        i = 0
        while i < self.batch_size:
            # Extract the frame data of the current dataset
            frame_data = self.data[self.dataset_pointer]
            # Get the frame pointer for the current dataset
            idx = self.frame_pointer
            # While there is still seq_length number of frames left in the current dataset
            if idx + self.seq_length < frame_data.shape[0]:
                # All the data in this sequence
                seq_frame_data = frame_data[idx:idx+self.seq_length+1, :]
                seq_source_frame_data = frame_data[idx:idx+self.seq_length, :]
                seq_target_frame_data = frame_data[idx+1:idx+self.seq_length+1, :]
                # Number of unique peds in this sequence of frames
                ped_id_list = np.unique(seq_frame_data[:, :, 0])
                num_unique_peds = ped_id_list.shape[0]

                source_data = np.zeros((self.seq_length, self.max_num_peds, 3))
                target_data = np.zeros((self.seq_length, self.max_num_peds, 3))

                for seq in range(self.seq_length):
                    sseq_frame_data = seq_source_frame_data[seq, :]
                    tseq_frame_data = seq_target_frame_data[seq, :]
                    for ped in range(num_unique_peds):
                        ped_id = ped_id_list[ped]

                        if ped_id == 0:
                            continue
                        else:
                            sped = sseq_frame_data[sseq_frame_data[:, 0] == ped_id, :]
                            tped = np.squeeze(tseq_frame_data[tseq_frame_data[:, 0] == ped_id, :])
                            if sped.size != 0:
                                source_data[seq, ped, :] = sped
                            if tped.size != 0:
                                target_data[seq, ped, :] = tped

                x_batch.append(source_data)
                y_batch.append(target_data)

                # Advance the frame pointer to a random point
                if random_update:
                    self.frame_pointer += random.randint(1, self.seq_length)
                else:
                    self.frame_pointer += self.seq_length

                d.append(self.dataset_pointer)
                i += 1
            else:
                # Not enough frames left
                # Increment the dataset pointer and set the frame_pointer to zero
                self.tick_batch_pointer()

        return x_batch, y_batch, d

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
