'''
Train script for the DESIRE implementation
Initiates the training of the model

Author : Todor Davchev
Date : 13th February 2017
'''
import argparse
import os
import pickle
import sys
import time

import ipdb
import numpy as np
import tensorflow as tf

#import desire.utils.data_loader as dl
import data_loader as dl ##changed
# from desire.model import model

# from grid import getSequenceGridMask


def main():
    '''
    Main function. Sets up all arguments
    '''
    parser = argparse.ArgumentParser()
    # RNN size parameter (dimension of the output/hidden state)
    parser.add_argument('--rnn_size', type=int, default=512,
                        help='size of RNN hidden state')
    # Number of layers parameter
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in the RNN')
    # Model currently not used. Only LSTM implemented
    # Type of recurrent unit parameter
    parser.add_argument('--model', type=str, default='gru',
                        help='rnn, gru, or lstm')
    # Size of each batch parameter
    parser.add_argument('--batch_size', type=int, default=10,
                        help='minibatch size')
    # Length of sequence to be considered parameter
    parser.add_argument('--seq_length', type=int, default=8,
                        help='RNN sequence length')
    # Number of epochs parameter
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='number of epochs')
    # Frequency at which the model should be saved parameter
    parser.add_argument('--save_every', type=int, default=400,
                        help='save frequency')
    # Gradient value at which it should be clipped
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    # Learning rate parameter
    parser.add_argument('--learning_rate', type=float, default=0.005,
                        help='learning rate')
    # Decay rate for the learning rate parameter
    parser.add_argument('--decay_rate', type=float, default=0.95,
                        help='decay rate for rmsprop')
    # Dropout not implemented.
    # Dropout probability parameter
    parser.add_argument('--keep_prob', type=float, default=0.8,
                        help='dropout keep probability')
    # Dimension of the embeddings parameter
    parser.add_argument('--embedding_size', type=int, default=64,
                        help='Embedding dimension for the spatial coordinates')
    # Size of neighborhood to be considered parameter
    parser.add_argument('--neighborhood_size', type=int, default=32,
                        help='Neighborhood size to be considered for social grid')
    # Size of the social grid parameter
    parser.add_argument('--grid_size', type=int, default=4,
                        help='Grid size of the social grid')
    # Maximum number of pedestrians to be considered
    parser.add_argument('--max_num_obj', type=int, default=60,
                        help='Maximum Number of Moving objects')
    # The leave out dataset
    parser.add_argument('--leave_dataset', type=int, default=5,
                        help='The dataset index to be left out in training')
    # The latent size for CVAE
    parser.add_argument('--latent_size', type=int, default=128,
                        help='The dataset index to be left out in training')
    # The CVAE encoder's dimension
    parser.add_argument('--e_dim', type=int, default=256,
                        help='The encoder\'s output dimension')
    parser.add_argument('--d_dim', type=int, default=16,
                        help='The decoder\'s output dimension')
    parser.add_argument('--stride', type=int, default=1,
                        help='Stride size for the Temporal Convolution')

    args = parser.parse_args()
    train(args)


def train(args):
    '''
    The actual train function
    '''
    # Create the DataLoader object
    data_loader = dl.DataLoader(args.batch_size, args.seq_length,
                             args.max_num_obj, args.leave_dataset, preprocess=False)

    with open(os.path.join('save', 'config.pkl'), 'wb') as file:
        pickle.dump(args, file)

    # Create a model object with the arguments
    model = model.DESIREModel(args)

    # Initialize a TensorFlow session
    with tf.Session() as sess:
        # Initialize all variables in the graph
        sess.run(tf.global_variables_initializer())
        #sess.run(tf.initialize_all_variables())
        # Initialize a saver that saves all the variables in the graph
        saver = tf.train.Saver(tf.global_variables())
        #saver = tf.train.Saver(tf.all_variables())

        #summary_writer = tf.train.SummaryWriter('/tmp/lstm/logs', graph_def=sess.graph_def)

        # For each epoch
        for epoch in range(args.num_epochs): #100
            # Assign the learning rate value for this epoch
            sess.run(
                tf.assign(
                    model.learning_rate, args.learning_rate * (args.decay_rate ** epoch)
                    )
                )
            # Reset the data pointers in the data_loader
            data_loader.reset_batch_pointer()

            # For each batch
            for batch in range(data_loader.num_batches):#58
                # Tic
                start = time.time()

                # Get the source, target and dataset data for the next batch
                # x, y are input and target data which are lists containing numpy
                # arrays of size seq_length x max_num_objs x 3
                # d is the list of dataset indices from which each batch is generated
                # (used to differentiate between datasets)
                xval, yval, dval = data_loader.next_batch() # x and y are 10x8x40x3

                # variable to store the loss for this batch
                loss_batch = 0

                # For each sequence in the batch
                for batch in range(data_loader.batch_size):
                    # x_batch, y_batch and d_batch contains the source,
                    # target and dataset index data for
                    # seq_length long consecutive frames in the dataset
                    # x_batch, y_batch would be numpy arrays of size seq_length x max_num_objs x 3
                    # d_batch would be a scalar identifying the dataset from which this sequence is
                    # extracted
                    x_batch, y_batch, d_batch = xval[batch], yval[batch], dval[batch]

                    # need to split it by 8 x 4 ...
                    # grid_batch = getSequenceGridMask(x_batch, dataset_data,
                    #                                  args.neighborhood_size, args.grid_size)
                    x_batch = np.reshape(x_batch,
                                         [args.seq_length,
                                          args.max_num_obj,
                                          3])

                    # grid_batch = np.reshape(grid_batch,
                    #                         [args.seq_length,
                    #                          args.obs_length,
                    #                          args.max_num_obj,
                    #                          args.max_num_obj,
                    #                          args.grid_size*args.grid_size])

                    y_batch = np.reshape(y_batch,
                                         [args.seq_length,
                                          args.max_num_obj,
                                          3])

                    # Feed the source, target data
                    feed = {
                        model.input_data: x_batch,
                        model.target_data: y_batch
                        }

                    train_loss = sess.run(model.cost, feed)

                    loss_batch += train_loss

                end = time.time()
                loss_batch = loss_batch / data_loader.batch_size
                print(
                    "{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}"
                    .format(
                        epoch * data_loader.num_batches + batch,
                        args.num_epochs * data_loader.num_batches,
                        epoch,
                        loss_batch, end - start))
                sys.stdout.flush()

                # Save the model if the current epoch and batch number match the frequency
                if (epoch * data_loader.num_batches + batch) % args.save_every == 0 \
                    and ((epoch * data_loader.num_batches + batch) > 0):

                    checkpoint_path = os.path.join('save', 'social_model.ckpt')
                    saver.save(
                        sess,
                        checkpoint_path,
                        global_step=epoch * data_loader.num_batches + batch
                    )
                    print("model saved to {}".format(checkpoint_path))
                    sys.stdout.flush()

if __name__ == '__main__':
    main()
