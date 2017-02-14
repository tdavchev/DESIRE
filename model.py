'''
DESIRE: Deep Stochastic IOC RNN Encoder-decoder for Distant Future
Prediction in Dynamic Scenes with Multiple Interacting Agents

Author: Todor Davchev
Date: 13th February 2017
'''

import copy
import random
import time

# from grid import getSequenceGridMask
import ipdb
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import rnn, rnn_cell, seq2seq

class DESIREModel(object):
    '''
    DESIRE model. Represents a Stochastic Inverse Reinforcement Learning
    Encoder-Decoder for Distant Future Prediction in Dynamic Scenes with
    Multiple Interacting Agents
    '''

    def __init__(self, args):
        self.args = args
        self.rnn_size = args.rnn_size # hidden_features
        self.seq_length = args.seq_length # time_steps
        self.input_size = 3
        self.encoder_output = 128
        self.num_layers = args.num_layers
        self.batch_size = batch_size

        xval = None
        yval_enc = None
        yval = None
        optimizer = None
        accuracy = None
        acc_summary = None

        build_model()

    def build_model(self):
        '''
        Building the DESIRE Model
        '''
        xval = tf.placeholder("float", [None, self.seq_length, self.input_size])
        hidden_state_x = tf.placeholder("float", [None, self.rnn_size], name="Hidden_x")
        yval_enc = tf.placeholder("float", [None, self.seq_length, self.input_size])
        hidden_state_y = tf.placeholder("float", [None, self.rnn_size], name="Hidden_y")
        yval = tf.placeholder("float", [None, output_classes], name="Output")

        # Weights adn Biases for hidden layer and output layer
        w_hidden = tf.Variable(tf.random_normal([self.input_size, self.rnn_size]))
        w_out = tf.Variable(tf.random_normal([self.rnn_size, self.encoder_output]))
        b_hidden = tf.Variable(tf.random_normal([hidden_features]))
        b_out = tf.Variable(tf.random_normal([output_classes]))

        # The Formula for the Model
        input_x_ = tf.reshape(X, [-1, self.input_size])
        lstm_cell = tf.nn.rnn_cell.GRUCell(self.rnn_size)
        input_x_2 = tf.split(0, self.seq_length, input_x_)

        input_y_ = tf.reshape(X, [-1, self.input_size])
        lstm_cell_y = tf.nn.rnn_cell.GRUCell(self.rnn_size)
        input_y_2 = tf.split(0, self.seq_length, input_y_)

        cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell]* self.num_layers, state_is_tuple=True)
        hidden_state = cells.zero_state(self.batch_size, tf.float32)

        cells_y = tf.nn.rnn_cell.MultiRNNCell([lstm_cell]* self.num_layers, state_is_tuple=True)
        hidden_state = cells_y.zero_state(self.batch_size, tf.float32)

        with tf.variable_scope("basic_rnn_seq2seq"):
            _, enc_state_x = rnn.rnn(cells, input_x_2, dtype=dtypes.float32)

        with tf.variable_scope("basic_rnn_seq2seq"):
            _, enc_state_y = rnn.rnn(cells_y, input_y_2, dtype=dtypes.float32)
