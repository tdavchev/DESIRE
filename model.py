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
import prettytensor as pt
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell, seq2seq
from tensorflow.python.framework import dtypes

from convolutional_vae_util import deconv2d


class DESIREModel(object):
    '''
    DESIRE model. Represents a Stochastic Inverse Reinforcement Learning
    Encoder-Decoder for Distant Future Prediction in Dynamic Scenes with
    Multiple Interacting Agents
    '''

    def __init__(self, args):
        args = args
        input_size = 3
        rnn_size = args.rnn_size # hidden_features
        seq_length = args.seq_length # time_steps
        encoder_output = 512
        num_layers = args.num_layers
        batch_size = args.batch_size
        output_classes = 5 #primerno
        latent_size = args.latent_size
        input_shape = [int(np.sqrt(2*rnn_size)), int(np.sqrt(2*rnn_size))]
        vae_input_size = np.prod(input_shape)

        xval = None
        yval_enc = None
        yval = None
        optimizer = None
        accuracy = None
        acc_summary = None

        build_model()

    def vae_decoder(self, zval, projection_size, activ=tf.nn.elu, phase=pt.Phase.train):
        '''
        C-VAE Decoder from https://github.com/jramapuram/CVAE/blob/master/cvae.py
        '''
        with pt.defaults_scope(activation_fn=activ,
                               batch_normalize=True,
                               learned_moments_update_rate=0.0003,
                               variance_epsilon=0.001,
                               scale_after_normalization=True,
                               phase=phase):
            return (pt.wrap(zval).
                    reshape([-1, 1, 1, latent_size]).
                    deconv2d(4, 128, edges='VALID', phase=phase).
                    deconv2d(5, 64, edges='VALID', phase=phase).
                    deconv2d(5, 32, stride=2, phase=phase).
                    deconv2d(5, 1, stride=2, activation_fn=tf.nn.sigmoid, phase=phase).
                    flatten()).tensor

    def vae_encoder(self, inputs, latent_size, activ=tf.nn.elu, phase=pt.Phase.train):
        '''
        C-VAE Encoder from https://github.com/jramapuram/CVAE/blob/master/cvae.py
        Accepts a cube as inputs and performs 2D convolutions on it
        '''
        with pt.defaults_scope(activation_fn=activ,
                               batch_normalize=True,
                               learned_moments_update_rate=0.0003,
                               variance_epsilon=0.001,
                               scale_after_normalization=True,
                               phase=phase):
            params = (pt.wrap(inputs).
                      reshape([-1, input_shape[0], input_shape[1], 1]).
                      conv2d(5, 32, stride=2).
                      conv2d(5, 64, stride=2).
                      conv2d(5, 128, edges='VALID').
                      flatten().
                      fully_connected(latent_size * 2, activation_fn=None)).tensor

        mean = params[:, :latent_size]
        stddev = params[:, latent_size:]
        return [mean, stddev]

    def build_model(self):
        '''
        Building the DESIRE Model
        '''
        xval = tf.placeholder("float", [None, seq_length, input_size])
        hidden_state_x = tf.placeholder("float", [None, rnn_size], name="Hidden_x")
        yval_enc = tf.placeholder("float", [None, seq_length, input_size])
        hidden_state_y = tf.placeholder("float", [None, rnn_size], name="Hidden_y")

        yval = tf.placeholder("float", [None, output_classes], name="Output")

        # Weights adn Biases for hidden layer and output layer
        # TODO:Make sure you learn the dimensionalities!!!!!
        w_hidden_enc1 = tf.Variable(tf.random_normal([vae_input_size, vae_input_size]))
        b_hidden_enc1 = tf.Variable(tf.random_normal([vae_input_size]))

        w_post_vae = tf.Variable(tf.random_normal([vae_input_size, rnn_size]))
        b_post_vae = tf.Variable(tf.random_normal([rnn_size]))

        w_out = tf.Variable(tf.random_normal([rnn_size, output_classes]))
        b_out = tf.Variable(tf.random_normal([output_classes]))

        # The Formula for the Model
        input_x_ = tf.reshape(xval, [-1, input_size])
        lstm_cell = tf.nn.rnn_cell.GRUCell(rnn_size)
        input_x_2 = tf.split(0, seq_length, input_x_)

        input_y_ = tf.reshape(yval_enc, [-1, input_size])
        lstm_cell_y = tf.nn.rnn_cell.GRUCell(rnn_size)
        input_y_2 = tf.split(0, seq_length, input_y_)

        cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell]* num_layers, state_is_tuple=True)
        hidden_state_x = cells.zero_state(batch_size, tf.float32)

        cells_y = tf.nn.rnn_cell.MultiRNNCell([lstm_cell]* num_layers, state_is_tuple=True)
        hidden_state_y = cells_y.zero_state(batch_size, tf.float32)

        # Encoder
        with tf.variable_scope("encoder1"):
            _, enc_state_x = rnn.rnn(cells, input_x_2, dtype=dtypes.float32)

        with tf.variable_scope("encoder2"):
            _, enc_state_y = rnn.rnn(cells_y, input_y_2, dtype=dtypes.float32)

        # summary c
        concat_ = tf.squeeze(tf.concat(2, [enc_state_x, enc_state_y]), [0])

        # fc layer
        vae_inputs = tf.add(tf.matmul(concat_, w_hidden_enc1), b_hidden_enc1)
        vae_inputs = tf.nn.relu(vae_inputs)

        # Convolutional VAE
        # z = mu + sigma * epsilon
        # epsilon is a sample from a N(0, 1) distribution
        # Encode our data into z and return the mean and covariance
        # TODO: Checkout the batch size, seq size
        with tf.variable_scope("zval34321"):
            z_mean, z_log_sigma_sq = vae_encoder(vae_inputs, latent_size)
            eps_batch = z_log_sigma_sq.get_shape().as_list()[0] \
                if z_log_sigma_sq.get_shape().as_list()[0] is not None else batch_size
            eps = tf.random_normal(
                [eps_batch, latent_size], 0.0, 1.0, dtype=tf.float32)
            zval = tf.add(z_mean, tf.mul(tf.sqrt(tf.exp(z_log_sigma_sq)), eps))
            # Get the reconstructed mean from the decoder
            x_reconstr_mean = vae_decoder(zval, vae_input_size)
            z_summary = tf.summary.histogram("zval", zval)

        # fc layer
        multipl = tf.add(tf.matmul(x_reconstr_mean, w_post_vae), b_post_vae)
        multipl = tf.nn.relu(multipl)

        # Decoder 1
        decoder1_inputs = tf.squeeze(tf.mul(multipl, enc_state_x), [0])
        decoder1_inputs = tf.split(0, batch_size, decoder1_inputs)
        with tf.variable_scope("decoder1"):
            outputs, state = seq2seq.rnn_decoder(decoder1_inputs, enc_state_x, cells)

