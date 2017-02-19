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
        self.args = args
        # input depth = sequence
        # input_height = max number of people
        # input width = id,x,y
        self.temporal_kernel_size = 2
        self.temporal_num_channels = 1
        self.temporal_depth = 2
        self.input_size = 3
        self.rnn_size = args.rnn_size # hidden_features
        self.seq_length = args.seq_length # time_steps
        self.encoder_output = 512
        self.num_layers = args.num_layers
        self.batch_size = args.batch_size
        self.latent_size = args.latent_size
        self.input_shape = [int(np.sqrt(2*self.rnn_size)), int(np.sqrt(2*self.rnn_size))]
        self.vae_input_size = np.prod(self.input_shape)
        self.max_num_obj = args.max_num_obj

        self.input_data = None
        self.target_data_enc = None
        self.target_data = None
        self.optimizer = None
        self.accuracy = None
        self.acc_summary = None
        self.learning_rate = None
        self.output_size = None
        self.hidden_state_x = None
        self.hidden_state_y = None
        self.gru_states = None
        self.output_states = None
        self.spatial_input = None
        self.enc_state_x = None
        self.enc_state_y = None

        self.build_model()

    def build_model(self):
        '''
        Building the DESIRE Model
        '''
        # TODO: fix input size to be of size MNOx3 and convolve over the MNOx2D matrix
        # TODO: fix temporal_data to be of seqxMNOxinput sizeinstead
        # self.temporal_data = tf.placeholder("float", [seq_length, input_size, 1])
        self.input_data = tf.placeholder(tf.float32, \
            [self.args.seq_length, self.args.max_num_obj, self.input_size], name="input_data")
        self.hidden_state_x = tf.placeholder("float", [self.rnn_size], name="Hidden_x")
        self.target_data_enc = tf.placeholder("float", \
            [self.seq_length, self.args.max_num_obj, self.input_size], name="target_data_enc")
        self.hidden_state_y = tf.placeholder(tf.float32, [self.rnn_size], name="Hidden_y")
        self.target_data = tf.placeholder("float", \
            [self.seq_length, self.args.max_num_obj, self.input_size], name="target_data")
        self.learning_rate = \
            tf.Variable(self.args.learning_rate, trainable=False, name="learning_rate")
        self.output_size = 5

        weights, biases = self.define_weights()

        # # The Formula for the Model
        # # Temporal convolution
        # with tf.variable_scope("temporal_convolution"):
        #     self.rho_i = tf.nn.relu(tf.add( \
        #         tf.nn.depthwise_conv2d(
        #             self.temporal_data, weights["temporal_w"], [1, 1, 1, 1], padding='VALID'),
        #         biases["temporal_b"]))

        # Encoder
        with tf.variable_scope("gru_cell"):
            lstm_cell = tf.nn.rnn_cell.GRUCell(self.rnn_size)
            cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell]* num_layers, state_is_tuple=False)

        with tf.variable_scope("gru_y_cell"):
            lstm_cell_y = tf.nn.rnn_cell.GRUCell(self.rnn_size)
            cells_y = tf.nn.rnn_cell.MultiRNNCell([lstm_cell]* num_layers, state_is_tuple=False)

        # Define LSTM states for each pedestrian
        with tf.variable_scope("gru_states"):
            self.gru_states = tf.zeros([args.max_num_obj, cells.state_size], name="gru_states")
            self.enc_state_x = tf.split(0, args.max_num_obj, self.gru_states)
            self.enc_state_y = tf.split(0, args.max_num_obj, self.gru_states)

        # Define hidden output states for each pedestrian
        with tf.variable_scope("output_states"):
            output_states = \
                tf.split(0, self.args.max_num_obj, \
                    tf.zeros([self.args.max_num_obj, cells.output_size]))

        # List of tensors each of shape args.maxNumPedsx3 corresponding to
        # each frame in the sequence
        with tf.name_scope("frame_data_tensors"):
            frame_data = [tf.squeeze(input_, [0]) \
                for input_ in tf.split(0, self.args.seq_length, self.input_data)]

        with tf.name_scope("frame_target_data_tensors"):
            frame_target_data = [tf.squeeze(target_, [0]) \
                for target_ in tf.split(0, self.args.seq_length, self.target_data)]

        # Cost
        with tf.name_scope("Cost_related_stuff"):
            self.cost = tf.constant(0.0, name="cost")
            self.counter = tf.constant(0.0, name="counter")
            self.increment = tf.constant(1.0, name="increment")

        # Containers to store output distribution parameters
        with tf.name_scope("Distribution_parameters_stuff"):
            self.initial_output = \
                tf.split(0, self.args.max_num_obj, \
                    tf.zeros([self.args.max_num_obj, self.output_size]))

        # Tensor to represent non-existent ped
        with tf.name_scope("Non_existent_obj_stuff"):
            nonexistent_obj = tf.constant(0.0, name="zero_obj")

        for seq, frame in enumerate(frame_data):
            current_frame_data = frame  # MNP x 3 tensor
            current_target_frame_data = frame_target_data[seq] # MNP x 3 tensor
            for obj in xrange(0, args.max_num_obj):
                obj_id = current_frame_data[obj, 0]
                with tf.name_scope("extract_input_obj"):
                    spatial_input_x = tf.slice(current_frame_data, [obj, 1], [1, 2])
                    spatial_input_y = tf.slice(current_target_frame_data, [obj, 1], [1, 2])

                with tf.variable_scope("encoding_operations_x") as scope:
                    if seq > 0 or obj > 0:
                        scope.reuse_variables()
                    _, self.enc_state_x = rnn.rnn(cells, [spatial_input_x], dtype=dtypes.float32)

                with tf.variable_scope("encoding_operations_y") as scope:
                    if seq > 0 or obj > 0:
                        scope.reuse_variables()
                    _, self.enc_state_y = rnn.rnn(cells_y, [spatial_input_y], dtype=dtypes.float32)

                with tf.name_scope("concatenate_embeddings"):
                    # Concatenate the summaries c1 and c2
                    complete_input = tf.concat(1, [self.enc_state_x, self.enc_state_y])

                # fc layer
                with tf.variable_scope("fc_c"):
                    vae_inputs = tf.nn.relu( \
                        tf.nn.xw_plus_b( \
                            complete_input, weights["w_hidden_enc1"], biases["b_hidden_enc1"]))

                # Convolutional VAE
                # z = mu + sigma * epsilon
                # epsilon is a sample from a N(0, 1) distribution
                # Encode our data into z and return the mean and covariance
                with tf.variable_scope("zval"):
                    z_mean, z_log_sigma_sq = vae_encoder(vae_inputs, self.latent_size)
                    eps_batch = z_log_sigma_sq.get_shape().as_list()[0] \
                        if z_log_sigma_sq.get_shape().as_list()[0] is not None else self.batch_size
                    eps = tf.random_normal(
                        [eps_batch, self.latent_size], 0.0, 1.0, dtype=tf.float32)
                    zval = tf.add(z_mean, tf.mul(tf.sqrt(tf.exp(z_log_sigma_sq)), eps))
                    # Get the reconstructed mean from the decoder
                    x_reconstr_mean = vae_decoder(zval, self.vae_input_size)
                    z_summary = tf.summary.histogram("zval", zval)

                # fc layer
                with tf.variable_scope("fc_softmax"):
                    multipl = tf.add(
                        tf.matmul(x_reconstr_mean, weights["w_post_vae"]),
                        biases["b_post_vae"])
                    multipl = tf.nn.relu(multipl)
                    multipl = tf.nn.softmax(multipl)

                # Decoder 1
                with tf.variable_scope("hidden_states"):
                    self.hidden_state_x = tf.mul(multipl, self.enc_state_x)
                    output_states, self.hidden_state_x = \
                        seq2seq.rnn_decoder([self.hidden_state_x], self.enc_state_x, cells)

                # Apply the linear layer. Output would be a tensor of shape 1 x output_size
                with tf.name_scope("output_linear_layer"):
                    self.initial_output[obj] = \
                        tf.nn.xw_plus_b(output_states[obj], weights["output_w"], biases["output_b"])

                with tf.name_scope("extract_target_obj"):
                    # Extract x and y coordinates of the target data
                    # x_data and y_data would be tensors of shape 1 x 1
                    [x_data, y_data] = \
                        tf.split(1, 2, tf.slice(current_target_frame_data, [obj, 1], [1, 2]))
                    target_obj_id = current_target_frame_data[obj, 0]

                with tf.name_scope("get_coef"):
                    # Extract coef from output of the linear output layer
                    [o_mux, o_muy, o_sx, o_sy, o_corr] = self.get_coef(self.initial_output[obj])

                # TODO: check if KLD loss actually has reconstruction loss in it
                # TODO: make sure that the CVAE implementation is truly from the same paper
                # TODO: Figure out how/if necessary to divide by K the reconstr_loss
                with tf.name_scope("calculate_loss"):
                    # Calculate loss for the current ped
                    reconstr_loss = \
                        self.get_reconstr_loss(o_mux, o_muy, o_sx, o_sy, o_corr, x_data, y_data)
                    kld_loss = self.kld_loss(
                        vae_inputs,
                        x_reconstr_mean,
                        z_log_sigma_sq,
                        z_mean
                    )
                    loss = tf.reduce_mean(reconstr_loss+kld_loss)

                with tf.name_scope("increment_cost"):
                    # If it is a non-existent object, it should not contribute to cost
                    # If the object doesn't exist in the next frame, he/she/it should not
                    # contribute to cost as well
                    self.cost = tf.select( \
                        tf.logical_or( \
                            tf.equal(obj_id, nonexistent_obj), \
                            tf.equal(target_obj_id, nonexistent_obj)), \
                        self.cost, \
                        tf.add(self.cost, loss))
                    self.counter = tf.select( \
                        tf.logical_or( \
                            tf.equal(obj_id, nonexistent_obj), \
                            tf.equal(target_obj_id, nonexistent_obj)), \
                        self.counter, \
                        tf.add(self.counter, self.increment))

        with tf.name_scope("mean_cost"):
            # Mean of the cost
            self.cost = tf.div(self.cost, self.counter)

        # Get all trainable variables
        tvars = tf.trainable_variables()

        # Get the final LSTM states
        self.final_states = tf.concat(0, self.hidden_state_x)

        # Get the final distribution parameters
        self.final_output = self.initial_output

        # Compute gradients
        self.gradients = tf.gradients(self.cost, tvars)

        # Clip the gradients
        grads, _ = tf.clip_by_global_norm(self.gradients, self.args.grad_clip)

        # Define the optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)

        # The train operator
        # train_op = optimizer.apply_gradients(zip(grads, tvars))

    def define_weights(self):
        ''' Define Model's weights'''
        # Weights adn Biases for hidden layer and output layer
        # TODO:Make sure you learn the dimensionalities!!!!!
        weights, biases = {}, {}
        with tf.variable_scope("temporal_weights"):
            weights["temporal_w"] = tf.Variable(tf.random_normal( \
                [1, self.temporal_kernel_size, self.temporal_num_channels, self.temporal_depth]))
            biases["temporal_b"] = tf.Variable(tf.random_normal( \
                [temporal_num_channels*temporal_depth]))

        with tf.variable_scope("hidden_enc_weights"):
            weights["w_hidden_enc1"] = tf.Variable(tf.random_normal( \
                [self.vae_input_size, self.vae_input_size]))
            biases["b_hidden_enc1"] = tf.Variable(tf.random_normal( \
                [self.vae_input_size]))

        with tf.variable_scope("post_vae_weights"):
            weights["w_post_vae"] = tf.Variable(tf.random_normal( \
                [self.vae_input_size, self.rnn_size]))
            biases["b_post_vae"] = tf.Variable(tf.random_normal( \
                [self.rnn_size]))

        with tf.variable_scope("output_weights"):
            weights["output_w"] = tf.Variable(tf.random_normal( \
                [self.rnn_size, self.output_size]))
            biases["output_b"] = tf.Variable(tf.random_normal( \
                [self.output_size]))

        return weights, biases

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
                      reshape([-1, self.input_shape[0], self.input_shape[1], 1]).
                      conv2d(5, 32, stride=2).
                      conv2d(5, 64, stride=2).
                      conv2d(5, 128, edges='VALID').
                      flatten().
                      fully_connected(latent_size * 2, activation_fn=None)).tensor

        mean = params[:, :latent_size]
        stddev = params[:, latent_size:]
        return [mean, stddev]

    def tf_2d_normal(self, x, y, mux, muy, sx, sy, rho):
        '''
        Function that implements the PDF of a 2D normal distribution
        params:
        x : input x points
        y : input y points
        mux : mean of the distribution in x
        muy : mean of the distribution in y
        sx : std dev of the distribution in x
        sy : std dev of the distribution in y
        rho : Correlation factor of the distribution
        '''
        # eq 3 in the paper
        # and eq 24 & 25 in Graves (2013)
        # Calculate (x - mux) and (y-muy)
        normx = tf.sub(x, mux)
        normy = tf.sub(y, muy)
        # Calculate sx*sy
        sxsy = tf.mul(sx, sy)
        # Calculate the exponential factor
        z = tf.square(tf.div(normx, sx)) + tf.square(tf.div(normy, sy)) \
            - 2*tf.div(tf.mul(rho, tf.mul(normx, normy)), sxsy)
        negRho = 1 - tf.square(rho)
        # Numerator
        result = tf.exp(tf.div(-z, 2*negRho))
        # Normalization constant
        denom = 2 * np.pi * tf.mul(sxsy, tf.sqrt(negRho))
        # Final PDF calculation
        result = tf.div(result, denom)
        return result

    def get_reconstr_loss(self, z_mux, z_muy, z_sx, z_sy, z_corr, x_data, y_data):
        '''
        Function to calculate given a 2D distribution over x and y, and target data
        of observed x and y points
        params:
        z_mux : mean of the distribution in x
        z_muy : mean of the distribution in y
        z_sx : std dev of the distribution in x
        z_sy : std dev of the distribution in y
        z_rho : Correlation factor of the distribution
        x_data : target x points
        y_data : target y points
        '''
        # step = tf.constant(1e-3, dtype=tf.float32, shape=(1, 1))

        # Calculate the PDF of the data w.r.t to the distribution
        result0 = tf_2d_normal(x_data, y_data, z_mux, z_muy, z_sx, z_sy, z_corr)

        # For numerical stability purposes
        epsilon = 1e-20

        # Apply the log operation
        result1 = -tf.log(tf.maximum(result0, epsilon))  # Numerical stability

        # Sum up all log probabilities for each data point
        return tf.reduce_sum(result1)

    def get_coef(self, output):
        '''eq 20 -> 22 of Graves (2013)'''

        z = output
        # Split the output into 5 parts corresponding to means, std devs and corr
        z_mux, z_muy, z_sx, z_sy, z_corr = tf.split(1, 5, z)

        # The output must be exponentiated for the std devs
        z_sx = tf.exp(z_sx)
        z_sy = tf.exp(z_sy)
        # Tanh applied to keep it in the range [-1, 1]
        z_corr = tf.tanh(z_corr)

        return [z_mux, z_muy, z_sx, z_sy, z_corr]

    def kld_loss(self, inputs, x_reconstr_mean, z_log_sigma_sq, z_mean):
        '''Taken from https://jmetzen.github.io/2015-11-27/vae.html'''
        # The loss is composed of two terms:
        # 1.) The reconstruction loss (the negative log probability
        #     of the input under the reconstructed Bernoulli distribution
        #     induced by the decoder in the data space).
        #     This can be interpreted as the number of "nats" required
        #     for reconstructing the input when the activation in latent
        #     is given.
        # reconstr_loss = \
        #     -tf.reduce_sum(inputs * tf.log(tf.clip_by_value(x_reconstr_mean, 1e-10, 1.0))
        #                    + (1.0 - inputs) * tf.log(tf.clip_by_value(1.0 -
        #                                                             x_reconstr_mean, 1e-10, 1.0)),
        #                    1)
        # 2.) The latent loss, which is defined as the Kullback Libeler divergence
        # between the distribution in latent space induced by the encoder on
        #     the data and some prior. This acts as a kind of regularize.
        #     This can be interpreted as the number of "nats" required
        #     for transmitting the the latent space distribution given
        #     the prior.
        latent_loss = -0.5 * tf.reduce_sum(1.0 + z_log_sigma_sq \
                                                - tf.square(z_mean) \
                                                - tf.exp(z_log_sigma_sq), 1)
        kld_loss = tf.reduce_mean(reconstr_loss + latent_loss)   # average over batch

        return kld_loss

    def sample_gaussian_2d(self, mux, muy, sx, sy, rho):
        '''
        Function to sample a point from a given 2D normal distribution
        params:
        mux : mean of the distribution in x
        muy : mean of the distribution in y
        sx : std dev of the distribution in x
        sy : std dev of the distribution in y
        rho : Correlation factor of the distribution
        '''
        # Extract mean
        mean = [mux, muy]
        # Extract covariance matrix
        cov = [[sx*sx, rho*sx*sy], [rho*sx*sy, sy*sy]]
        # Sample a point from the multivariate normal distribution
        x = np.random.multivariate_normal(mean, cov, 1)
        return x[0][0], x[0][1]

    def sample(self, sess, traj, grid, dimensions, true_traj, num=10):
        # traj is a sequence of frames (of length obs_length)
        # so traj shape is (obs_length x maxNumPeds x 3)
        # grid is a tensor of shape obs_length x maxNumPeds x maxNumPeds x (gs**2)
        states = sess.run(self.gru_states)
        # print "Fitting"
        # For each frame in the sequence
        for index, frame in enumerate(traj[:-1]):
            data = np.reshape(frame, (1, self.max_num_obj, 3))
            target_data = np.reshape(traj[index+1], (1, self.max_num_obj, 3))

            feed = {
                self.input_data: data,
                self.gru_states: states,
                self.target_data: target_data
            }
            [states, cost] = sess.run([self.final_states, self.cost], feed)
            # print cost

        ret = traj

        last_frame = traj[-1]

        prev_data = np.reshape(last_frame, (1, self.max_num_obj, 3))

        prev_target_data = np.reshape(true_traj[traj.shape[0]], (1, self.max_num_obj, 3))
        # Prediction
        for t in range(num):
            print "**** NEW PREDICTION TIME STEP", t, "****"
            feed = {
                self.input_data: prev_data,
                self.gru_states: states,
                self.target_data: prev_target_data
            }
            [output, states, cost] = sess.run(
                [self.final_output, self.final_states, self.cost], feed)
            print "Cost", cost
            # Output is a list of lists where the inner lists contain matrices of shape 1x5.
            # The outer list contains only one element (since seq_length=1) and the inner list
            # contains maxNumPeds elements
            # output = output[0]
            newpos = np.zeros((1, self.max_num_obj, 3))
            for objindex, objoutput in enumerate(output):
                [o_mux, o_muy, o_sx, o_sy, o_corr] = np.split(objoutput[0], 5, 0)
                mux, muy, sx, sy, corr = \
                    o_mux[0], o_muy[0], np.exp(o_sx[0]), np.exp(o_sy[0]), np.tanh(o_corr[0])

                next_x, next_y = self.sample_gaussian_2d(mux, muy, sx, sy, corr)
                if next_x > 1.0:
                    next_x = 1.0
                if next_y > 1.0:
                    next_y = 1.0

                if prev_data[0, objindex, 0] != 0:
                    print "Pedestrian ID", prev_data[0, objindex, 0]
                    print "Predicted parameters", mux, muy, sx, sy, corr
                    print "New Position", next_x, next_y
                    print "Target Position", prev_target_data[0, objindex, 1], \
                        prev_target_data[0, objindex, 2]
                    print

                newpos[0, objindex, :] = [prev_data[0, objindex, 0], next_x, next_y]
            ret = np.vstack((ret, newpos))
            prev_data = newpos
            if t != num - 1:
                prev_target_data = \
                    np.reshape(true_traj[traj.shape[0] + t + 1], (1, max_num_obj, 3))

        # The returned ret is of shape (obs_length+pred_length) x maxNumPeds x 3
        return ret
