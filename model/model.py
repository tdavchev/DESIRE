'''
DESIRE: Deep Stochastic IOC RNN Encoder-decoder for Distant Future
Prediction in Dynamic Scenes with Multiple Interacting Agents

Author: Todor Davchev
Date: 13th February 2017
'''

import copy
import random
import time
import sys

# from grid import getSequenceGridMask
import ipdb
import numpy as np
import prettytensor as pt
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell, seq2seq
from tensorflow.python.framework import dtypes

sys.path.append("/home/s1045064/deep-learning/DESIRE")
execfile("utils/convolutional_vae_util.py")
# from convolutional_vae_util import deconv2d


class DESIREModel(object):
    '''
    DESIRE model. Represents a Stochastic Inverse Reinforcement Learning
    Encoder-Decoder for Distant Future Prediction in Dynamic Scenes with
    Multiple Interacting Agents
    '''

    def __init__(self, args):
        self.args = args
        # TODO: remove the unnecesary variables
        # TODO: rename decoder_output to hidden_features
        self.filter_height = 1
        self.filter_width = 3 # kernel of 3
        self.in_channels = 2 # one for x and y for each trajectory point
        self.out_channels = 16 # this is the feature map
        self.strides = 1 #[1, self.args.stride, self.args.stride, 1]
        self.input_size = 3
        self.grid_size = self.args.grid_size
        self.encoder_output = self.args.e_dim
        self.rnn_size = self.args.rnn_size # hidden_features
        # self.decoder_output = self.args.d_dim # hidden_features
        self.seq_length = self.args.seq_length # time_steps
        self.num_layers = self.args.num_layers
        self.batch_size = self.args.batch_size
        self.z_dim = self.args.latent_size
        # self.input_shape = [int(np.sqrt(2*self.rnn_size)),
        #                     int(np.sqrt(2*self.rnn_size))]
        # self.vae_input_size = np.prod(self.input_shape)
        self.vae_input_size = self.rnn_size
        self.max_num_obj = self.args.max_num_obj
        self.k_traj = 7

        self.output_states = None
        self.input_data = None
        # self.target_data_enc = None
        self.target_data = None
        self.optimizer = None
        self.accuracy = None
        self.acc_summary = None
        self.learning_rate = None
        self.output_size = None
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
        self.input_data = tf.placeholder(
            tf.float32,
            shape=[self.args.max_num_obj, self.args.seq_length, self.input_size],
            name="input_data"
        )
        self.target_data = tf.placeholder(
            tf.float32,
            shape=[self.args.max_num_obj, 2*self.seq_length, self.input_size],
            name="target_data"
        )
        self.learning_rate = tf.Variable(
            self.args.learning_rate,
            trainable=False,
            name="learning_rate"
        )

        self.z = tf.placeholder(
            tf.float32,
            shape=[None, self.z_dim],
            name="z"
        )
        # a grid cell of other pedestrian
        # i
        # self.grid_data = tf.placeholder(
        #     tf.float32,
        #     [self.seq_length, self.max_num_obj, self.max_num_obj, self.grid_size*self.grid_size],
        #     name="grid_data"
        # )

        self.output_size = self.seq_length * 2

        weights, biases = self.define_weights()

        temporal_shape = self.input_data.get_shape().as_list()
        temporal_shape[-1] = temporal_shape[-1] - 1
        temporal_input = tf.slice(
            self.input_data,
            [0, 0, 0],
            temporal_shape
        )
        temporal_ids = self.input_data[:, :, 2:]
        # The Formula for the Model
        # Temporal convolution
        with tf.variable_scope("temporal_convolution"):
            self.temporal_input = tf.nn.relu(tf.add( \
                tf.nn.conv1d(
                    temporal_input, weights["temporal_w"],
                    self.strides,
                    padding='SAME'
                ),
                biases["temporal_b"]))

        temporal_shape_y = self.target_data.get_shape().as_list()
        temporal_shape_y[-1] = temporal_shape_y[-1] - 1
        temporal_input_y = tf.slice(
            self.target_data,
            [0, 0, 0],
            temporal_shape_y
        )
        temporal_ids_y = self.target_data[:, :, 2:]
        # The Formula for the Model
        # Temporal convolution
        with tf.variable_scope("temporal_convolution_y"):
            self.temporal_input_y = tf.nn.relu(tf.add( \
                tf.nn.conv1d(
                    temporal_input_y, weights["temporal_w_y"],
                    self.strides,
                    padding='SAME'
                ),
                biases["temporal_b_y"]))

        # Encoder
        with tf.variable_scope("gru_cell"):
            gru_cell = tf.nn.rnn_cell.GRUCell(self.rnn_size)
            cells = tf.nn.rnn_cell.MultiRNNCell(
                [gru_cell]*self.num_layers,
                state_is_tuple=False
            )

        with tf.variable_scope("gru_y_cell"):
            gru_cell_y = tf.nn.rnn_cell.GRUCell(self.rnn_size)
            cells_y = tf.nn.rnn_cell.MultiRNNCell(
                [gru_cell_y]*self.num_layers,
                state_is_tuple=False
            )

        # Define GRU states for each pedestrian
        with tf.variable_scope("gru_states"):
            self.gru_states = tf.zeros(
                [self.args.max_num_obj, cells.state_size],
                name="gru_states"
            )
            self.enc_state_x = tf.split(
                0, self.args.max_num_obj, self.gru_states
            )
            self.h_of_x = tf.split(
                0, self.args.max_num_obj, self.gru_states
            )

        with tf.variable_scope("gru_states_y"):
            self.gru_states_y = tf.zeros(
                [self.args.max_num_obj, cells_y.state_size],
                name="gru_states_y"
            )
            self.enc_state_y = tf.split(
                0, self.args.max_num_obj, self.gru_states_y
            )
            self.h_of_y = tf.split(
                0, self.args.max_num_obj, self.gru_states_y
            )

        with tf.variable_scope("feature_pooling"):
            self.f_pool = \
                tf.zeros([self.args.max_num_obj, 7, self.seq_length, 2*self.out_channels])
            self.feature_pooling = \
                tf.split(0, self.args.max_num_obj, self.f_pool)
            self.feature_pooling = [tf.squeeze(_input, [0]) for _input in self.feature_pooling]

        # Define hidden output states for each pedestrian ??
        with tf.variable_scope("output_states"):
            self.output_states = \
                tf.split(0, self.args.max_num_obj, \
                    tf.zeros([
                        self.args.max_num_obj,
                        self.k_traj,
                        self.seq_length,
                        cells.output_size
                    ]))

        # List of tensors each of shape args.maxNumPedsx3 corresponding to
        # each frame in the sequence
        with tf.name_scope("frame_data_tensors"):
            frame_data = [tf.squeeze(input_, [0]) \
                for input_ in tf.split(0, self.args.max_num_obj, self.temporal_input)]

        with tf.name_scope("frame_data_tensors_y"):
            frame_data_y = [tf.squeeze(input_, [0]) \
                for input_ in tf.split(0, self.args.max_num_obj, self.temporal_input_y)]

        # Cost
        with tf.name_scope("cost_related_stuff"):
            self.cost = tf.constant(0.0, name="cost")
            self.counter = tf.constant(0.0, name="counter")
            self.increment = tf.constant(1.0, name="increment")

        # Containers to store output distribution parameters
        with tf.name_scope("distribution_parameters_stuff"):
            self.initial_output = \
                tf.split(0, self.args.max_num_obj, \
                    tf.zeros([self.args.max_num_obj, self.output_size]))

        # Tensor to represent non-existent ped
        with tf.name_scope("non_existent_obj_stuff"):
            nonexistent_obj = tf.constant(0.0, name="zero_obj")

        # with tf.name_scope("grid_frame_data_tensors"):
        #     # This would contain a list of tensors each of
        #     # shape OBJ x OBJ x (GS**2) encoding the mask
        #     grid_frame_data = [tf.squeeze(input_, [0]) \
        #         for input_ in tf.split(0, self.seq_length, self.grid_data)]

        for obj in xrange(0, self.args.max_num_obj):
            obj_id = temporal_ids[obj][0][0]
            pooling_list = []
            with tf.name_scope("extract_input_obj"):
                spatial_input_x = tf.split(
                    0, self.seq_length,
                    frame_data[obj]
                )
                spatial_input_y = tf.split(
                    0, 2*self.seq_length,
                    frame_data_y[obj]
                )

            with tf.variable_scope("encoding_operations_x", \
                        reuse=True if obj > 0 else None):
                _, self.enc_state_x[obj] = \
                    rnn.rnn(cells, spatial_input_x, dtype=dtypes.float32)

            with tf.variable_scope("encoding_operations_y", \
                        reuse=True if obj > 0 else None):
                _, self.enc_state_y[obj] = \
                    rnn.rnn(cells_y, spatial_input_y, dtype=dtypes.float32)

            with tf.name_scope("concatenate_embeddings"):
                # Concatenate the summaries c1 and c2
                complete_input = tf.concat(
                    1,
                    [self.enc_state_x[obj], self.enc_state_y[obj]]
                )

            # Convolutional VAE
            # fc layer
            with tf.variable_scope("fc_c"):
                vae_inputs = \
                    tf.nn.relu( \
                        tf.nn.xw_plus_b( \
                            complete_input, weights["w_hidden_enc1"], biases["b_hidden_enc1"]))

            # Encode our data into z and return the mean and covariance
            # =============================== Q(z|X) ======================================
            with tf.variable_scope("recognition", reuse=True if obj > 0 else None):
                z_mu = tf.nn.xw_plus_b(vae_inputs, weights["w_mu"], biases["b_mu"])
                z_logvar = tf.nn.xw_plus_b(vae_inputs, weights["w_sigma"], biases["b_sigma"])
                eps = [tf.random_normal(shape=tf.shape(z_mu)) for k in xrange(self.k_traj)]
                z_sample = [z_mu + tf.exp(z_logvar / 2) * eps[k] for k in xrange(self.k_traj)]

            # =============================== P(X|z) ======================================
            with tf.variable_scope("generation", reuse=True if obj > 0 else None):
                inputs = [
                    tf.concat(1, values=[z_sample[k], self.enc_state_y[obj]])
                    for k in xrange(self.k_traj)]
                hidden = [
                    tf.nn.relu(
                        tf.nn.xw_plus_b(
                            inputs[k], weights["w_generation_1"], biases["b_generation_1"]))
                    for k in xrange(self.k_traj)]
                logits = [
                    tf.nn.xw_plus_b(
                        hidden[k],
                        weights["w_generation_2"],
                        biases["b_generation_2"])
                    for k in xrange(self.k_traj)]
                # prob = tf.nn.sigmoid(logits)
                x_samples = logits

            # fc layer
            with tf.variable_scope("fc_softmax"):
                multipl = [
                    tf.add(
                        tf.matmul(
                            x_samples[k], weights["w_post_vae"]),
                        biases["b_post_vae"])
                    for k in xrange(self.k_traj)]
                # multipl = tf.nn.relu(multipl)
                multipl = tf.nn.softmax(multipl)

            # Take a note of H_x for the ranking and refinement module
            # TODO: is it deepcopied now?
            with tf.variable_scope("h_of_x", reuse=True if obj > 0 else None):
                # h_of_x = tf.Variable(self.enc_state_x.initialized_value())
                self.h_of_x = tf.multiply(self.enc_state_x, 1)
                self.h_of_x = tf.squeeze(tf.split(0, self.max_num_obj, self.h_of_x), [1])

            output = []

            loop_function = None
            with tf.variable_scope("rnn_decoder"):
                state = self.enc_state_x[obj]
                outputs = [[] for k in xrange(self.k_traj)]
                prev = None
                decoder_inputs = [
                    tf.pad(tf.mul(multipl[k], self.enc_state_x[obj]), [[0, 39], [0, 0]], "CONSTANT")
                    for k in xrange(self.k_traj)]
                for k, inp in enumerate(decoder_inputs):
                    inp = tf.split(0, 2*self.seq_length, inp)
                    for t_step, location in enumerate(inp):
                        if k > 0 or t_step > 0:
                            tf.get_variable_scope().reuse_variables()
                        output, state = cells(location, state)
                        outputs[k].append(output)
                        if loop_function is not None:
                            prev = output

            #TODO: Create a regression tensor and add it to the output_states[obj]
            #TODO: The two for-loops can go to a function
            #TODO: Fix it with IRL
            gamma_velocity = []
            for prediction_k in xrange(len(self.output_states[obj])):
                pooling_list.append([])
                gamma_velocity.append([])
                for step_t in xrange(len(self.output_states[obj][prediction_k])):
                    current_step_frame = self.output_states[obj][prediction_k][step_t]
                    pooling_list[prediction_k].append(
                        tf.concat(
                            0, [tf.multiply
                                (
                                    current_step_frame[0],
                                    self.temporal[obj][:100]
                                ),
                                tf.multiply
                                (
                                    current_step_frame[1],
                                    self.temporal[obj][100:]
                                )]
                            ))
                    gamma_velocity[prediction_k].append(
                        tf.nn.relu( \
                            tf.nn.xw_plus_b( \
                                 tf.reshape(current_step_frame, [1, 2]),
                                 weights["w_velocity_gamma"],
                                 biases["b_velocity_gamma"]))
                    )

            self.feature_pooling[obj] = tf.pack(pooling_list)
            # fc layer
            with tf.variable_scope("fc_relu"):
                gamma = tf.pack(gamma_velocity)

            social_tensor = self.getSocialTensor(current_grid_frame_data, self.output_states)
            # TODO: check if KLD loss actually has reconstruction loss in it
            # TODO: make sure that the CVAE implementation is truly from the same paper
            # TODO: Figure out how/if necessary to divide by K the reconstr_loss
            # TODO: The reconstr loss does not sample from a distribution anymore but instead
            #       chooses the most probable trajectory from the IOC (verify)
            with tf.name_scope("calculate_loss"):
                # Calculate loss for the current ped
                # reconstr_loss = \
                #     self.get_reconstr_loss(o_mux, o_muy, o_sx, o_sy, o_corr, x_data, y_data)
                # kld_loss = self.kld_loss(
                #     vae_inputs,
                #     x_reconstr_mean,
                #     z_log_sigma_sq,
                #     z_mean
                # )
                # E[log P(X|z)]
                recon_loss = tf.reduce_sum(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, targets=X), 1)
                # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
                kl_loss = 0.5 * tf.reduce_sum(tf.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1)
                # VAE loss
                vae_loss = tf.reduce_mean(recon_loss + kl_loss)
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
        self.final_states = tf.concat(0, self.enc_state_x)

        # Get the final distribution parameters
        self.final_output = self.initial_output

        # Compute gradients
        self.gradients = tf.gradients(self.cost, tvars)

        # Clip the gradients
        grads, _ = tf.clip_by_global_norm(self.gradients, self.args.grad_clip)

        # Define the optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)

        # self.loss_summary = tf.scalar_summary("loss", loss)
        # self.cost_summary = tf.scalar_summary("cost", self.cost)
        # self.summaries = tf.merge_all_summaries()
        # self.summary_writer = tf.train.SummaryWriter( \
        #     "logs/" + self.get_name() + self.get_formatted_datetime(), sess.graph)

        # The train operator
        # train_op = optimizer.apply_gradients(zip(grads, tvars))

    def get_name(self):
        '''formated name'''
        return "cvae_input_%dx%d_latent%d_edim%d_ddim%d" % (self.input_shape[0],
                                                            self.input_shape[
                                                                1],
                                                            self.latent_size,
                                                            self.args.e_dim,
                                                            self.args.d_dim)

    def get_formatted_datetime(self):
        '''formated datetime'''
        return str(datetime.datetime.now()).replace(" ", "_") \
                                            .replace("-", "_") \
                                            .replace(":", "_")

    def define_weights(self):
        ''' Define Model's weights'''
        # Weights adn Biases for hidden layer and output layer
        # TODO:Make sure you learn the dimensionalities!!!!!
        weights, biases = {}, {}
        with tf.variable_scope("temporal_weights"):
            # This is the filter window
            weights["temporal_w"] = tf.Variable(tf.truncated_normal( \
                [self.filter_width, self.in_channels, self.out_channels],
                stddev=0.1))
            biases["temporal_b"] = tf.Variable(tf.random_normal( \
                [self.out_channels]))

        with tf.variable_scope("temporal_weights_y"):
            # This is the filter window
            weights["temporal_w_y"] = tf.Variable(tf.truncated_normal( \
                [1, self.in_channels, self.out_channels],
                stddev=0.1))
            biases["temporal_b_y"] = tf.Variable(tf.random_normal( \
                [self.out_channels]))

        with tf.variable_scope("hidden_enc_weights"):
            weights["w_hidden_enc1"] = tf.Variable(tf.random_normal( \
                [2*self.rnn_size, self.vae_input_size]))
            biases["b_hidden_enc1"] = tf.Variable(tf.random_normal( \
                [self.vae_input_size]))

        with tf.variable_scope("mu_weights"):
            weights["w_mu"] = tf.Variable(tf.random_normal( \
                [self.vae_input_size, self.z_dim]))
            biases["b_mu"] = tf.Variable(tf.zeros( \
                [self.z_dim]))

        with tf.variable_scope("generation_weights_1"):
            weights["w_generation_1"] = tf.Variable(tf.random_normal( \
                [self.z_dim + self.rnn_size, self.vae_input_size]))
            biases["b_generation_1"] = tf.Variable(tf.zeros( \
                [self.vae_input_size]))

        with tf.variable_scope("generation_weights_2"):
            weights["w_generation_2"] = tf.Variable(tf.random_normal( \
                [self.vae_input_size, self.rnn_size]))
            biases["b_generation_2"] = tf.Variable(tf.zeros( \
                [self.rnn_size]))

        with tf.variable_scope("sigma_weights"):
            weights["w_sigma"] = tf.Variable(tf.random_normal( \
                [self.vae_input_size, self.z_dim]))
            biases["b_sigma"] = tf.Variable(tf.zeros( \
                [self.z_dim]))

        with tf.variable_scope("post_vae_weights"):
            weights["w_post_vae"] = tf.Variable(tf.random_normal( \
                [self.vae_input_size, self.rnn_size]))
            biases["b_post_vae"] = tf.Variable(tf.zeros( \
                [self.rnn_size]))

        with tf.variable_scope("velocity_gamma_weights"):
            weights["w_velocity_gamma"] = tf.Variable(tf.random_normal( \
                [2, 2]))
            biases["b_velocity_gamma"] = tf.Variable(tf.random_normal( \
                [2]))

        # with tf.variable_scope("output_weights"):
        #     weights["output_w"] = tf.Variable(tf.random_normal( \
        #         [self.rnn_size, self.output_size]))
        #     biases["output_b"] = tf.Variable(tf.random_normal( \
        #         [self.output_size]))

        return weights, biases

    def tf_2d_normal(self, x_val, y_val, mux, muy, sx_val, sy_val, rho):
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
        normx = tf.sub(x_val, mux)
        normy = tf.sub(y_val, muy)
        # Calculate sx_val*sy_val
        sxsy = tf.mul(sx_val, sy_val)
        # Calculate the exponential factor
        z_val = tf.square(tf.div(normx, sx_val)) + tf.square(tf.div(normy, sy_val)) \
            - 2*tf.div(tf.mul(rho, tf.mul(normx, normy)), sxsy)
        neg_rho = 1 - tf.square(rho)
        # Numerator
        result = tf.exp(tf.div(-z_val, 2*neg_rho))
        # Normalization constant
        denom = 2 * np.pi * tf.mul(sxsy, tf.sqrt(neg_rho))
        # Final PDF calculation
        result = tf.div(result, denom)
        return result

    def get_social_tensor(self, grid_frame_data, output_states):
        '''
        Modified from: https://github.com/vvanirudh/social-lstm-tf
        Computes the social tensor for all the maxNumPeds in the frame
        params:
        grid_frame_data : A tensor of shape MNP x MNP x (GS**2)
        output_states : A list of tensors each of shape 1 x RNN_size of length MNP
        '''
        # Create a zero tensor of shape MNP x (GS**2) x RNN_size
        social_tensor = tf.zeros([self.args.maxNumPeds, self.grid_size*self.grid_size, self.rnn_size], name="social_tensor")
        # Create a list of zero tensors each of shape 1 x (GS**2) x RNN_size of length MNP
        social_tensor = tf.split(0, self.args.maxNumPeds, social_tensor)
        # Concatenate list of hidden states to form a tensor of shape MNP x RNN_size
        hidden_states = tf.concat(0, output_states)
        # Split the grid_frame_data into grid_data for each pedestrians
        # Consists of a list of tensors each of shape 1 x MNP x (GS**2) of length MNP
        grid_frame_ped_data = tf.split(0, self.args.maxNumPeds, grid_frame_data)
        # Squeeze tensors to form MNP x (GS**2) matrices
        grid_frame_ped_data = [tf.squeeze(input_, [0]) for input_ in grid_frame_ped_data]

        # For each pedestrian
        for ped in range(self.args.maxNumPeds):
            # Compute social tensor for the current pedestrian
            with tf.name_scope("tensor_calculation"):
                social_tensor_ped = tf.matmul(tf.transpose(grid_frame_ped_data[ped]), hidden_states)
                social_tensor[ped] = tf.reshape(social_tensor_ped, [1, self.grid_size*self.grid_size, self.rnn_size])

        # Concatenate the social tensor from a list to a tensor of shape MNP x (GS**2) x RNN_size
        social_tensor = tf.concat(0, social_tensor)
        # Reshape the tensor to match the dimensions MNP x (GS**2 * RNN_size)
        social_tensor = tf.reshape(social_tensor, [self.args.maxNumPeds, self.grid_size*self.grid_size*self.rnn_size])
        return social_tensor

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
        result0 = self.tf_2d_normal(x_data, y_data, z_mux, z_muy, z_sx, z_sy, z_corr)

        # For numerical stability purposes
        epsilon = 1e-20

        # Apply the log operation
        result1 = -tf.log(tf.maximum(result0, epsilon))  # Numerical stability

        # Sum up all log probabilities for each data point
        return tf.reduce_sum(result1)

    def get_coef(self, output):
        '''eq 20 -> 22 of Graves (2013)'''

        z_val = output
        # Split the output into 5 parts corresponding to means, std devs and corr
        z_mux, z_muy, z_sx, z_sy, z_corr = tf.split(1, 5, z_val)

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
        # kld_loss = tf.reduce_mean(reconstr_loss + latent_loss)   # average over batch
        kld_loss = tf.reduce_mean(latent_loss)   # average over batch

        return kld_loss

    def sample_gaussian_2d(self, mux, muy, sx_val, sy_val, rho):
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
        cov = [[sx_val*sx_val, rho*sx_val*sy_val], [rho*sx_val*sy_val, sy_val*sy_val]]
        # Sample a point from the multivariate normal distribution
        x_val = np.random.multivariate_normal(mean, cov, 1)
        return x_val[0][0], x_val[0][1]

    def sample(self, sess, traj, grid, dimensions, true_traj, num=10):
        '''
        Sampling method
        traj is a sequence of frames (of length obs_length)
        so traj shape is (obs_length x maxNumPeds x 3)
        grid is a tensor of shape obs_length x maxNumPeds x maxNumPeds x (gs**2)
        states = sess.run(self.gru_states)
         '''
        # print "Fitting"
        # For each frame in the sequence
        for index, frame in enumerate(traj[:-1]):
            data = np.reshape(frame, (1, self.args.max_num_obj, 3))
            target_data = np.reshape(traj[index+1], (1, self.args.max_num_obj, 3))

            feed = {
                self.input_data: data,
                self.gru_states: states,
                self.target_data: target_data
            }
            [states, cost] = sess.run([self.final_states, self.cost], feed)
            # print cost

        ret = traj

        last_frame = traj[-1]

        prev_data = np.reshape(last_frame, (1, self.args.max_num_obj, 3))

        prev_target_data = np.reshape(true_traj[traj.shape[0]], (1, self.args.max_num_obj, 3))
        # Prediction
        for t_step in range(num):
            print "**** NEW PREDICTION TIME STEP", t_step, "****"
            sys.stdout.flush()
            feed = {
                self.input_data: prev_data,
                self.gru_states: states,
                self.target_data: prev_target_data
            }
            [output, states, cost] = sess.run(
                [self.final_output, self.final_states, self.cost], feed)
            print "Cost", cost
            sys.stdout.flush()
            # Output is a list of lists where the inner lists contain matrices of shape 1x5.
            # The outer list contains only one element (since seq_length=1) and the inner list
            # contains maxNumPeds elements
            # output = output[0]
            newpos = np.zeros((1, self.args.max_num_obj, 3))
            for objindex, objoutput in enumerate(output):
                [o_mux, o_muy, o_sx, o_sy, o_corr] = np.split(objoutput[0], 5, 0)
                mux, muy, sx_val, sy_val, corr = \
                    o_mux[0], o_muy[0], np.exp(o_sx[0]), np.exp(o_sy[0]), np.tanh(o_corr[0])

                next_x, next_y = self.sample_gaussian_2d(mux, muy, sx_val, sy_val, corr)
                if next_x > 1.0:
                    next_x = 1.0
                if next_y > 1.0:
                    next_y = 1.0

                if prev_data[0, objindex, 0] != 0:
                    print "Pedestrian ID", prev_data[0, objindex, 0]
                    print "Predicted parameters", mux, muy, sx_val, sy_val, corr
                    print "New Position", next_x, next_y
                    print "Target Position", prev_target_data[0, objindex, 1], \
                        prev_target_data[0, objindex, 2]
                    print
                    sys.stdout.flush()

                newpos[0, objindex, :] = [prev_data[0, objindex, 0], next_x, next_y]
            ret = np.vstack((ret, newpos))
            prev_data = newpos
            if t_step != num - 1:
                prev_target_data = \
                    np.reshape(true_traj[traj.shape[0] + t_step + 1], (1, self.args.max_num_obj, 3))

        # The returned ret is of shape (obs_length+pred_length) x maxNumPeds x 3
        return ret
