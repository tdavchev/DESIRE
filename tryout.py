import numpy as np
#import desire.utils.data_loader as dl
import data_loader as dl ##changed
import matplotlib.pyplot as plt
import tensorflow as tf
import time

# execfile("utils/data_loader.py")

seq_length = 6
batch_size = 1
max_num_obj = 6

data_loader = dl.DataLoader(max_num_obj, seq_length, max_num_obj, 5)
def get_coef(output):
    # eq 20 -> 22 of Graves (2013)

    z = output
    # Split the output into 5 parts corresponding to means, std devs and corr
    z_mux, z_muy, z_sx, z_sy, z_corr = tf.split(z, 5, 1)

    # The output must be exponentiated for the std devsd
    z_sx = tf.exp(z_sx)
    z_sy = tf.exp(z_sy)
    # Tanh applied to keep it in the range [-1, 1]
    z_corr = tf.tanh(z_corr)

    return [z_mux, z_muy, z_sx, z_sy, z_corr]

def get_lossfunc(z_mux, z_muy, z_sx, z_sy, z_corr, x_data, y_data):
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

    # Calculate the PDF of the data w.r.t to the distribution
    result0 = tf_2d_normal(x_data, y_data, z_mux, z_muy, z_sx, z_sy, z_corr)
    # For numerical stability purposes
    epsilon = 1e-20

    # Apply the log operation
    result1 = -tf.log(tf.maximum(result0, epsilon))  # Numerical stability

    # Sum up all log probabilities for each data point
    return tf.reduce_sum(result1)

def tf_2d_normal(x, y, mux, muy, sx, sy, rho):
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
    normx = tf.subtract(x, mux)
    normy = tf.subtract(y, muy)
    # Calculate sx*sy
    sxsy = tf.multiply(sx, sy)
    # Calculate the exponential factor
    z = tf.square(tf.div(normx, sx)) + tf.square(tf.div(normy, sy)) - 2*tf.div(tf.multiply(rho, tf.multiply(normx, normy)), sxsy)
    negRho = 1 - tf.square(rho)
    # Numerator
    result = tf.exp(tf.div(-z, 2*negRho))
    # Normalization constant
    denom = 2 * np.pi * tf.multiply(sxsy, tf.sqrt(negRho))
    # Final PDF calculation
    result = tf.div(result, denom)
    return result

# Size of the encoding layer (the hidden layer)
encoding_dim = 8 # feel free to change this value

input_size = 3

grad_clip = 10.0
learning_rate = 0.005

# Input and target placeholders
inputs = tf.placeholder(tf.float32, (seq_length, max_num_obj, input_size), name="inputs")

nonexistent_ped = tf.constant(0.0, name="zero_ped")
# inputs = tf.reshape(inputs_, shape=[-1, input_size])
targets = tf.placeholder(tf.float32, (seq_length, max_num_obj, input_size), name="targets")
# targets = tf.reshape(targets_, shape=[-1, input_size])
lr = tf.Variable(learning_rate, trainable=False, name="learning_rate")
frame_target_data = [tf.squeeze(target_, [0]) for target_ in tf.split(targets, seq_length, 0)]

cost = tf.constant(0.0, name="cost")
counter = tf.constant(0.0, name="counter")
increment = tf.constant(1.0, name="increment")

# frame_data = tf.split(0, args.seq_length, self.input_data, name="frame_data")
frame_data = [tf.squeeze(input_, [0]) for input_ in tf.split(inputs, seq_length, 0)]

for seq, frame in enumerate(frame_data):
# Output of hidden layer, single fully connected layer here with ReLU activation
    current_frame = frame
    print("seq {}", seq)
    for ped in range(max_num_obj):
        print("ped: {}", ped)
        pedID = current_frame[ped, 0]
        spat_input = tf.slice(current_frame, [ped, 1], [1, 2])
        encoded = tf.layers.dense(spat_input, 5, activation=tf.nn.relu)

#         # Output layer logits, fully connected layer with no activation
#         logits = tf.layers.dense(encoded, input_size, activation=None)
#         # Sigmoid output from logits
#         decoded = tf.sigmoid(logits, name = "decoded")
        [x_data, y_data] = tf.split(tf.slice(frame_target_data[seq], [ped, 1], [1, 2]), 2, 1)
        target_pedID = frame_target_data[seq][ped, 0]
        [o_mux, o_muy, o_sx, o_sy, o_corr] = get_coef(encoded)
        lossfunc = get_lossfunc(o_mux, o_muy, o_sx, o_sy, o_corr, x_data, y_data)
        # If it is a non-existent ped, it should not contribute to cost
        # If the ped doesn't exist in the next frame, he/she should not contribute to cost as well
        cost = tf.where(tf.logical_or(tf.equal(pedID, nonexistent_ped), tf.equal(target_pedID, nonexistent_ped)), cost, tf.add(cost, lossfunc))
        counter = tf.where(tf.logical_or(tf.equal(pedID, nonexistent_ped), tf.equal(target_pedID, nonexistent_ped)), counter, tf.add(counter, increment))

cost = tf.div(cost, counter)

tvars = tf.trainable_variables()
gradients = tf.gradients(cost, tvars)

grads, _ = tf.clip_by_global_norm(gradients, grad_clip)
# Adam optimizer
# Define the optimizer
optimizer = tf.train.RMSPropOptimizer(lr)

# The train operator
train_op = optimizer.apply_gradients(zip(grads, tvars))


epochs = 20
batch_size = 10
loss = 0
# Initialize a TensorFlow session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
        for ii in range(data_loader.num_batches):
            start = time.time()
            x, y, d = data_loader.next_batch()
            loss_batch = 0
            for batch in range(batch_size):
                xval = x[batch]
    #             xval = np.swapaxes(xval, 0, 1)
    #             print(xval.shape)
                feed = {inputs: xval, targets: xval}
                train_loss, _ = sess.run([cost, train_op], feed)

                loss_batch += train_loss
            end = time.time()
            loss_batch = loss_batch / batch_size
            loss += loss_batch
            print(
                "{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}"
                .format(
                    e * data_loader.num_batches + ii,
                    epochs * data_loader.num_batches,
                    e,
                    loss_batch, end - start))