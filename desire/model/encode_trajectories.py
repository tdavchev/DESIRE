
class TrajectoryEncoder(object):
    def __init__(self):
        # Size of the encoding layer (the hidden layer)
        self.encoding_dim = 8 # feel free to change this value

        self.input_size = 3

        self.grad_clip = 10.0
        self.learning_rate = 0.005
        self.build_model()

    def build_model(self):

        # Input and target placeholders
        self.inputs = tf.placeholder(tf.float32, (self.seq_length, self.max_num_obj, self.input_size), name="inputs")

        self.nonexistent_ped = tf.constant(0.0, name="zero_ped")
        # inputs = tf.reshape(inputs_, shape=[-1, input_size])
        self.targets = tf.placeholder(tf.float32, (self.seq_length, self.max_num_obj, self.input_size), name="targets")
        # targets = tf.reshape(targets_, shape=[-1, input_size])
        self.lr = tf.Variable(self.learning_rate, trainable=False, name="learning_rate")
        self.frame_target_data = [tf.squeeze(target_, [0]) for target_ in tf.split(self.targets, self.seq_length, 0)]

        self.cost = tf.constant(0.0, name="cost")
        self.counter = tf.constant(0.0, name="counter")
        self.increment = tf.constant(1.0, name="increment")

        # frame_data = tf.split(0, args.seq_length, self.input_data, name="frame_data")
        self.frame_data = [tf.squeeze(input_, [0]) for input_ in tf.split(self.inputs, self.seq_length, 0)]

        for seq, frame in enumerate(self.frame_data):
        # Output of hidden layer, single fully connected layer here with ReLU activation
            current_frame = frame
            for ped in range(max_num_obj):
                pedID = current_frame[ped, 0]
                spat_input = tf.slice(current_frame, [ped, 1], [1, 2])
                encoded = tf.layers.dense(spat_input, 5, activation=tf.nn.relu)

        #         # Output layer logits, fully connected layer with no activation
        #         logits = tf.layers.dense(encoded, input_size, activation=None)
        #         # Sigmoid output from logits
        #         decoded = tf.sigmoid(logits, name = "decoded")
                [x_data, y_data] = tf.split(tf.slice(self.frame_target_data[seq], [ped, 1], [1, 2]), 2, 1)
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