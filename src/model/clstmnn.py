import tensorflow as tf
from tensorlayer import activation
from model.cnn import Cnn
from model.lstm import Lstm


class Clstmnn:
    def __init__(self, config):
        self.config = config
        self.cnn = Cnn(config)
        self.lstm = Lstm(config)
        pass

    def inference(self, x, mode_name='training'):
        with tf.name_scope('inputs'):
            x = tf.reshape(x, [-1, self.config.IMAGE_SIZE.HEIGHT, self.config.IMAGE_SIZE.WIDTH, self.config.IMAGE_SIZE.CHANNELS])

        tf.summary.image("/inputs", x, max_outputs=4)

        lstm = self.lstm.cell(self.config.LSTM_HIDDEN_UNITS)

        # state = tf.zeros([self.config.BATCH_SIZE, lstm.state_size])
        # logits = None

        # for i in range(self.config.NUMBER_STEPS):
        #   if i > 0:
        #     tf.get_variable_scope().reuse_variables()

        with tf.variable_scope('features'):
            features = self.cnn.inference(x=x, mode_name=mode_name)  # extract features with CNN

        lstm_inputs = tf.split(features, self.config.BATCH_SIZE, 0)

        with tf.variable_scope('seq'):
            # output_cell, state = lstm(features, state)  # get seq dependency with LSTM
            output, state = tf.contrib.rnn.static_rnn(lstm, lstm_inputs, dtype=tf.float32)

        glued_output = tf.concat(output, 0)

        with tf.variable_scope('dense'):
            dense = tf.layers.dense(inputs=glued_output, units=2, activation=activation.lrelu)  # predict X and Y with Fully connected layer

        logits = dense

        return logits
