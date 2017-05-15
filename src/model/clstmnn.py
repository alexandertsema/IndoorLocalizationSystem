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

    def inference(self, x, mode_name):
        with tf.name_scope('inputs'):
            x = tf.reshape(x, [-1, self.config.IMAGE_SIZE.HEIGHT, self.config.IMAGE_SIZE.WIDTH, self.config.IMAGE_SIZE.CHANNELS])

        tf.summary.image("/inputs", x, max_outputs=4)

        # lstm = self.lstm.cell(self.config.LSTM_HIDDEN_UNITS)

        with tf.variable_scope('features'):
            features = self.cnn.inference(x=x, mode_name=mode_name)  # extract features with CNN

        features_sequence = tf.split(features, self.config.BATCH_SIZE, 0)

        with tf.variable_scope('seq'):
            output, state = self.lstm.inference(x=features_sequence, mode_name=mode_name)  # get seq dependency with LSTM
            # output, state = tf.contrib.rnn.static_rnn(lstm, features_sequence, dtype=tf.float32)  # get seq dependency with LSTM

        with tf.variable_scope('dense'):
            dense = tf.layers.dense(inputs=output[-1], units=2, activation=activation.lrelu)  # predict X and Y with Fully connected layer

        logits = dense

        return logits
