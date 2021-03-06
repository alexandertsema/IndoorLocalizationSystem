import tensorflow as tf
from model.cnn import Cnn
from model.lstm import Lstm


class Clstmnn:
    def __init__(self, config):
        self.config = config
        self.cnn = Cnn(config)
        self.lstm = Lstm(config)
        pass

    def inference(self, x, mode_name, reuse_lstm):
        with tf.name_scope('inputs'):
            x = tf.reshape(x, [-1, self.config.IMAGE_SIZE.HEIGHT, self.config.IMAGE_SIZE.WIDTH, self.config.IMAGE_SIZE.CHANNELS])

        tf.summary.image("/inputs", x, max_outputs=4)

        with tf.variable_scope('features'):
            features = self.cnn.inference(x=x, mode_name=mode_name)  # extract features with CNN

        features_sequence = tf.split(features, self.config.BATCH_SIZE, 0)

        with tf.variable_scope('seq'):
            output, state = self.lstm.inference(x=features_sequence, mode_name=mode_name, reuse=reuse_lstm)  # get seq dependency with LSTM

        concat_output = tf.concat(output, 0)

        with tf.variable_scope('dense'):
            dense = tf.layers.dense(inputs=concat_output, units=2, activation=tf.sigmoid)  # predict X and Y with Fully connected layer

        logits = dense

        return logits
