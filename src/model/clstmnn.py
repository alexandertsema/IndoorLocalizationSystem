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

        features = self.cnn.inference(x=x, mode_name=mode_name)

        return self.lstm.inference(x=features, mode_name='training', lstm_size=200)
