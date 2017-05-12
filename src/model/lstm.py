import tensorflow as tf


class Lstm:
    def __init__(self, config):
        self.config = config
        pass

    def inference(self, x, mode_name, lstm_size):
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)

        state = tf.zeros([self.config.BATCH_SIZE, lstm.state_size])

        output = None
        for _ in range(self.config.TILE_1_NUMBER_OF_EXAMPLES):
            output, state = lstm(x, state)

        return output