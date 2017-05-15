import tensorflow as tf


class Lstm:
    def __init__(self, config):
        self.config = config
        pass

    @staticmethod
    def cell(lstm_size):
        return tf.contrib.rnn.BasicLSTMCell(num_units=lstm_size, state_is_tuple=False)

    def lstm(self):
        return tf.contrib.rnn.MultiRNNCell([self.cell(self.config.LSTM_HIDDEN_UNITS) for _ in range(self.config.LSTM_LAYERS)])

    def inference(self, x, mode_name):
        return tf.contrib.rnn.static_rnn(self.lstm(), x, dtype=tf.float32)
