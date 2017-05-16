import tensorflow as tf


class Lstm:
    def __init__(self, config):
        self.config = config
        pass

    @staticmethod
    def cell(lstm_size, reuse):
        return tf.contrib.rnn.BasicLSTMCell(num_units=lstm_size, state_is_tuple=False, reuse=reuse)

    def lstm(self, reuse):
        return tf.contrib.rnn.MultiRNNCell([self.cell(self.config.LSTM_HIDDEN_UNITS, reuse) for _ in range(self.config.LSTM_LAYERS)])

    def inference(self, x, mode_name, reuse):
        return tf.contrib.rnn.static_rnn(self.lstm(reuse), x, dtype=tf.float32)
