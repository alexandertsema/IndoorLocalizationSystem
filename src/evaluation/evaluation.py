import tensorflow as tf


class Evaluation:
    def __init__(self, config):
        self.config = config
        pass

    def loss(self, predictions, labels, mode_name):  # Calculate the average cross entropy loss across the batch.
        if predictions.get_shape()[0].value is 1:   # if we don't use batch, but single example (which is given by LSTM)
            labels = tf.split(labels, self.config.BATCH_SIZE)[-1]

        with tf.variable_scope('loss_function_{}'.format(mode_name)):
            loss = tf.losses.mean_squared_error(labels=labels, predictions=predictions)

        tf.add_to_collection('loss_{}'.format(mode_name), loss)
        tf.summary.scalar('loss_{}'.format(mode_name), loss)

        return tf.add_n(tf.get_collection('loss_{}'.format(mode_name)), name='loss_{}'.format(mode_name))

    def accuracy(self, predictions, labels, mode_name):  # Calculate the average accuracy across the batch.
        with tf.variable_scope('correct_prediction_{}'.format(mode_name)):
            num_correct = self.correct_number(predictions, labels)
        with tf.variable_scope('accuracy_{}'.format(mode_name)):
            acc_percent = (num_correct / self.config.BATCH_SIZE) * 100

        tf.add_to_collection('accuracy_{}'.format(mode_name), acc_percent)
        tf.summary.scalar('accuracy_{}'.format(mode_name), acc_percent)

        return tf.add_n(tf.get_collection('accuracy_{}'.format(mode_name)), name='accuracy_{}'.format(mode_name))

    @staticmethod
    def correct_number(predictions, labels):
        casted = tf.nn.in_top_k(predictions, labels, 1)
        return tf.reduce_sum(casted)
