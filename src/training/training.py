import tensorflow as tf


class Training:
    def __init__(self, config):
        self.config = config
        pass

    def train(self, loss, global_step, num_examples_per_epoch_for_train):
        num_batches_per_epoch = num_examples_per_epoch_for_train / self.config.BATCH_SIZE
        decay_steps = int(num_batches_per_epoch * self.config.NUM_EPOCHS_PER_DECAY)

        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(learning_rate=self.config.INITIAL_LEARNING_RATE,
                                        global_step=global_step,
                                        decay_steps=decay_steps,
                                        decay_rate=self.config.LEARNING_RATE_DECAY_FACTOR,
                                        staircase=True)

        tf.summary.scalar('learning_rate', lr)

        with tf.variable_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(lr)
            train_op = optimizer.minimize(loss=loss, global_step=global_step)

        return train_op
