import os

import tensorflow as tf

from src.data.abstract.data_set import DataSet as DataSet_Abstract
from src.data.converter import Converter
from src.data.inputs import Inputs


class DataSet(DataSet_Abstract):
    def __init__(self, config):
        self.config = config
        DataSet_Abstract.__init__(self)
        pass

    def get_data_sets(self, batch_size):

        self.training_set.x,  self.training_set.y = self.read_tf_records_file('training_set', batch_size)
        self.validation_set.x, self.validation_set.y = self.read_tf_records_file('validation_set', batch_size)
        self.testing_set.x, self.testing_set.y = self.read_tf_records_file('testing_set', batch_size)

        if self.training_set.x is None and self.validation_set.x is None \
                and self.training_set.y is None and self.validation_set.y is None \
                and self.testing_set.x is None and self.testing_set.y is None:
            print()
            print('Datasets do not exist, creating new ones...')

            inputs = Inputs(self.config)
            converter = Converter(self.config)

            training_inputs, validation_inputs, testing_inputs = inputs.read_files()

            if training_inputs.size is not 0:
                print('Converting training_set...')
                converter.convert_to_tf_records(training_inputs, 'training_set')
            if validation_inputs.size is not 0:
                print('Converting validation_set...')
                converter.convert_to_tf_records(validation_inputs, 'validation_set')
            if testing_inputs.size is not 0:
                print('Converting testing_set...')
                converter.convert_to_tf_records(testing_inputs, 'testing_set')
            self.get_data_sets(batch_size)

        if os.path.exists(os.path.join(self.config.DATA_SET_PATH, 'meta')):
            f = open(os.path.join(self.config.DATA_SET_PATH, 'meta'), "r")
            lines = f.readlines()
            for line in lines:
                if line.__contains__('training_set'):
                    self.training_set.size = int(line.split('-')[1])
                if line.__contains__('validation_set'):
                    self.validation_set.size = int(line.split('-')[1])
                if line.__contains__('testing_set'):
                    self.testing_set.size = int(line.split('-')[1])

        return self

    def read_tf_records_file(self, name, batch_size):
        filename = os.path.join(self.config.DATA_SET_PATH, name + '.tfrecords')

        if not os.path.exists(filename):
            return None, None

        with tf.name_scope('input'):
            print()
            print("Creating queue for {}...".format(name))
            filename_queue = tf.train.string_input_producer(
                [filename], num_epochs=self.config.EPOCHS)

            image, label_x, label_y = self.read_and_decode(filename_queue)

            print("Creating batches for {}...".format(name))
            images, sparse_labels = tf.train.batch(
                [image, [label_x, label_y]], batch_size=batch_size, num_threads=self.config.NUM_PREPROCESSING_THREADS,
                capacity=1000 + 3 * batch_size)

            return images, sparse_labels

    def read_and_decode(self, filename_queue):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image_raw': tf.FixedLenFeature([], tf.string),
                'label_x':     tf.FixedLenFeature([], tf.float32),
                'label_y':     tf.FixedLenFeature([], tf.float32)
            })

        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image.set_shape([self.config.IMAGE_SIZE.WIDTH * self.config.IMAGE_SIZE.HEIGHT * self.config.IMAGE_SIZE.CHANNELS])

        # TODO: Could apply distortions here.

        image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

        label_x = tf.cast(features['label_x'], tf.float32)
        label_y = tf.cast(features['label_y'], tf.float32)

        return image, label_x, label_y
