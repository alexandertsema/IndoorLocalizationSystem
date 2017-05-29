import os
import random
import sys
import numpy as np
from scipy import misc

from data.abstract.data_set import DataSet
from data.point import Point


class Inputs(DataSet):
    def __init__(self, config):
        DataSet.__init__(self)
        self.config = config
        self.training_set.x = []
        self.validation_set.x = []
        self.testing_set.x = []
        self.training_set.y = []
        self.validation_set.y = []
        self.testing_set.y = []
        pass

    def read_files(self):
        print('Reading data from', self.config.PATH)

        lines = []    # read labels
        if os.path.exists(os.path.join(self.config.PATH_LABELS)):
            f = open(os.path.join(self.config.PATH_LABELS), "r")
            lines = f.readlines()

        x = []
        y = []
        for line in lines:
            x.append(float(line.split(' ')[1]))
            y.append(float(line.split(' ')[2]))

        x = self._normalize(x)
        y = self._normalize(y)

        i = 0
        data_catalogs = os.listdir(self.config.PATH)
        data_catalogs.sort()
        for catalog in data_catalogs:
            examples = os.listdir(os.path.join(self.config.PATH, catalog))  # read images
            examples.sort(key=lambda s: int(s.split('.')[0]))
            for sample in examples:
                if not sample.__contains__(".jpg"):
                    continue

                if i < len(lines) * self.config.TRAINING_PERC:
                    data_set = self.training_set
                elif len(lines) * self.config.TRAINING_PERC < i < len(lines) * self.config.TRAINING_PERC + len(lines) * self.config.VALIDATION_PERC:
                    data_set = self.validation_set
                else:
                    data_set = self.testing_set

                data_set.x.append(self._raw_bytes(os.path.join(self.config.PATH, catalog, sample)))
                data_set.y.append(Point(x[i], y[i]))
                data_set.size += 1

                i += 1

                sys.stdout.write('\r>> Samples read: {}'.format(self.training_set.size + self.validation_set.size + self.testing_set.size))
                sys.stdout.flush()

        print()
        print('Data set is {} samples'.format(self.training_set.size + self.validation_set.size + self.testing_set.size))
        print('Training set is {} samples'.format(self.training_set.size))
        print('Validation set is {} samples'.format(self.validation_set.size))
        print('Testing set is {} samples'.format(self.testing_set.size))

        return self.training_set, self.validation_set, self.testing_set  # TODO: normalize Y to [0,MAX] (relu is [0,max] and data set's coordinate system is not negative, so... no need to normalize)

    @staticmethod
    def biased_random(prob_true=0.5):
        return random.random() < prob_true

    def _raw_bytes(self, file_name):
        raw_image = self._decoded_image(file_name)
        return self._preprocess_image(raw_image)

    @staticmethod
    def _decoded_image(file_name):
        return misc.imread(file_name)

    def _preprocess_image(self, raw_image):
        #  TODO could add distortions here
        return misc.imresize(raw_image, (self.config.IMAGE_SIZE.HEIGHT, self.config.IMAGE_SIZE.WIDTH, self.config.IMAGE_SIZE.CHANNELS), interp='bilinear', mode=None)

    def _normalize(self, vector):
        arr = np.array(vector)
        min = np.amin(arr)
        max = np.amax(arr)
        for i in range(len(arr)):
            arr[i] = (((arr[i]-min)*(self.config.NORM_RANGE[1] - self.config.NORM_RANGE[0]))/(max-min)) + self.config.NORM_RANGE[0]
        return arr

