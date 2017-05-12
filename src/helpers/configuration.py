from data.input_set import InputSet
from data.point import Point


class Configuration(object):
    def __init__(self):
        self.PATH = "/home/alex/Documents/Project/data/FRAME_DATABASES/1_TILE"
        self.PATH_LABELS = "/home/alex/Documents/Project/LSTM_labels"
        self.DATA_SET_PATH = "/home/alex/Documents/Project/LSTM_data_sets"

        self.TILE_1_NUMBER_OF_EXAMPLES = 3618

        self.TILE_1_A = InputSet(1, 465, Point(1.11, 5.68), Point(1.11, 7.69))  # y change
        self.TILE_1_B = InputSet(466, 1090, Point(1.11, 7.69), Point(3.41, 7.69))  # x change
        self.TILE_1_C = InputSet(1091, 2330, Point(3.41, 7.69), Point(3.41, 0.54))  # y change
        self.TILE_1_D = InputSet(2331, 2700, Point(3.41, 0.54), Point(1.11, 0.54))  # x change
        self.TILE_1_E = InputSet(2701, 3618, Point(1.11, 0.54), Point(1.11, 5.68))  # y change

        class Size(object):
            def __init__(self, width, height, channels):
                self.WIDTH = width
                self.HEIGHT = height
                self.CHANNELS = channels
                pass

        self.IMAGE_SIZE = Size(268, 32, 3)

        self.VALIDATION_PERC = 0.0
        self.TESTING_PERC = 0.0
        self.TRAINING_PERC = 1 - self.VALIDATION_PERC - self.TESTING_PERC

        self.EPOCHS = 100
        self.LOG_PERIOD = 10  # steps
        self.SAVE_PERIOD = 500  # steps
        self.MIN_FRACTION_OF_EXAMPLES_IN_QUEUE = 0.4
        self.NUM_PREPROCESSING_THREADS = 16

        pass