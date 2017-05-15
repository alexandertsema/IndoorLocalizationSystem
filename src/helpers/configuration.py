from data.input_set import InputSet
from data.point import Point


class Configuration(object):
    def __init__(self):
        """
        path params
        """
        self.PATH = "/home/alex/Documents/Project/data/FRAME_DATABASES/1_TILE"
        self.PATH_LABELS = "/home/alex/Documents/Project/LSTM_labels"
        self.DATA_SET_PATH = "/home/alex/Documents/Project/LSTM_data_sets"
        self.OUTPUT_PATH = '/home/alex/PycharmProjects/IndoorLocalizationSystem/runs/out_'

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

        """
        modes
        """

        class Mode(object):
            def __init__(self):
                self.TRAINING = 'training'
                self.VALIDATION = 'validation'
                self.TESTING = 'testing'
                pass

        self.MODE = Mode()

        """
        model params
        """
        self.LEAKY_RELU_ALPHA = 0.1
        self.LSTM_HIDDEN_UNITS = 512
        self.LSTM_LAYERS = 2

        """
        training params
        """

        self.BATCH_SIZE = 16    # SEQUENCE_LENGTH
        self.EPOCHS = 100000
        self.LOG_PERIOD = 10  # steps
        self.SAVE_PERIOD = 10000  # steps
        self.MIN_FRACTION_OF_EXAMPLES_IN_QUEUE = 0.4
        self.NUM_PREPROCESSING_THREADS = 16
        self.NUM_EPOCHS_PER_DECAY = 1000  # Epochs after which learning rate decays.
        self.INITIAL_LEARNING_RATE = 0.001
        self.LEARNING_RATE_DECAY_FACTOR = 0.1
        self.TARGET_LOSS = 0.0000001

        """
        evaluation params
        """

        self.VALIDATION_PERIOD = 100  # steps
        self.TESTING_PERIOD = 500  # steps

        """
        testing params
        """

        self.TESTING_BATCH_SIZE = 1
        self.TESTING_EPOCHS = 1

        self.VALIDATION_PERC = 0.0
        self.TESTING_PERC = 0.0
        self.TRAINING_PERC = 1 - self.VALIDATION_PERC - self.TESTING_PERC

        pass
