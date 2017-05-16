from data.input_set import InputSet
from data.point import Point


class Configuration(object):
    def __init__(self):

        """
        path params
        """
        self.PATH = "/home/alex/Documents/Project/data/FRAME_DATABASES"
        self.PATH_LABELS = "/home/alex/Documents/Project/LSTM_labels/tiles"
        self.DATA_SET_PATH = "/home/alex/PycharmProjects/IndoorLocalizationSystem/data"
        self.OUTPUT_PATH = '/home/alex/PycharmProjects/IndoorLocalizationSystem/runs/out_'
        self.MAP_PATH = ''

        """
        dataset params
        """
        class TileType(object):
            def __init__(self):
                self.A = 'A'
                self.B = 'B'
                self.C = 'C'
                self.D = 'D'
                self.E = 'E'
                pass

        self.TILE_TYPE = TileType()
        self.TILE_1_NUMBER_OF_EXAMPLES = 3618
        self.TILE_1_A = InputSet(1, 465, Point(1.11, 5.68), Point(1.11, 7.69), self.TILE_TYPE.A)  # y change
        self.TILE_1_B = InputSet(466, 1090, Point(1.11, 7.69), Point(3.41, 7.69), self.TILE_TYPE.B)  # x change
        self.TILE_1_C = InputSet(1091, 2330, Point(3.41, 7.69), Point(3.41, 0.54), self.TILE_TYPE.C)  # y change
        self.TILE_1_D = InputSet(2331, 2700, Point(3.41, 0.54), Point(1.11, 0.54), self.TILE_TYPE.D)  # x change
        self.TILE_1_E = InputSet(2701, 3618, Point(1.11, 0.54), Point(1.11, 5.68), self.TILE_TYPE.E)  # y change

        self.TILE_2_NUMBER_OF_EXAMPLES = 3579
        self.TILE_2_A = InputSet(1, 510, Point(1.11, 5.68), Point(1.11, 7.69), self.TILE_TYPE.A)  # y change
        self.TILE_2_B = InputSet(511, 1150, Point(1.11, 7.69), Point(3.41, 7.69), self.TILE_TYPE.B)  # x change
        self.TILE_2_C = InputSet(1151, 2370, Point(3.41, 7.69), Point(3.41, 0.54), self.TILE_TYPE.C)  # y change
        self.TILE_2_D = InputSet(2371, 2760, Point(3.41, 0.54), Point(1.11, 0.54), self.TILE_TYPE.D)  # x change
        self.TILE_2_E = InputSet(2761, 3579, Point(1.11, 0.54), Point(1.11, 5.68), self.TILE_TYPE.E)  # y change

        self.TILE_3_NUMBER_OF_EXAMPLES = 3380
        self.TILE_3_A = InputSet(1, 430, Point(1.11, 5.68), Point(1.11, 7.69), self.TILE_TYPE.A)  # y change
        self.TILE_3_B = InputSet(431, 960, Point(1.11, 7.69), Point(3.41, 7.69), self.TILE_TYPE.B)  # x change
        self.TILE_3_C = InputSet(961, 2160, Point(3.41, 7.69), Point(3.41, 0.54), self.TILE_TYPE.C)  # y change
        self.TILE_3_D = InputSet(2161, 2540, Point(3.41, 0.54), Point(1.11, 0.54), self.TILE_TYPE.D)  # x change
        self.TILE_3_E = InputSet(2541, 3380, Point(1.11, 0.54), Point(1.11, 5.68), self.TILE_TYPE.E)  # y change

        self.TILE_4_NUMBER_OF_EXAMPLES = 3223
        self.TILE_4_A = InputSet(1, 410, Point(1.11, 5.68), Point(1.11, 7.69), self.TILE_TYPE.A)  # y change
        self.TILE_4_B = InputSet(411, 910, Point(1.11, 7.69), Point(3.41, 7.69), self.TILE_TYPE.B)  # x change
        self.TILE_4_C = InputSet(911, 2080, Point(3.41, 7.69), Point(3.41, 0.54), self.TILE_TYPE.C)  # y change
        self.TILE_4_D = InputSet(2081, 2430, Point(3.41, 0.54), Point(1.11, 0.54), self.TILE_TYPE.D)  # x change
        self.TILE_4_E = InputSet(2431, 3223, Point(1.11, 0.54), Point(1.11, 5.68), self.TILE_TYPE.E)  # y change

        self.TILE_5_NUMBER_OF_EXAMPLES = 3068
        self.TILE_5_A = InputSet(1, 400, Point(1.11, 5.68), Point(1.11, 7.69), self.TILE_TYPE.A)  # y change
        self.TILE_5_B = InputSet(401, 910, Point(1.11, 7.69), Point(3.41, 7.69), self.TILE_TYPE.B)  # x change
        self.TILE_5_C = InputSet(911, 1960, Point(3.41, 7.69), Point(3.41, 0.54), self.TILE_TYPE.C)  # y change
        self.TILE_5_D = InputSet(1961, 2320, Point(3.41, 0.54), Point(1.11, 0.54), self.TILE_TYPE.D)  # x change
        self.TILE_5_E = InputSet(2321, 3068, Point(1.11, 0.54), Point(1.11, 5.68), self.TILE_TYPE.E)  # y change

        self.TILE_6_NUMBER_OF_EXAMPLES = 2973
        self.TILE_6_A = InputSet(1, 410, Point(1.11, 5.68), Point(1.11, 7.69), self.TILE_TYPE.A)  # y change
        self.TILE_6_B = InputSet(411, 910, Point(1.11, 7.69), Point(3.41, 7.69), self.TILE_TYPE.B)  # x change
        self.TILE_6_C = InputSet(911, 1930, Point(3.41, 7.69), Point(3.41, 0.54), self.TILE_TYPE.C)  # y change
        self.TILE_6_D = InputSet(1931, 2270, Point(3.41, 0.54), Point(1.11, 0.54), self.TILE_TYPE.D)  # x change
        self.TILE_6_E = InputSet(2271, 2973, Point(1.11, 0.54), Point(1.11, 5.68), self.TILE_TYPE.E)  # y change

        self.TILE_7_NUMBER_OF_EXAMPLES = 2930
        self.TILE_7_A = InputSet(1, 370, Point(1.11, 5.68), Point(1.11, 7.69), self.TILE_TYPE.A)  # y change
        self.TILE_7_B = InputSet(371, 830, Point(1.11, 7.69), Point(3.41, 7.69), self.TILE_TYPE.B)  # x change
        self.TILE_7_C = InputSet(831, 1875, Point(3.41, 7.69), Point(3.41, 0.54), self.TILE_TYPE.C)  # y change
        self.TILE_7_D = InputSet(1876, 2200, Point(3.41, 0.54), Point(1.11, 0.54), self.TILE_TYPE.D)  # x change
        self.TILE_7_E = InputSet(2201, 2930, Point(1.11, 0.54), Point(1.11, 5.68), self.TILE_TYPE.E)  # y change

        """
        input images params
        """
        class Size(object):
            def __init__(self, width, height, channels):
                self.WIDTH = width
                self.HEIGHT = height
                self.CHANNELS = channels
                pass

        self.IMAGE_SIZE = Size(268, 32, 3)

        """
        modes params
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
        self.SAVE_PERIOD = 1000  # steps
        self.MIN_FRACTION_OF_EXAMPLES_IN_QUEUE = 0.4
        self.NUM_PREPROCESSING_THREADS = 16
        self.NUM_EPOCHS_PER_DECAY = 1000  # Epochs after which learning rate decays.
        self.INITIAL_LEARNING_RATE = 0.001
        self.LEARNING_RATE_DECAY_FACTOR = 0.1
        self.TARGET_LOSS = 0.0000001

        """
        testing params
        """

        self.TESTING_BATCH_SIZE = 16   # SEQUENCE_LENGTH
        self.TESTING_EPOCHS = 1

        self.VALIDATION_PERC = 0.2
        self.TESTING_PERC = 0.1
        self.TRAINING_PERC = 1 - self.VALIDATION_PERC - self.TESTING_PERC

        pass
