class DataSet(object):
    def __init__(self):
        class Label(object):
            def __init__(self):
                self.x = 0
                self.y = 0
                pass

        class TrainingSet(object):
            def __init__(self):
                self.x = None
                self.y = Label()
                self.size = 0

        class ValidationSet(object):
            def __init__(self):
                self.x = None
                self.y = Label()
                self.size = 0

        class TestingSet(object):
            def __init__(self):
                self.x = None
                self.y = Label()
                self.size = 0

        self.training_set = TrainingSet()
        self.validation_set = ValidationSet()
        self.testing_set = TestingSet()
        pass
