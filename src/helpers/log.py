from datetime import datetime


class Logger:
    def __init__(self, config):
        self.config = config
        pass

    def log(self, global_step, epoch, step, duration, loss, accuracy, mode):
        if mode == self.config.MODE.TRAINING:
            num_examples_per_step = self.config.BATCH_SIZE
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = float(duration)
            format_str = '%s: [Global_Step: %d Epoch: %d Step: %d] loss = %.4f, accuracy = %.4f (%.1f examples/sec; %.3f sec/batch)'
            print(format_str % (datetime.now(), global_step, epoch, step, loss, accuracy, examples_per_sec, sec_per_batch))
        if mode == self.config.MODE.VALIDATION:
            format_str = '%s: >> VALIDATION: [Global_Step: %d Epoch: %d Step: %d] loss = %.4f, accuracy = %.4f'
            print(format_str % (datetime.now(), global_step, epoch, step, loss, accuracy))
        if mode == self.config.MODE.TESTING:
            format_str = '%s: >> TESTING: [Global_Step: %d Epoch: %d Step: %d] loss = %.4f, accuracy = %.4f'
            print(format_str % (datetime.now(), global_step, epoch, step, loss, accuracy))
        pass
