import sys
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from data.data_set import DataSet
from evaluation.evaluation import Evaluation
from helpers.configuration import Configuration
from helpers.sessions import Sessions
from model.clstmnn import Clstmnn
from visualization.visualization import Visualization


def test(session_name=None, is_visualize=False):
    if session_name is None:
        session_name = input("Session name: ")

    config = Configuration()  # general settings
    data_sets = DataSet(config)  # data sets retrieval
    model = Clstmnn(config)  # model builder
    evaluation = Evaluation(config)
    visualization = Visualization(config)

    with tf.Graph().as_default():
        data_set = data_sets.get_data_sets(config.TESTING_BATCH_SIZE)

        print('Building model...')
        predictions_testing = model.inference(x=data_set.testing_set.x, mode_name=config.MODE.TESTING, reuse_lstm=None)
        mse = evaluation.loss(predictions=predictions_testing, labels=data_set.testing_set.y, mode_name=config.MODE.TESTING)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        merged = tf.summary.merge_all()
        saver = tf.train.Saver()

        print('Starting session...')
        with tf.Session() as sess:
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            summary_writer = tf.summary.FileWriter(config.OUTPUT_PATH + session_name + '_tested', sess.graph)
            sessions_helper = Sessions(config=config, session=sess, saver=saver, session_name=session_name, summary_writer=summary_writer, coordinator=coord, threads=threads)

            sessions_helper.restore()

            print()
            summary = None
            start_time = time.time()
            mses = []
            actual_labels = []
            predicted_labels = []
            for epoch in range(config.TESTING_EPOCHS):
                for step in range(int(data_set.testing_set.size / config.TESTING_BATCH_SIZE)):
                    summary = sess.run(merged)

                    sys.stdout.write('\r>> Examples tested: {}/{}'.format(step, int(data_set.testing_set.size / config.TESTING_BATCH_SIZE)))
                    sys.stdout.flush()

                    example_image, actual_label, predicted_label, mse_result = sess.run([data_set.testing_set.x, data_set.testing_set.y, predictions_testing, mse])

                    mses.append(mse_result)
                    actual_labels.append(actual_label)
                    predicted_labels.append(predicted_label)

                    if is_visualize:
                        visualization.show_example(predicted_label, actual_label, example_image, mse_result)

            summary_writer.add_summary(summary, 1)

            print()
            print('testing completed in %s' % (time.time() - start_time))
            print('%s: MSE @ 1 = %.3f' % (datetime.now(), np.array(mses).mean()))

            visualization.display_on_map(actual_labels, predicted_labels)

            sessions_helper.end()



