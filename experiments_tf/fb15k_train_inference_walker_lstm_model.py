import tensorflow as tf

import inference_rules.tf.inference_walker_lstm_model


def main(argv):
    inference_rules.tf.inference_walker_lstm_model.main(argv)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.flags.DEFINE_string('model-dir', 'output/lstm/model-adam-2', 'Model directory')
    tf.flags.DEFINE_string('train-dir', 'output/FB15K/merged_walks', 'Schedule')
    tf.flags.DEFINE_string('schedule', 'train_and_evaluate', 'Schedule')
    tf.flags.DEFINE_boolean('sigmoid-dist', True, 'Schedule')
    tf.flags.DEFINE_string('hparams', '', 'Hyperparameters')
    tf.flags.DEFINE_boolean('augment', False, 'Augment')
    tf.app.run()
