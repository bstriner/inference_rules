import tensorflow as tf

import inference_rules.trainer


def main(argv):
    inference_rules.trainer.main(argv)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.flags.DEFINE_string('model_dir', 'output/model/v2', 'Model directory')
    tf.flags.DEFINE_string('schedule', 'train_and_evaluate', 'Schedule')
    tf.flags.DEFINE_integer('batch_size', 32, 'Batch size')
    tf.flags.DEFINE_string('hparams', '', 'Hyperparameters')
    tf.app.run()
