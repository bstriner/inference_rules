import tensorflow as tf

import inference_rules.trainer


def main(argv):
    inference_rules.trainer.main(argv)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.flags.DEFINE_string('model_dir', 'output/model/transe_walks_secondary/v3', 'Model directory')
    tf.flags.DEFINE_string('schedule', 'train_and_evaluate', 'Schedule')
    tf.flags.DEFINE_integer('batch_size', 32, 'Batch size')
    tf.flags.DEFINE_integer('save_checkpoints_secs', 60 * 20, 'save_checkpoints_secs')
    tf.flags.DEFINE_string(
        'hparams',
        (
                'enable_transe=True,enable_walks1=True,enable_walks2=True,enable_secondary=True,' +
                'units=256,negative_samples=20'
        ),
        'Hyperparameters')
    tf.app.run()
