import csv
import os

import tensorflow as tf
# from tensorflow.python.training.basic_session_run_hooks import CheckpointSaverListener
from tensorflow.python.training.session_run_hook import SessionRunHook


def format_rule(rule, relations):
    text = []
    for i, w in enumerate(rule):
        if i > 0:
            e0 = chr(ord('A') + i - 1)
        else:
            e0 = 'S'
        if i < rule.shape[0] - 1:
            e1 = chr(ord('A') + i)
        else:
            e1 = 'T'

        if w < len(relations):
            text.append("{}({}, {})".format(relations[w], e0, e1))
        else:
            text.append("{}({}, {})".format(relations[w - len(relations)], e1, e0))
    return " ^ ".join(text)


def format_query(query, relations):
    return "{}(S,T)".format(relations[query])


class PluginWriteRules(SessionRunHook):
    """Listener that evaluates and exports a model after creating a checkpoint.
    The `EvalAndExportListener` waits for the associated `CheckpointSaverHook`
    to save a checkpoint. It then uses the provided `eval_fn` and `export_fn` to
    first evaluate the model using the newly-created checkpoint, and then export
    the model according to the `export_strategies` provided in the `Experiment`.
    This listener is experimental and may be changed or removed in the future.
    """

    def __init__(self, estimator, input_fn, input_hook, relations, rule_depth):
        """Initializes an `EvalAndExportListener`.
        Args:
          eval_fn: function which evaluates the model with the following signature:
            `(name, checkpoint_path) -> eval_result`
          export_fn: function which exports the model according to a set of export
            strategies. Has the following signature:
            `(eval_result, checkpoint_path) -> export_results`
          model_dir: directory which contains estimator parameters and checkpoints.
        """
        self.estimator = estimator
        self.input_fn = input_fn
        self.input_hook = input_hook
        self.relations = relations
        self.rule_depth = rule_depth

    def end(self, session):
        """Evaluates and exports the model after a checkpoint is created."""
        # Load and cache the path of the most recent checkpoint to avoid duplicate
        # searches on GCS.
        # latest_path = saver.latest_checkpoint(self.model_dir)
        # global_step_value = session._sess.run()
        feed_dict = self.input_hook.build_feed_dict(session.graph)

        global_step_tensor = tf.train.get_global_step(session.graph)
        walk_tensors = tf.get_collection("WALKS")
        walk_query_tensors = tf.get_collection("WALK_QUERIES")
        beta_tensors = tf.get_collection("BETAS")
        ret = session.run([global_step_tensor] + walk_tensors + walk_query_tensors + beta_tensors, feed_dict=feed_dict)
        # pull rets
        idx = 0
        global_step = ret[idx]
        idx += 1
        walks = [ret[i] for i in range(idx, idx + self.rule_depth)]
        idx += self.rule_depth
        walk_queries = [ret[i] for i in range(idx, idx + self.rule_depth)]
        idx += self.rule_depth
        betas = [ret[i] for i in range(idx, idx + self.rule_depth)]
        idx += self.rule_depth
        assert idx == len(ret)
        beta_dim = betas[0].shape[1]

        pred_path = os.path.join(self.estimator.model_dir, 'rules')
        os.makedirs(pred_path, exist_ok=True)
        csv_path = os.path.join(pred_path, 'rules-{:08d}.csv'.format(global_step))
        with open(csv_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Query', 'Rule'] + ['Beta {}'.format(i) for i in range(beta_dim)])
            for d in range(self.rule_depth):
                for i in range(walks[d].shape[0]):
                    q = format_query(query=walk_queries[d][i], relations=self.relations)
                    walk = walks[d][i, :]
                    text = format_rule(relations=self.relations, rule=walk)
                    bs = [betas[d][i, j] for j in range(beta_dim)]
                    writer.writerow([q, text] + bs)

    @property
    def eval_result(self):
        return None  # self._eval_result

    @property
    def export_results(self):
        return None  # self._export_results
