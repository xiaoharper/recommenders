# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import itertools
import numpy as np
import pandas as pd
import tensorflow as tf

MODEL_DIR = "model_checkpoints"


# TODO check params update
OPTIMIZERS = dict(
    adadelta=tf.keras.optimizers.Adadelta,
    adagrad=tf.keras.optimizers.Adagrad,
    adam=tf.keras.optimizers.Adam,
    # TODO: adamax
    ftrl=tf.keras.optimizers.Ftrl,
    # TODO: nadam
    rmsprop=tf.keras.optimizers.RMSprop,
    sgd=tf.keras.optimizers.SGD,
)


def pandas_input_fn_for_saved_model(df, feat_name_type):
    """Pandas input function for TensorFlow SavedModel.
    
    Args:
        df (pd.DataFrame): Data containing features.
        feat_name_type (dict): Feature name and type spec. E.g.
            `{'userID': int, 'itemID': int, 'rating': float}`
        
    Returns:
        func: Input function 
    
    """
    for feat_type in feat_name_type.values():
        assert feat_type in (int, float, list)

    def input_fn():
        examples = [None] * len(df)
        for i, sample in df.iterrows():
            ex = tf.train.Example()
            for feat_name, feat_type in feat_name_type.items():
                feat = ex.features.feature[feat_name]
                if feat_type == int:
                    feat.int64_list.value.extend([sample[feat_name]])
                elif feat_type == float:
                    feat.float_list.value.extend([sample[feat_name]])
                elif feat_type == list:
                    feat.float_list.value.extend(sample[feat_name])
            examples[i] = ex.SerializeToString()
        return {"inputs": tf.constant(examples)}

    return input_fn


def pandas_input_fn(
    df, y_col=None, batch_size=128, num_epochs=1, shuffle=False, seed=None, shuffle_buffer_size=1000
):
    """Pandas input function for TensorFlow high-level API Estimator.
    This function returns a `tf.data.Dataset` function.

    Args:
        df (pd.DataFrame): Data containing features.
        y_col (str): Label column name if df has it.
        batch_size (int): Batch size for the input function.
        num_epochs (int): Number of epochs to iterate over data. If None will run forever.
        shuffle (bool): If True, shuffles the data queue.
        seed (int): Random seed for shuffle.

    Returns:
        tf.data.Dataset function
    """

    X_df = df.copy()
    if y_col is not None:
        y = X_df.pop(y_col).values
        if isinstance(y[0], np.float64):
            y = y.astype(np.float32)
    else:
        y = None

    X = {}
    for col in X_df.columns:
        values = X_df[col].values
        if isinstance(values[0], (list, np.ndarray)):
            values = np.array([l for l in values], dtype=np.float32)
        elif isinstance(values[0], np.float64):
            values = values.astype(np.float32)
        X[col] = values

    return lambda: _dataset(
        x=X,
        y=y,
        batch_size=batch_size,
        num_epochs=num_epochs,
        shuffle=shuffle,
        seed=seed,
        shuffle_buffer_size=shuffle_buffer_size,
    )


def _dataset(x, y=None, batch_size=128, num_epochs=1, shuffle=False, seed=None, shuffle_buffer_size=1000):
    if y is None:
        dataset = tf.data.Dataset.from_tensor_slices(x)
    else:
        dataset = tf.data.Dataset.from_tensor_slices((x, y))

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=shuffle_buffer_size, seed=seed, reshuffle_each_iteration=True
        )
    elif seed is not None:
        import warnings

        warnings.warn("Seed was set but `shuffle=False`. Seed will be ignored.")

    return dataset.repeat(num_epochs).batch(batch_size)


def build_optimizer(name, lr=0.001, **kwargs):
    """Get an optimizer for TensorFlow high-level API Estimator.

    Args:
        name (str): Optimizer name. Note, to use 'Momentum', should specify
        lr (float): Learning rate
        kwargs: Optimizer arguments as key-value pairs

    Returns:
        tf.keras.optimizers.Optimizer
    """
    name = name.lower()

    try:
        optimizer_class = OPTIMIZERS[name]
    except KeyError:
        raise KeyError("Optimizer name should be one of: {}".format(list(OPTIMIZERS)))

    # Set parameters
    params = {}
    if name == "ftrl":
        params["l1_regularization_strength"] = kwargs.get(
            "l1_regularization_strength", 0.0
        )
        params["l2_regularization_strength"] = kwargs.get(
            "l2_regularization_strength", 0.0
        )
    elif name == "momentum" or name == "rmsprop":
        params["momentum"] = kwargs.get("momentum", 0.0)

    return optimizer_class(learning_rate=lr, **params)


def export_model(model, tf_feat_cols, base_dir):
    """Export TensorFlow estimator model for serving.
    
    Args:
        model (tf.estimator.Estimator): Model to export.
        tf_feat_cols (list(tf.feature_column)): Feature columns.
        base_dir (str): Base directory to export the model.
    
    Returns:
        str: Exported model path
    """
    serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
        tf.feature_column.make_parse_example_spec(tf_feat_cols)
    )

    exported_path = model.export_saved_model(
        base_dir, serving_input_fn
    )

    return exported_path.decode("utf-8")


def evaluation_log_hook(
    estimator,
    logger,
    true_df,
    y_col,
    eval_df,
    every_n_iter=10000,
    model_dir=None,
    batch_size=256,
    eval_fns=None,
    **eval_kwargs
):
    """Evaluation log hook for TensorFlow high-level API Estimator.
    
    .. note::

        TensorFlow Estimator model uses the last checkpoint weights for evaluation or prediction.
        In order to get the most up-to-date evaluation results while training,
        set model's `save_checkpoints_steps` to be equal or greater than hook's `every_n_iter`.

    Args:
        estimator (tf.estimator.Estimator): Model to evaluate.
        logger (Logger): Custom logger to log the results.
            E.g., define a subclass of Logger for AzureML logging.
        true_df (pd.DataFrame): Ground-truth data.
        y_col (str): Label column name in true_df
        eval_df (pd.DataFrame): Evaluation data without label column.
        every_n_iter (int): Evaluation frequency (steps).
        model_dir (str): Model directory to save the summaries to. If None, does not record.
        batch_size (int): Number of samples fed into the model at a time.
            Note, the batch size doesn't affect on evaluation results.
        eval_fns (iterable of functions): List of evaluation functions that have signature of
            (true_df, prediction_df, **eval_kwargs)->(float). If None, loss is calculated on true_df.
        **eval_kwargs: Evaluation function's keyword arguments.
            Note, prediction column name should be 'prediction'

    Returns:
        tf.train.SessionRunHook: Session run hook to evaluate the model while training.
    """

    return _TrainLogHook(
        estimator,
        logger,
        true_df,
        y_col,
        eval_df,
        every_n_iter,
        model_dir,
        batch_size,
        eval_fns,
        **eval_kwargs
    )


class _TrainLogHook(tf.estimator.SessionRunHook):
    def __init__(
        self,
        estimator,
        logger,
        true_df,
        y_col,
        eval_df,
        every_n_iter=10000,
        model_dir=None,
        batch_size=256,
        eval_fns=None,
        **eval_kwargs
    ):
        """Evaluation log hook class"""
        self.model = estimator
        self.logger = logger
        self.true_df = true_df
        self.y_col = y_col
        self.eval_df = eval_df
        self.every_n_iter = every_n_iter
        self.model_dir = model_dir
        self.batch_size = batch_size
        self.eval_fns = eval_fns
        self.eval_kwargs = eval_kwargs

        self.summary_writer = None
        self.global_step_tensor = None
        self.step = 0

    def begin(self):
        if self.model_dir is not None:
            self.summary_writer = tf.summary.FileWriterCache.get(self.model_dir)
            self.global_step_tensor = tf.train.get_or_create_global_step()
        else:
            self.step = 0

    def before_run(self, run_context):
        if self.global_step_tensor is not None:
            requests = {"global_step": self.global_step_tensor}
            return tf.train.SessionRunArgs(requests)
        else:
            return None

    def after_run(self, run_context, run_values):
        if self.global_step_tensor is not None:
            self.step = run_values.results["global_step"]
        else:
            self.step += 1

        if self.step % self.every_n_iter == 0:
            if self.eval_fns is None:
                result = self.model.evaluate(
                    input_fn=pandas_input_fn(
                        df=self.true_df, y_col=self.y_col, batch_size=self.batch_size
                    )
                )["average_loss"]
                self._log("validation_loss", result)
            else:
                predictions = list(
                    itertools.islice(
                        self.model.predict(
                            input_fn=pandas_input_fn(
                                df=self.eval_df, batch_size=self.batch_size
                            )
                        ),
                        len(self.eval_df),
                    )
                )
                prediction_df = self.eval_df.copy()
                prediction_df["prediction"] = [p["predictions"][0] for p in predictions]
                for fn in self.eval_fns:
                    result = fn(self.true_df, prediction_df, **self.eval_kwargs)
                    self._log(fn.__name__, result)

    def end(self, session):
        if self.summary_writer is not None:
            self.summary_writer.flush()

    def _log(self, tag, value):
        self.logger.log(tag, value)
        if self.summary_writer is not None:
            summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
            self.summary_writer.add_summary(summary, self.step)


class MetricsLogger:
    """Metrics logger"""

    def __init__(self):
        """Initializer"""
        self._log = {}

    def log(self, metric, value):
        """Log metrics. Each metric's log will be stored in the corresponding list.

        Args:
            metric (str): Metric name.
            value (float): Value.
        """
        if metric not in self._log:
            self._log[metric] = []
        self._log[metric].append(value)

    def get_log(self):
        """Getter
        
        Returns:
            dict: Log metrics.
        """
        return self._log