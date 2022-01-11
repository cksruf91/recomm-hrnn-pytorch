import os

import mlflow
import numpy as np


class ModelCheckPoint:

    def __init__(self, file, mf_logger=None, save_best=True, monitor='val_loss', mode='min'):
        self.file = file
        save_dir = os.path.dirname(self.file)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        self.mf_logger = mf_logger
        self.save_best = save_best
        self.monitor = monitor
        self.mode = mode
        init_values = {'min': np.inf, 'max': -np.inf}
        self.best_score = init_values[mode]

    def __call__(self, model, history):
        val_score = history[self.monitor]
        check_point = self.file.format(**history)

        if not self.save_best:
            self.save_model(model, check_point)
        elif self._best(val_score, self.best_score):
            self.best_score = val_score
            self.save_model(model, check_point)

    def _best(self, val, best):
        if self.mode == 'min':
            return val <= best
        else:
            return val >= best

    def save_model(self, model, file_name):
        if self.mf_logger is not None:
            self.mf_logger.log_model(model, "torch_model")
        model.save(file_name)


class TrainHistory:

    def __init__(self, file):
        self.file = file
        if os.path.isfile(self.file):
            with open(self.file, 'a') as f:
                f.write('\n')

    def __call__(self, model, history):
        with open(self.file, 'a+') as f:
            f.write(str(history) + '\n')


class MlflowLogger:

    def __init__(self, experiment_name: str, model_params: dict, run_name=None):
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.model_params = model_params
        self._set_env()
        self.run_id = self._get_run_id()

    def __call__(self, model, history):
        with mlflow.start_run(run_id=self.run_id):
            mlflow.log_metrics(history, step=history['epoch'])

    def __eq__(self, other):
        return "MLFlow" == other

    def _get_run_id(self):
        with mlflow.start_run(run_name=self.run_name) as mlflow_run:
            mlflow.log_params(self.model_params)
            run_id = mlflow_run.info.run_id
        return run_id

    def _set_env(self):
        if os.getenv('MLFLOW_TRACKING_URI') is None:
            raise ValueError("Environment variable MLFLOW_TRACKING_URI is not exist")

        mlflow.set_experiment(self.experiment_name)

    def log_model(self, model, name):
        with mlflow.start_run(run_id=self.run_id):
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path=name
            )