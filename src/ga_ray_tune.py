# Подбор гиперпараметров GAModel с использование ray tune

import ray
from ray import tune
from ray.tune.integration.wandb import WandbLoggerCallback
from ray.tune.schedulers import PopulationBasedTraining, ASHAScheduler
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.stopper import TrialPlateauStopper, MaximumIterationStopper, TimeoutStopper, CombinedStopper

import numpy as np
import pandas as pd
import os
import argparse as ap
from datetime import timedelta

import tensorflow as tf
from sklearn.metrics import f1_score

from src.data.store import read_ds
from src.models.base import GAModel

BASE_DIR = os.path.abspath(os.path.expanduser("./outputs_ray_tune_bert"))

# Для динамического роста памяти
physical_devices = tf.config.list_physical_devices('GPU')
for i in range(len(physical_devices)):
    try:
      tf.config.experimental.set_memory_growth(physical_devices[i], True)
    except:
      # Invalid device or cannot modify virtual devices once initialized.
      pass


class RayTrainableGAModel(tune.Trainable):
    def setup(self, config, tr_ds_path, vl_ds_path, base_dir=BASE_DIR,
              max_n_epochs=None, early_stopping_patience=None, valideate_on_training=False):
        self.x_tr, self.y_tr = read_ds(tr_ds_path)
        self.x_vl, self.y_vl = read_ds(vl_ds_path)

        # uncoment for debugging
        # self.x_tr = self.x_tr[:100]
        # self.y_tr = self.y_tr[:100]
        # self.x_vl = self.x_vl[:100]
        # self.y_vl = self.y_vl[:100]

        self.max_n_epochs = max_n_epochs
        self.early_stopping_patience = early_stopping_patience
        self.valideate_on_training = valideate_on_training
        if self.early_stopping_patience is not None:
            config["patience"] = self.early_stopping_patience
        self.word_dim = len(self.x_tr[0][0][0])
        self.y_dim = len(self.y_tr[0])
        self.model = GAModel(word_dim=self.word_dim, y_dim=self.y_dim, **config)

    def step(self):
        if self.max_n_epochs is not None:
            nb_epochs = self.max_n_epochs
        else:
            nb_epochs = 1
        if self.valideate_on_training:
            validation_data = (self.x_vl, self.y_vl)
        else:
            validation_data = None
        self.model.fit(self.x_tr, self.y_tr, valid_data=validation_data, nb_epochs=nb_epochs)
        pred = self.model.predict(self.x_vl, batch_size=4)
        f1_weighted = f1_score(np.argmax(self.y_vl, axis=1), np.argmax(pred, axis=1), average="weighted")
        return {"f1_weighted": f1_weighted}

    def save_checkpoint(self, tmp_checkpoint_dir):
        # res_dir = os.path.join(tmp_checkpoint_dir, "model_weights.h5")
        self.model.save_model(tmp_checkpoint_dir)
        # self.model.model.save_weights(res_dir)
        return tmp_checkpoint_dir

    def load_checkpoint(self, checkpoint):
        # self.model = self.model.load_model(checkpoint)
        self.model.model.load_weights(os.path.join(checkpoint, "model.h5"))


if __name__ == "__main__":
    args_parser = ap.ArgumentParser(formatter_class=ap.RawTextHelpFormatter)
    args_parser.add_argument("tr", help="Путь до тренировочной части в формате векторов")
    args_parser.add_argument("vl", help="Путь до валидационной части в формате векторов")
    args_parser.add_argument("ts", help="Путь до тестовой части в формате векторов")
    args_parser.add_argument("--debug", action="store_true", default=False, help="Флаг для отладки")
    args_parser.add_argument("--training_iteration", default=4, type=int,
                             help="Число итераций обучения для одного экземпляра модели")
    args_parser.add_argument("--num_samples", default=4, type=int,
                             help="Число различных начальных гиперпараметров для исследования")
    args_parser.add_argument("--keep_checkpoints_num", default=4, type=int,
                             help="Сколько сохранять моделей, для гарантированного сохранения лучшей модели"
                                  " ставить число равное training_iteration * num_samples")
    args_parser.add_argument("--run-name", default="run_ga_model", help="Название всего запуска")
    args_parser.add_argument("--max_concurrent", default=None,
                             help="Сколько одновременно запусков может быть", type=int)
    args_parser.add_argument("--no-synt", action="store_true", default=False,
                             help="Флаг, если указан, то синтаксис не используется")
    args_parser.add_argument("--x-const", action="store_true", default=False,
                             help="Флаг, если указан, то все морфологические признаки заменяются одной константой")
    args_parser.add_argument("--search-alg", default="pbt",
                             help="Алгоритм подбора гиперпараметров, возможные значения:\n"
                                  " - pbt - Population Based Training,\n"
                                  " - asha - ASHA scheduler,\n"
                                  " - hyperopt - HyperOpt")
    args_parser.add_argument("--hpo-n-max-epochs", default=100, type=int,
                             help="Максимальное число эпох для обучения, при использовании алгоритма hyperopt")
    args_parser.add_argument("--hpo-patience", default=10, type=int,
                             help="Число эпох до раннего останова, если точность по валидационному множеству падает,"
                                  " при использовании алгоритма hyperopt")
    args_parser.add_argument("--hpo-n-initial-points", default=20, type=int,
                            help="Число запусков со случайными гиперпараметрами до начала оптимизации методом HyperOpt")
    args_parser.add_argument("--use-wandb", action="store_true", default=False,
                             help="Сохранять ли лог в проект dev-GraphHyperOpt-ray на Weights&Biases, "
                                  "обязательно нужен файл с ключом для доступа к wandb по пути ~/wandb_api.txt")
    args = args_parser.parse_args()

    stoppers = CombinedStopper(
        TrialPlateauStopper(metric="f1_weighted", num_results=4, mode="max"),
        MaximumIterationStopper(max_iter=args.training_iteration),
        TimeoutStopper(timedelta(hours=4))  # 4 часа на весь эксперимент
    )

    d_search_config = {
        "batch_size": tune.qrandint(2, 32, 2),
        "learning_rate": tune.loguniform(0.0001, 0.01),
        "neurons_1_multi": tune.qrandint(4, 8, 2),
        "neurons_2": tune.qrandint(4, 16, 4),
        "n_blocks": tune.randint(1, 3),
        "n_heads": tune.randint(1, 3),
        "dropout": tune.quniform(0.0, 0.7, 0.1),
        "use_residual": tune.choice([True, False]),
        "add_positional_encoding": tune.choice([True, False])
    }

    # uncoment for debugging
    # d_search_config = {
    #         "batch_size": tune.qrandint(2, 32, 2),
    #         "learning_rate": tune.loguniform(0.0001, 0.01),
    #         "neurons_1_multi": tune.qrandint(4, 32, 2),
    #         "neurons_2": tune.qrandint(4, 128, 4),
    #         "n_blocks": tune.randint(1, 8),
    #         "n_heads": tune.randint(1, 10),
    #         "dropout": tune.quniform(0.0, 0.7, 0.1),
    #         "use_residual": tune.choice([True, False]),
    #         "add_positional_encoding": tune.choice([True, False])
    #         }

    if args.no_synt:
        d_search_config["no_synt"] = True
    if args.x_const:
        d_search_config["x_const"] = True

    if args.search_alg.lower() in ["pbt", "asha"]:
        if args.search_alg.lower() == "pbt":
            scheduler = PopulationBasedTraining(
                time_attr="training_iteration",
                perturbation_interval=1,
                hyperparam_mutations={
                    "learning_rate": tune.loguniform(0.0001, 0.01),
                    "batch_size": tune.qrandint(2, 32, 2)
                }
            )
        elif args.search_alg.lower() == "asha":
            scheduler = ASHAScheduler(metric="f1_weighted", mode="max")
        if args.max_concurrent is not None:
            scheduler = ConcurrencyLimiter(scheduler, max_concurrent=args.max_concurrent, batch=True)
        search_alg = None
        d_run_config = d_search_config
        max_n_epochs = 1
        patience = None
        valideate_on_training = False  # Относится только к валидации в самой модели на Keras, не к ray
    elif args.search_alg.lower() == "hyperopt":
        scheduler = None
        patience = args.hpo_patience
        max_n_epochs = args.hpo_n_max_epochs
        valideate_on_training = True  # Относится только к валидации в самой модели на Keras, не к ray
        search_alg = HyperOptSearch(space=d_search_config,
                                    metric="f1_weighted", mode="max",
                                    n_initial_points=args.hpo_n_initial_points)
        d_run_config = None

    # config = {"learning_rate": tune.loguniform(1e-6, 1e-3),
    #           "train_batch_size": tune.randint(1, 32)}
    callbacks = None
    if args.use_wandb:
        config_wandb = {
            "project": "dev-GraphHyperOpt-ray",
            "api_key_file": os.path.expanduser("~/wandb_api.txt"),
            "log_config": True,
            "group": args.run_name
        }
        callbacks = [WandbLoggerCallback(**config_wandb)]

    analysis = tune.run(
        tune.with_parameters(RayTrainableGAModel, tr_ds_path=args.tr, vl_ds_path=args.vl,
                             max_n_epochs=max_n_epochs, early_stopping_patience=patience,
                             valideate_on_training=valideate_on_training),
        metric="f1_weighted",
        mode="max",
        name=args.run_name,
        scheduler=scheduler,
        search_alg=search_alg,
        stop=stoppers,
        num_samples=args.num_samples,
        resources_per_trial={"cpu": 0, "gpu": 1},
        config=d_run_config,
        checkpoint_score_attr="f1_weighted",
        checkpoint_freq=1,
        keep_checkpoints_num=args.keep_checkpoints_num,
        checkpoint_at_end=True,
        log_to_file=True,
        raise_on_failed_trial=False,
        callbacks=callbacks
    )
    best_trial = analysis.get_best_trial(scope="all")
    best_checkpoint = analysis.get_best_checkpoint(best_trial)
    print("Лучшая модель:")
    print(best_trial)
    print(best_trial.config)
    print(best_checkpoint)

    x_ts, y_ts = read_ds(args.ts)
    word_dim = len(x_ts[0][0][0])
    y_dim = len(y_ts[0])
    model = GAModel(word_dim=word_dim, y_dim=y_dim, **best_trial.config)
    model.load_model(best_checkpoint)
    # model.model.load_weights(os.path.join(best_checkpoint, "model_weights.h5"))
    pred = model.predict(x_ts)
    print("Test, f1_weighted: {0};".format(f1_score(np.argmax(y_ts, axis=1), np.argmax(pred, axis=1),
                                                    average="weighted")))
    print("")

    print("Выполнено")
