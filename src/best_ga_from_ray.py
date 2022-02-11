import numpy as np
import pandas as pd
import argparse as ap
import os
import joblib

from ray import tune

from src.data.store import read_ds
from src.models.base import GAModel
from src.data.evaluate import evaluate_all_tasks

DEFAULT_MODEL_DIR = os.path.join(os.path.split(os.path.split(os.path.abspath(__file__))[0])[0], "models")
DEFAULT_DS_DIR = os.path.join(os.path.split(os.path.split(os.path.abspath(__file__))[0])[0], "data")
DEFAULT_DS_DIR_ORIG_VECT = os.path.join(os.path.split(os.path.split(os.path.abspath(__file__))[0])[0], "data", "orig")

if __name__ == "__main__":
    args_parser = ap.ArgumentParser(
        description="Проверка моделей из ray tune на тестовом множестве. "
                    "Если указан флаг train-models то производится переобучение моделей, "
                    "обучение будет учитывать только начальную конфигурацию, "
                    "но не будет учитывать весь pipeline подобранный ray tune в случае pbt")
    args_parser.add_argument("config_name",
                             help="Название конфигурации, доступно: ga_hpo_v1, ga_pbt_v2, gamodel_v3_no_synt, gamodel_v4_x_const")
    args_parser.add_argument("--use-wandb", action="store_true",
                             help="Если указан флаг, то результат сохраниться на Weights&Biases")
    args_parser.add_argument("--ray-results-dir", default=DEFAULT_MODEL_DIR)
    args_parser.add_argument("--train-models", action="store_true",
                             help="Флаг обучения, если указан то производится обучение моделей, если нет, "
                                  "то используются сохранённые модели из ray tune")
    args_parser.add_argument("--save-output", action="store_true",
                             help="Флаг, если указан, то предсказания будут сохранены в файл df_pred_best_ray.csv, "
                                  "а оригинальный тест в ds_ts.csv")
    args = args_parser.parse_args()
    
    ray_results_dir = args.ray_results_dir

    if args.config_name == "ga_hpo_v1":
        d_configs = {"gender": {
            "config": {'add_positional_encoding': False, 'batch_size': 20, 'dropout': 0.30000000000000004,
                       'learning_rate': 0.00022952245219673756, 'n_blocks': 3, 'n_heads': 4,
                       'neurons_1_multi': 6, 'neurons_2': 120, 'use_residual': False},
            "checkpoint": f"{ray_results_dir}/hpo_gender_v1/RayTrainableGAModel_d9cbb33a_76_add_positional_encoding=False,batch_size=20,dropout=0.3,learning_rate=0.00022952,n_blocks=3,n_head_2022-01-11_16-38-05/checkpoint_000001/"
        },
            "age_group": {
                "config": {'add_positional_encoding': False, 'batch_size': 20, 'dropout': 0.5,
                           'learning_rate': 0.00023078633183788198, 'n_blocks': 3, 'n_heads': 6,
                           'neurons_1_multi': 8, 'neurons_2': 56, 'use_residual': True},
                "checkpoint": f"{ray_results_dir}/hpo_age_group_v1/RayTrainableGAModel_5d62ce9e_74_add_positional_encoding=False,batch_size=20,dropout=0.5,learning_rate=0.00023079,n_blocks=3,n_head_2022-01-12_00-34-12/checkpoint_000001/"
            },
            "age_imitation": {
                "config": {'add_positional_encoding': False, 'batch_size': 18, 'dropout': 0.30000000000000004,
                           'learning_rate': 0.0002943220382571464, 'n_blocks': 3, 'n_heads': 5, 'neurons_1_multi': 24,
                           'neurons_2': 72, 'use_residual': True},
                "checkpoint": f"{ray_results_dir}/hpo_age_imitation_v1/RayTrainableGAModel_09c2a4f2_32_add_positional_encoding=False,batch_size=18,dropout=0.3,learning_rate=0.00029432,n_blocks=3,n_head_2022-01-12_08-24-19/checkpoint_000001/"
            },
            "gender_imitation": {
                "config": {'add_positional_encoding': False, 'batch_size': 14, 'dropout': 0.6000000000000001,
                           'learning_rate': 0.0004767739894073551, 'n_blocks': 2, 'n_heads': 3, 'neurons_1_multi': 18,
                           'neurons_2': 76, 'use_residual': False},
                "checkpoint": f"{ray_results_dir}/hpo_gender_imitation_v1/RayTrainableGAModel_f741a1f8_19_add_positional_encoding=False,batch_size=14,dropout=0.6,learning_rate=0.00047677,n_blocks=2,n_head_2022-01-12_05-53-28/checkpoint_000001/"
            },
            "no_imitation": {
                "config": {'add_positional_encoding': False, 'batch_size': 6, 'dropout': 0.5,
                           'learning_rate': 0.00030550045430648756, 'n_blocks': 4, 'n_heads': 9, 'neurons_1_multi': 30,
                           'neurons_2': 24, 'use_residual': True},
                "checkpoint": f"{ray_results_dir}/hpo_no_imitation_v1/RayTrainableGAModel_dd1a8f16_33_add_positional_encoding=False,batch_size=6,dropout=0.5,learning_rate=0.0003055,n_blocks=4,n_heads=_2022-01-12_02-46-38/checkpoint_000001/"
            },
            "style_imitation": {
                "config": {'add_positional_encoding': False, 'batch_size': 24, 'dropout': 0.4,
                           'learning_rate': 0.00011970890405301316, 'n_blocks': 5, 'n_heads': 9, 'neurons_1_multi': 28,
                           'neurons_2': 76, 'use_residual': False},
                "checkpoint": f"{ray_results_dir}/hpo_style_imitation_v1/RayTrainableGAModel_207b9a64_27_add_positional_encoding=False,batch_size=24,dropout=0.4,learning_rate=0.00011971,n_blocks=5,n_head_2022-01-12_11-23-54/checkpoint_000001/"
            }
        }
        model_type = "gamodel_ray_hpo"
        res_file = "res_ray_hpo.json"
    elif args.config_name == "ga_pbt_v2":
        d_configs = {
            "gender": {
                "config": {'batch_size': 20, 'learning_rate': 0.0001758409118178723, 'neurons_1_multi': 14, 'neurons_2': 100, 'n_blocks': 3, 'n_heads': 3, 'dropout': 0.0, 'use_residual': True, 'add_positional_encoding': False},
                "checkpoint": f"{ray_results_dir}/pbt_gender_v2/RayTrainableGAModel_a2ee4_00053_53_add_positional_encoding=False,batch_size=28,dropout=0.7,learning_rate=0.0002482,n_blocks=4,n_he_2022-01-15_08-36-36/checkpoint_000008/"},
            "age_group": {
                "config": {'batch_size': 39, 'learning_rate': 0.000727320372815119, 'neurons_1_multi': 14, 'neurons_2': 52, 'n_blocks': 6, 'n_heads': 2, 'dropout': 0.0, 'use_residual': True, 'add_positional_encoding': False},
                "checkpoint": f"{ray_results_dir}/pbt_age_group_v2/RayTrainableGAModel_6dfd8_00040_40_add_positional_encoding=False,batch_size=16,dropout=0.0,learning_rate=0.0055383,n_blocks=3,n_he_2022-01-15_12-34-48/checkpoint_000014/"},
            "age_imitation": {
                "config": {'batch_size': 22, 'learning_rate': 0.00027158654941509495, 'neurons_1_multi': 20, 'neurons_2': 64, 'n_blocks': 3, 'n_heads': 8, 'dropout': 0.30000000000000004, 'use_residual': False, 'add_positional_encoding': False},
                "checkpoint": f"{ray_results_dir}/pbt_age_imitation_v2/RayTrainableGAModel_23198_00003_3_add_positional_encoding=True,batch_size=16,dropout=0.7,learning_rate=0.00026914,n_blocks=5,n_hea_2022-01-15_16-21-56/checkpoint_000010/"},
            "gender_imitation": {
                "config": {'batch_size': 24, 'learning_rate': 5.714843191785773e-05, 'neurons_1_multi': 8, 'neurons_2': 8, 'n_blocks': 7, 'n_heads': 7, 'dropout': 0.0, 'use_residual': True, 'add_positional_encoding': False},
                "checkpoint": f"{ray_results_dir}/pbt_gender_imitation_v2/RayTrainableGAModel_d7088_00008_8_add_positional_encoding=False,batch_size=28,dropout=0.0,learning_rate=0.0019007,n_blocks=7,n_hea_2022-01-15_20-23-37/checkpoint_000013/"},
            "no_imitation": {
                "config": {'batch_size': 28, 'learning_rate': 0.0005138859906348661, 'neurons_1_multi': 16, 'neurons_2': 124, 'n_blocks': 4, 'n_heads': 7, 'dropout': 0.4, 'use_residual': True, 'add_positional_encoding': False},
                "checkpoint": f"{ray_results_dir}/pbt_no_imitation_v2/RayTrainableGAModel_f267f_00070_70_add_positional_encoding=True,batch_size=18,dropout=0.3,learning_rate=0.0083136,n_blocks=6,n_hea_2022-01-15_23-03-15/checkpoint_000008/"},
            "style_imitation": {
                "config": {'batch_size': 14, 'learning_rate': 0.00042054424379559603, 'neurons_1_multi': 16, 'neurons_2': 104, 'n_blocks': 2, 'n_heads': 4, 'dropout': 0.2, 'use_residual': True, 'add_positional_encoding': False},
                "checkpoint": f"{ray_results_dir}/pbt_style_imitation_v2/RayTrainableGAModel_a90e3_00047_47_add_positional_encoding=True,batch_size=32,dropout=0.1,learning_rate=0.0006261,n_blocks=4,n_hea_2022-01-16_02-49-13/checkpoint_000015/"}
        }
        model_type = "gamodel_ray_pbt_v2"
        res_file = "res_ray_pbt_v2.json"
    elif args.config_name == "gamodel_v3_no_synt":
        d_configs = {
            "gender": {
                "config": {'batch_size': 46, 'learning_rate': 0.0020647836307671023, 'neurons_1_multi': 8, 'neurons_2': 68, 'n_blocks': 2, 'n_heads': 3, 'dropout': 0.1, 'use_residual': False, 'add_positional_encoding': False, 'no_synt': True},
                "checkpoint": f"{ray_results_dir}/pbt_genderv3_no_synt/RayTrainableGAModel_2d24b_00052_52_add_positional_encoding=True,batch_size=26,dropout=0.4,learning_rate=0.0091454,n_blocks=1,n_hea_2022-01-28_16-02-13/checkpoint_000008/"},
# Почему-то эта модель с ошибкой, то ли что-то не так сохранилось, то ли ещё что-то
#            "age_group": {
#                "config": {'batch_size': 15, 'learning_rate': 0.00014646124559815834, 'neurons_1_multi': 6, 'neurons_2': 8, 'n_blocks': 2, 'n_heads': 5, 'dropout': 0.1, 'use_residual': True, 'add_positional_encoding': True, 'no_synt': True},
#                "checkpoint": f"{ray_results_dir}/pbt_age_groupv3_no_synt/RayTrainableGAModel_f0f4a_00043_43_add_positional_encoding=True,batch_size=20,dropout=0.2,learning_rate=0.003574,n_blocks=3,n_head_2022-01-28_20-01-48/checkpoint_000010/"},
            # Подобрал конфиг для checkpoint
            "age_group": {
               "config": {'batch_size': 39, 'learning_rate': 0.00014638010436027443, 'neurons_1_multi': 6, 'neurons_2': 128, 'n_blocks': 1, 'n_heads': 9, 'dropout': 0.0, 'use_residual': True, 'add_positional_encoding': False, 'no_synt': True},
               "checkpoint": f"{ray_results_dir}/pbt_age_groupv3_no_synt/RayTrainableGAModel_f0f4a_00043_43_add_positional_encoding=True,batch_size=20,dropout=0.2,learning_rate=0.003574,n_blocks=3,n_head_2022-01-28_20-01-48/checkpoint_000010/"},
            "age_imitation": {
                "config": {'batch_size': 26, 'learning_rate': 0.00016414457495278524, 'neurons_1_multi': 12, 'neurons_2': 20, 'n_blocks': 4, 'n_heads': 9, 'dropout': 0.1, 'use_residual': False, 'add_positional_encoding': False, 'no_synt': True},
                "checkpoint": f"{ray_results_dir}/pbt_age_imitationv3_no_synt/RayTrainableGAModel_fd95d_00076_76_add_positional_encoding=True,batch_size=12,dropout=0.4,learning_rate=0.00074305,n_blocks=6,n_he_2022-01-29_00-01-43/checkpoint_000011/"},
            "gender_imitation": {
                "config": {'batch_size': 20, 'learning_rate': 0.00044890296374129027, 'neurons_1_multi': 20, 'neurons_2': 56, 'n_blocks': 5, 'n_heads': 9, 'dropout': 0.2, 'use_residual': True, 'add_positional_encoding': False, 'no_synt': True},
                "checkpoint": f"{ray_results_dir}/pbt_gender_imitationv3_no_synt/RayTrainableGAModel_af393_00007_7_add_positional_encoding=True,batch_size=4,dropout=0.3,learning_rate=0.0032522,n_blocks=7,n_heads_2022-01-29_03-45-10/checkpoint_000031/"},
            "no_imitation": {
                "config": {'batch_size': 15, 'learning_rate': 0.0002459590433158391, 'neurons_1_multi': 28, 'neurons_2': 84, 'n_blocks': 1, 'n_heads': 9, 'dropout': 0.1, 'use_residual': False, 'add_positional_encoding': False, 'no_synt': True},
                "checkpoint": f"{ray_results_dir}/pbt_no_imitationv3_no_synt/RayTrainableGAModel_2ce41_00035_35_add_positional_encoding=False,batch_size=16,dropout=0.5,learning_rate=0.0015891,n_blocks=3,n_he_2022-01-29_05-32-25/checkpoint_000010/"},
            "style_imitation": {
                "config": {'batch_size': 31, 'learning_rate': 0.00010151647030757077, 'neurons_1_multi': 12, 'neurons_2': 44, 'n_blocks': 2, 'n_heads': 5, 'dropout': 0.1, 'use_residual': True, 'add_positional_encoding': False, 'no_synt': True},
                "checkpoint": f"{ray_results_dir}/pbt_style_imitationv3_no_synt/RayTrainableGAModel_ece6f_00003_3_add_positional_encoding=True,batch_size=12,dropout=0.6,learning_rate=0.0040922,n_blocks=6,n_head_2022-01-29_09-22-40/checkpoint_000013/"}
        }
        model_type = "gamodel_v3_no_synt"
        res_file = "gamodel_v3_no_synt.json"
    elif args.config_name == "gamodel_v4_x_const":
        d_configs = {
            "gender": {
                "config": {'batch_size': 24, 'learning_rate': 0.0003152263676188195, 'neurons_1_multi': 28, 'neurons_2': 56, 'n_blocks': 2, 'n_heads': 2, 'dropout': 0.2, 'use_residual': True, 'add_positional_encoding': False, 'x_const': True},
                "checkpoint": f"{ray_results_dir}/pbt_genderv4_x_const/RayTrainableGAModel_ecb37_00029_29_add_positional_encoding=False,batch_size=4,dropout=0.5,learning_rate=0.0036114,n_blocks=7,n_hea_2022-01-30_14-46-45/checkpoint_000007/"},
# # Почему-то эта модель с ошибкой, то ли что-то не так сохранилось, то ли ещё что-то
#             "age_group": {
#                 "config": {'batch_size': 9, 'learning_rate': 0.0003085917948722751, 'neurons_1_multi': 6, 'neurons_2': 64, 'n_blocks': 1, 'n_heads': 5, 'dropout': 0.30000000000000004, 'use_residual': True, 'add_positional_encoding': False, 'x_const': True},
#                 "checkpoint": f"{ray_results_dir}/pbt_age_groupv4_x_const/RayTrainableGAModel_b5b6d_00041_41_add_positional_encoding=True,batch_size=24,dropout=0.2,learning_rate=0.00033487,n_blocks=2,n_he_2022-01-30_18-51-18/checkpoint_000001/"},
            # Подобрал конфигурацию для checkpoint
            "age_group": {
                "config": {'batch_size': 24, 'learning_rate': 0.0003348651286640455, 'neurons_1_multi': 8, 'neurons_2': 48, 'n_blocks': 2, 'n_heads': 2, 'dropout': 0.2, 'use_residual': False, 'add_positional_encoding': True, 'x_const': True},
                "checkpoint": f"{ray_results_dir}/pbt_age_groupv4_x_const/RayTrainableGAModel_b5b6d_00041_41_add_positional_encoding=True,batch_size=24,dropout=0.2,learning_rate=0.00033487,n_blocks=2,n_he_2022-01-30_18-51-18/checkpoint_000001/"},
            "age_imitation": {
                "config": {'batch_size': 8, 'learning_rate': 8.595352057824465e-05, 'neurons_1_multi': 10, 'neurons_2': 44, 'n_blocks': 7, 'n_heads': 3, 'dropout': 0.7000000000000001, 'use_residual': False, 'add_positional_encoding': True, 'x_const': True},
                "checkpoint": f"{ray_results_dir}/pbt_age_imitationv4_x_const/RayTrainableGAModel_9fddd_00032_32_add_positional_encoding=False,batch_size=26,dropout=0.5,learning_rate=0.0051751,n_blocks=1,n_he_2022-01-30_21-01-41/checkpoint_000012/"},
            "gender_imitation": {
                "config": {'batch_size': 10, 'learning_rate': 9.089918317550383e-05, 'neurons_1_multi': 30, 'neurons_2': 96, 'n_blocks': 4, 'n_heads': 6, 'dropout': 0.0, 'use_residual': True, 'add_positional_encoding': True, 'x_const': True},
                "checkpoint": f"{ray_results_dir}/pbt_gender_imitationv4_x_const/RayTrainableGAModel_42498_00060_60_add_positional_encoding=False,batch_size=4,dropout=0.2,learning_rate=0.0042415,n_blocks=6,n_hea_2022-01-31_01-04-13/checkpoint_000004/"},
            "no_imitation": {
                "config": {'batch_size': 20, 'learning_rate': 8.934571597589663e-05, 'neurons_1_multi': 20, 'neurons_2': 108, 'n_blocks': 2, 'n_heads': 1, 'dropout': 0.0, 'use_residual': True, 'add_positional_encoding': True, 'x_const': True},
                "checkpoint": f"{ray_results_dir}/pbt_no_imitationv4_x_const/RayTrainableGAModel_a037c_00032_32_add_positional_encoding=False,batch_size=12,dropout=0.4,learning_rate=0.00045311,n_blocks=7,n_h_2022-01-31_02-03-56/checkpoint_000002/"},
            "style_imitation": {
                "config": {'batch_size': 12, 'learning_rate': 9.979186271070267e-05, 'neurons_1_multi': 8, 'neurons_2': 104, 'n_blocks': 7, 'n_heads': 3, 'dropout': 0.30000000000000004, 'use_residual': True, 'add_positional_encoding': True, 'x_const': True},
                "checkpoint": f"{ray_results_dir}/pbt_style_imitationv4_x_const/RayTrainableGAModel_6ed58_00027_27_add_positional_encoding=True,batch_size=24,dropout=0.1,learning_rate=0.0011069,n_blocks=2,n_hea_2022-01-31_03-58-24/checkpoint_000004/"},
        }
        model_type = "gamodel_v4_x_const"
        res_file = "gamodel_v4_x_const.json"
    elif args.config_name == "ga_old_best_v1":
        d_best = {'nb_epochs': 100,
                  'batch_size': 32,
                  'learning_rate': 0.001,
                  'neurons_1_multi': 4,
                  'neurons_2': 104,
                  'n_blocks': 5,
                  'n_heads': 5,
                  'dropout': 0.2,
                  'use_residual': True,
                  'add_positional_encoding': False}
        d_configs = {}
        for val in ["gender", "age_group", "age_imitation", "gender_imitation", "no_imitation", "style_imitation"]:
            d_configs[val] = {"config": d_best, "checkpoint": None}
            model_type = "gamodel_old_best_v1"
            res_file = "res_gamodel_old_best_v1.json"

    df_ts = pd.read_json(os.path.join(DEFAULT_DS_DIR, "raw/test.jsonl"), lines=True)
    df_ts_author_id = df_ts[["id", "author_id"]]
    df_ts_author_id = df_ts_author_id.set_index("id")
    l_pred = []
    for task_name, d_conf in d_configs.items():
        x_ts, y_ts = read_ds(os.path.join(DEFAULT_DS_DIR_ORIG_VECT, "vectorized/{0}/ts.h5".format(task_name)))
        doc_ids = pd.read_json(os.path.join(DEFAULT_DS_DIR_ORIG_VECT, "vectorized/{0}/ts_ids.jsonl".format(task_name)),
                               lines=True)
        y_vect = joblib.load(os.path.join(DEFAULT_DS_DIR_ORIG_VECT, "vectorized/{0}/y_enc.pkl".format(task_name)))
        model = GAModel(word_dim=len(x_ts[0][0][0]), y_dim=len(y_ts[0]), **d_conf["config"])

        if args.train_models:
            print("Обучаем модель GAModel с параметрами:")
            print(d_conf)
            x_tr, y_tr = read_ds(os.path.join(DEFAULT_DS_DIR_ORIG_VECT, "vectorized/{0}/tr.h5".format(task_name)))
            x_vl, y_vl = read_ds(os.path.join(DEFAULT_DS_DIR_ORIG_VECT, "vectorized/{0}/vl.h5".format(task_name)))
            model.fit(x_tr, y_tr, valid_data=(x_vl, y_vl))
        else:
            print("Загружаем веса модели из checkpoint:")
            if os.path.isfile(os.path.join(d_conf["checkpoint"], "model_weights.h5")):
                model_weights = os.path.join(d_conf["checkpoint"], "model_weights.h5")
            else:
                model_weights = os.path.join(d_conf["checkpoint"], "model.h5")
            print(model_weights)
            model.model.load_weights(model_weights)

        pred = model.predict(x_ts, batch_size=1)
        pred_bin = np.zeros_like(pred)
        pred_bin[np.arange(0, len(pred)), np.argmax(pred, axis=1)] = 1
        y_pred_labels = [val[0] for val in y_vect.inverse_transform(pred_bin)]
        df_task_pred = pd.DataFrame({task_name: y_pred_labels}, index=doc_ids[0].values)
        l_pred.append(df_task_pred)
        print(task_name)
    df_pred = pd.concat(l_pred, axis=1)
    df_pred = df_pred.join(df_ts_author_id)
    df_pred["id"] = df_pred.index
    if args.save_output:
        df_pred.to_csv("df_pred_best_ray.csv")
        df_ts.to_csv("df_ts.csv")
    if args.use_wandb:
        wandb_project = "deb-GraphHyperOpt"
        wandb_config = {"model_type": model_type,
                        "configs": d_configs}
    else:
        wandb_project = None
        wandb_config = None
    d_scores = evaluate_all_tasks(df_ts, df_pred,
                                  res_scores_file=res_file,
                                  wandb_project=wandb_project,
                                  wandb_config=wandb_config)
    print("Выполнено")

