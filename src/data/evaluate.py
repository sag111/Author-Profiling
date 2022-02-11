# Модуль оценки на несколько задач
# На вход получает два файла в формате JSONL, первый с реальными метками, второй с выходом модели
# В файлах id должны соответствовать друг другу
# Оценка производится по следующим полям: age_group, gender, no_imitation, age_imitation,
# gender_imitation, style_imitation

# Формат JONL
# "id" -- уникальный идентификатор документа;
# "text" -- исходный текст;
# "author_id" -- уникальный идентификатор автора
# "source" -- исходный датасет: "gender_imit_crowdsource" или "gender_imit_free_theme_crowdsource" или "age_imit_crowdsource";
# "age" -- возраст (int);
# "age_group" -- возрастная группа: "0-19" или "20-29" или "30-39" или "40-49" или "50+";
# "gender" -- пол: "male" или "female";
# "no_imitation" -- метка документа без имитации: no_any_imitation, with_any_imitation;
# "age_imitation" -- метка документа c имитацией возраста: 0, "younger", "older" или np.nan;
# "gender_imitation" -- метка документа c имитацией пола: 0, 1 или np.nan;
# "style_imitation" -- метка документа c имитацией стиля: 0, 1 или np.nan;
# "meta" -- мета-информация исходного документа.

import numpy as np
import pandas as pd
import argparse as ap
import json

import wandb

from sklearn.metrics import f1_score, classification_report


def evaluate_all_tasks(df_y_true, df_y_pred, res_scores_file=None,
                       wandb_project=None, wandb_entity=None, wandb_config=None):
    """
    Функция оценки на 6 задач

    Parameters
    ----------
    df_y_true: pandas DataFrame
        DataFrame со считанным файлом JSONL с реальными метками
    df_y_pred: pandas DataFrame
        DataFrame со считанным файлом JSONL с предсказанными метками

    Returns
    -------
    d_scores: dict
        Словарь с оценками, два ключа:
            - details: словарь с оценками для каждой задачи, для каждой задачи рассчитывается оценка по 3-м множествам:
                - all - всё тестовое множество
                - no_im - тестовая часть без имитации;
                - with_any_im - тестовая часть только с различными имитациями (все имитации в одной части пол, возраст, стиль);
            В каждой части:
                - f1_weighted;
                - f1_micro;
                - f1_macro;
                - classification_report - в текстовом виде;
                - d_classification_report - в виде словаря;
            - average: усреднение по всем задачам для каждой метрики f1:
                - f1_weighted;
                - f1_micro;
                - f1_macro;
    """

    l_columns = ["age_group", "gender", "no_imitation", "age_imitation",
                 "gender_imitation", "style_imitation"]
    df_y_true = df_y_true.replace("None", None)
    inp_true = pd.DataFrame(df_y_true, copy=True)
    inp_pred = pd.DataFrame(df_y_pred, copy=True)
    inp_true = inp_true[["id", "author_id"] + l_columns]
    inp_pred = inp_pred[["id", "author_id"] + l_columns]

    # for column_name in l_columns:
    #     inp_true[column_name] = inp_true[column_name].astype(str)
    #     inp_pred[column_name] = inp_pred[column_name].astype(str)

    # Проверка, что у нас одинаковое число документов
    if set(inp_true["id"]) != set(inp_pred["id"]):
        raise Exception("Разные длины у файлов")

    inp_true = inp_true.set_index("id")
    inp_pred = inp_pred.set_index("id")

    df_all = inp_true.join(inp_pred, rsuffix="_pred")

    d_scores = {"details": {},
                "average": {},
                "average_by_parts": {}}
    for column_name in l_columns:
        df_column_name = df_all[df_all[column_name].notnull()]
        y_true = df_column_name[column_name].astype(str).tolist()
        y_pred = df_column_name[column_name + "_pred"].astype(str).tolist()
        d_scores["details"][column_name] = {"all": {
            "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
            "f1_micro": f1_score(y_true, y_pred, average="micro"),
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
            "classification_report": classification_report(y_true, y_pred),
            "d_classification_report": classification_report(y_true, y_pred, output_dict=True)}}

        # Детализация по двум множествам
        #   - no_im - тексты без имитации;
        #   - with_any_im - тексты с любым из видов имитации (пол, возраст или стиль);
        df_column_name_no_im = df_column_name[df_column_name["no_imitation"] == "no_any_imitation"]
        df_column_name_with_any_im = df_column_name[df_column_name["no_imitation"] == "with_any_imitation"]
        for ds_part_name, ds in zip(["no_im", "with_any_im"], [df_column_name_no_im, df_column_name_with_any_im]):
            if len(ds) > 0:
                y_true = ds[column_name].astype(str).tolist()
                y_pred = ds[column_name + "_pred"].astype(str).tolist()
                d_scores["details"][column_name][ds_part_name] = {
                    "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
                    "f1_micro": f1_score(y_true, y_pred, average="micro"),
                    "f1_macro": f1_score(y_true, y_pred, average="macro"),
                    "classification_report": classification_report(y_true, y_pred),
                    "d_classification_report": classification_report(y_true, y_pred, output_dict=True)}
            else:
                d_scores["details"][column_name][ds_part_name] = {
                    "f1_weighted": None,
                    "f1_micro": None,
                    "f1_macro": None,
                    "classification_report": None,
                    "d_classification_report": None}

    # Расчёт средней точности для каждого типа усреднения F1 метрик
    # Усреднение делается только по всему тестовому множеству
    for part_name in ["all", "no_im", "with_any_im"]:
        d_scores["average"][part_name] = {"f1_weighted": None, "f1_micro": None, "f1_macro": None}
        for k in ["f1_weighted", "f1_micro", "f1_macro"]:
            d_scores["average"][part_name][k] = np.mean([val[part_name][k] for val in d_scores["details"].values()])

    # Сохраняем результаты в файл JSON
    if res_scores_file is not None:
        with open(res_scores_file, "w") as f:
            json.dump(d_scores, f, indent=None, ensure_ascii=False)

    # Сохраняем результаты в Weights&Biases
    if wandb_project is not None:
        wandb.init(project=wandb_project, entity=wandb_entity, config=wandb_config)
        wandb.summary.update(d_scores)

    return d_scores


if __name__ == "__main__":
    args_parser = ap.ArgumentParser(description="Модуль оценки на несколько задач. "
                                                "На вход получает два файла в формате JSONL, "
                                                "первый с реальными метками, второй с выходом модели. "
                                                "В файлах text_id должны соответствовать друг другу.")
    args_parser.add_argument("inp_true", help="Тестовый файл с реальными метками в формате JSONL")
    args_parser.add_argument("inp_pred", help="Метки модели в формате JSONL")
    args_parser.add_argument("--res-scores-file", default=None, help="Файл для сохранения метрик")
    args_parser.add_argument("--wandb-project", default=None, help="Название проекта в Weights&Biases")
    args_parser.add_argument("--wandb-entity", default=None, help="Идентификация в Weights&Biases")
    args_parser.add_argument("--wandb-config-file", default=None,
                             help="JSON файл с параметрам запуска для сохранения в Weights&Biases")
    args = args_parser.parse_args()

    # Считываем данные
    inp_true = pd.read_json(args.inp_true, lines=True)
    inp_pred = pd.read_json(args.inp_pred, lines=True)

    # Конфигурация для Weights&Biases
    if args.wandb_config_file is not None:
        with open(args.wandb_config_file, "r") as f:
            d_wandb_config = json.load(f)
    else:
        d_wandb_config = None

    d_scores = evaluate_all_tasks(inp_true, inp_pred, res_scores_file=args.res_scores_file,
                                  wandb_project=args.wandb_project, wandb_entity=args.wandb_entity,
                                  wandb_config=d_wandb_config)
    print("Детализация по отдельным задачам:")
    for k, v in d_scores["details"].items():
        print(k)
        print(v["all"]["classification_report"])
        print("")
    print("Среднее по всем задачам:")
    print(d_scores["average"])

    print("Выполнено")
