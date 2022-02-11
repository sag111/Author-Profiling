#!/bin/bash
set -e
# Скрипт для формирования выходов лучших моделей на тестовом множестве
L_RUN_NAMES=("ga_hpo_v1" "ga_pbt_v2" "gamodel_v3_no_synt" "gamodel_v4_x_const")
L_SCORE_FILES=("res_ray_hpo.json" "res_ray_pbt_v2.json" "gamodel_v3_no_synt.json" "gamodel_v4_x_const.json")
for i in "${!L_RUN_NAMES[@]}"; do
  RUN_NAME="${L_RUN_NAMES[i]}"
  echo "$RUN_NAME"_output
  python best_ga_from_ray.py $RUN_NAME --save-output
  mkdir "$RUN_NAME"_output
  mv df_pred_best_ray.csv "$RUN_NAME"_output/
  mv df_ts.csv "$RUN_NAME"_output/
  mv "${L_SCORE_FILES[i]}" "$RUN_NAME"_output/
  tar -czf "$RUN_NAME"_output.tar.gz "$RUN_NAME"_output
done