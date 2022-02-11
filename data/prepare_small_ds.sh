#!/bin/bash
set -e

export GRAPH_HYPER_OPT_PATH="$(dirname "$(dirname "$(readlink -fm "$0")")")"
export PYTHONPATH=$GRAPH_HYPER_OPT_PATH:$PYTHONPATH
export INP_FOLD_DIR=$GRAPH_HYPER_OPT_PATH/data/

python $GRAPH_HYPER_OPT_PATH/src/data/download_ds.py --small-ds

# Stanza morphology and syntax
python $GRAPH_HYPER_OPT_PATH/src/data/prep_text_to_sagnlpJSON.py


mkdir $INP_FOLD_DIR/vectorized

mkdir $INP_FOLD_DIR/vectorized/gender
python $GRAPH_HYPER_OPT_PATH/src/vectorize_crowdsourceForGrant_data.py $INP_FOLD_DIR/preprocess/prep_train.jsonl $INP_FOLD_DIR/preprocess/prep_valid.jsonl $INP_FOLD_DIR/preprocess/prep_test.jsonl $INP_FOLD_DIR/vectorized/gender/ gender

mkdir $INP_FOLD_DIR/vectorized/age_group
python $GRAPH_HYPER_OPT_PATH/src/vectorize_crowdsourceForGrant_data.py $INP_FOLD_DIR/preprocess/prep_train.jsonl $INP_FOLD_DIR/preprocess/prep_valid.jsonl $INP_FOLD_DIR/preprocess/prep_test.jsonl $INP_FOLD_DIR/vectorized/age_group/ age_group

mkdir $INP_FOLD_DIR/vectorized/gender_imitation
python $GRAPH_HYPER_OPT_PATH/src/vectorize_crowdsourceForGrant_data.py $INP_FOLD_DIR/preprocess/prep_train.jsonl $INP_FOLD_DIR/preprocess/prep_valid.jsonl $INP_FOLD_DIR/preprocess/prep_test.jsonl $INP_FOLD_DIR/vectorized/gender_imitation/ gender_imitation

mkdir $INP_FOLD_DIR/vectorized/age_imitation
python $GRAPH_HYPER_OPT_PATH/src/vectorize_crowdsourceForGrant_data.py $INP_FOLD_DIR/preprocess/prep_train.jsonl $INP_FOLD_DIR/preprocess/prep_valid.jsonl $INP_FOLD_DIR/preprocess/prep_test.jsonl $INP_FOLD_DIR/vectorized/age_imitation/ age_imitation

mkdir $INP_FOLD_DIR/vectorized/style_imitation
python $GRAPH_HYPER_OPT_PATH/src/vectorize_crowdsourceForGrant_data.py $INP_FOLD_DIR/preprocess/prep_train.jsonl $INP_FOLD_DIR/preprocess/prep_valid.jsonl $INP_FOLD_DIR/preprocess/prep_test.jsonl $INP_FOLD_DIR/vectorized/style_imitation/ style_imitation

mkdir $INP_FOLD_DIR/vectorized/no_imitation
python $GRAPH_HYPER_OPT_PATH/src/vectorize_crowdsourceForGrant_data.py $INP_FOLD_DIR/preprocess/prep_train.jsonl $INP_FOLD_DIR/preprocess/prep_valid.jsonl $INP_FOLD_DIR/preprocess/prep_test.jsonl $INP_FOLD_DIR/vectorized/no_imitation/ no_imitation
echo "Data preprocessing done"