# Модуль для разбивки на тренировочное, валидационное и тестовые множества и векторизации данных

import numpy as np
import argparse as ap
import os
import joblib
import json

from sklearn.preprocessing import MultiLabelBinarizer
from src.data.vectorizers import OneHotMorphWordVectorizer, get_synt_matrix, FTVectorizer, TFIDFSyntMatrix
from src.data.store import save_ds

def dump_jsonl(data, output_path, append=False):
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        for line in data:
            json_record = json.dumps(line, ensure_ascii=False)
            f.write(json_record + '\n')
    print('Wrote {} records to {}'.format(len(data), output_path))
    
def load_jsonl(input_path: str, y_field_for_none_check=None) -> list:
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            json_line = json.loads(line.rstrip('\n|\r'))
            if y_field_for_none_check is not None:
                if json_line[y_field_for_none_check] is not None:
                    data.append(json_line)
            else:
                data.append(json_line)
    print('Loaded {} records from {}'.format(len(data), input_path))
    
    return data

if __name__ == "__main__":
    args_parser = ap.ArgumentParser()
    args_parser.add_argument("inp_ds_tr")
    args_parser.add_argument("inp_ds_vl")
    args_parser.add_argument("inp_ds_ts")
    args_parser.add_argument("res_dir")
    args_parser.add_argument("y_field")
    args_parser.add_argument("--vect-type", default="morph", help="Тип векторизатора: morph, fasttext")
    args_parser.add_argument("--tf-idf-matrix", action="store_true", help="Флаг, если указан используется TF-IDF матрица, вместо синтаксической")
    args_parser.add_argument("--ft-model-path", default=None, help="Путь до модели FastText")
    args_parser.add_argument("--ft-model-type", default="gensim", help="Тип модели FastText: gensim, fasttext")
    args = args_parser.parse_args()

    inp_tr = load_jsonl(os.path.join(args.inp_ds_tr), y_field_for_none_check=args.y_field)
    # inp_tr = [doc for doc in inp_tr if not (isinstance(doc[args.y_field], float) \
    #                                     and np.isnan(doc[args.y_field]))]
    inp_vl = load_jsonl(os.path.join(args.inp_ds_vl), y_field_for_none_check=args.y_field)
    # inp_vl = [doc for doc in inp_vl if not (isinstance(doc[args.y_field], float) \
    #                                     and np.isnan(doc[args.y_field]))]
    inp_ts = load_jsonl(os.path.join(args.inp_ds_ts), y_field_for_none_check=args.y_field)
    # inp_ts = [doc for doc in inp_ts if not (isinstance(doc[args.y_field], float) \
    #                                     and np.isnan(doc[args.y_field]))]
    
    dump_jsonl([i["id"] for i in inp_tr], os.path.join(args.res_dir, "tr_ids.jsonl"))
    dump_jsonl([i["id"] for i in inp_ts], os.path.join(args.res_dir, "ts_ids.jsonl"))
    dump_jsonl([i["id"] for i in inp_vl], os.path.join(args.res_dir, "vl_ids.jsonl"))
    
    if args.vect_type == "morph":
        vect = OneHotMorphWordVectorizer()
        vect.fit(inp_tr)
        joblib.dump(vect, os.path.join(args.res_dir, "vect.pkl"))
    elif args.vect_type == "fasttext":
        vect = FTVectorizer(args.ft_model_path, args.ft_model_type)
        with open(os.path.join(args.res_dir, "vect_ft.txt"), "w") as f:
            f.write("model_path: {0}\n".format(args.ft_model_path))
            f.write("model_type: {0}".format(args.ft_model_type))
    x1_tr = vect.transform(inp_tr, pad_sequence=False, add_virtual_root=True)
    x1_vl = vect.transform(inp_vl, pad_sequence=False, add_virtual_root=True)
    x1_ts = vect.transform(inp_ts, pad_sequence=False, add_virtual_root=True)

    if args.tf_idf_matrix:
        tf_idf_matrix = TFIDFSyntMatrix()
        tf_idf_matrix.fit(inp_tr)
        x2_tr = tf_idf_matrix.transform(inp_tr)
        x2_vl = tf_idf_matrix.transform(inp_vl)
        x2_ts = tf_idf_matrix.transform(inp_ts)
        joblib.dump(tf_idf_matrix, os.path.join(args.res_dir, "tf_idf_matrix.pkl"))
    else:
        x2_tr = np.array([get_synt_matrix(doc) for doc in inp_tr])
        x2_vl = np.array([get_synt_matrix(doc) for doc in inp_vl])
        x2_ts = np.array([get_synt_matrix(doc) for doc in inp_ts])

    y_enc = MultiLabelBinarizer()
    y_tr = np.array([doc[args.y_field] for doc in inp_tr]).reshape((-1, 1))
    y_vl = np.array([doc[args.y_field] for doc in inp_vl]).reshape((-1, 1))
    y_ts = np.array([doc[args.y_field] for doc in inp_ts]).reshape((-1, 1))
    y_enc.fit(y_tr)
    joblib.dump(y_enc, os.path.join(args.res_dir, "y_enc.pkl"))
    y_tr = y_enc.transform(y_tr)
    y_vl = y_enc.transform(y_vl)
    y_ts = y_enc.transform(y_ts)

    save_ds(x1_tr, x2_tr, y_tr, os.path.join(args.res_dir, "tr.h5"))
    save_ds(x1_vl, x2_vl, y_vl, os.path.join(args.res_dir, "vl.h5"))
    save_ds(x1_ts, x2_ts, y_ts, os.path.join(args.res_dir, "ts.h5"))

    print("Выполнено")