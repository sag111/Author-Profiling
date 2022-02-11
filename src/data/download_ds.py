# Скрипт для загрузки корпуса с Huggingface
import argparse as ap
import os
from datasets import load_dataset

DEFAULT_RES_DIR = os.path.join(os.path.split(os.path.split(os.path.split(os.path.abspath(__file__))[0])[0])[0], "data")


if __name__ == "__main__":
    args_parser = ap.ArgumentParser()
    args_parser.add_argument("--res-dir", help="Result dir for downloading, default: {0}".format(DEFAULT_RES_DIR),
                             default=DEFAULT_RES_DIR)
    args_parser.add_argument("--small-ds",
                             help="Debug flag, if set, then script download only 1000 documents from training set,"
                                  " for debugging",
                             action="store_true")
    args = args_parser.parse_args()

    inp_tr = load_dataset("sagteam/author_profiling", split="train").to_pandas()
    inp_vl = load_dataset("sagteam/author_profiling", split="validation").to_pandas()
    inp_ts = load_dataset("sagteam/author_profiling", split="test").to_pandas()

    inp_tr = inp_tr.replace("None", None)
    inp_vl = inp_vl.replace("None", None)
    inp_ts = inp_ts.replace("None", None)

    if args.small_ds:
        print("Download only 1000 documents from training set")
        inp_tr = inp_tr.sample(frac=1, random_state=33).iloc[:1000]

    ds_res_dir = os.path.join(args.res_dir, "raw")
    if not os.path.exists(ds_res_dir):
        os.makedirs(ds_res_dir)
    inp_tr.to_json(os.path.join(ds_res_dir, "train.jsonl"), lines=True, orient="records")
    inp_vl.to_json(os.path.join(ds_res_dir, "valid.jsonl"), lines=True, orient="records")
    inp_ts.to_json(os.path.join(ds_res_dir, "test.jsonl"), lines=True, orient="records")
    print("Dataset downloaded to {0}".format(os.path.join(args.res_dir, "raw")))
    print("Done")
