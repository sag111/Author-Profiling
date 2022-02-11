import json
import hashlib
import numpy as np
import pandas as pd
import argparse as ap
from collections import Counter

def load_jsonl(input_path: str) -> list:
    """
    Read list of objects from a JSON lines file.
    """
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.rstrip('\n|\r')))
    print('Loaded {} records from {}/n'.format(len(data), input_path))
    
    return data

def calc_statistics(data):
    count_doc = 0
    uniq_hash_texts, uniq_token_forms, uniq_token_lemms, uniq_authors = set(), set(), set(), set()
    doc_tokens_len, doc_chars_len, doc_sents_len, genders, ages, age_groups = [], [], [], [], [], []
    no_imitation, gender_imitation, age_imitation, style_imitation = [], [], [], []
    for doc in data:
        count_doc += 1
        doc_len = 0
        
        # uniq texts
        hash_text = hashlib.md5(doc["text"].lower().strip().encode()).hexdigest()
        uniq_hash_texts.add(hash_text)
        
        # doc lens
        doc_chars_len.append(len(doc["text"]))
        doc_sents_len.append(len(doc["sentences"]))
        for sent in doc["sentences"]:
            doc_len += len(sent)
            # uniq tokens lens
            for token in sent:
                uniq_token_forms.add(token["forma"])
                uniq_token_lemms.add(token["lemma"])
        doc_tokens_len.append(doc_len)
        
        # uniq authors
        uniq_authors.add(str(doc["author_id"])+"_"+str(doc["gender"]))
        
        genders.append(doc["gender"])
        ages.append(doc["age"])
        age_groups.append(doc["age_group"])
        no_imitation.append(doc["no_imitation"])
        gender_imitation.append(doc["gender_imitation"])
        age_imitation.append(doc["age_imitation"])
        style_imitation.append(doc["style_imitation"])
        
    uniq_authors_count = Counter([i.split("_")[1] for i in uniq_authors])
    genders_count = Counter(genders)
    age_groups_count = Counter(age_groups)
    no_imitation_count = Counter(no_imitation)
    gender_imitation_count = Counter([i if i is not None else "nan" for i in gender_imitation])
    age_imitation_count = Counter([i if type(i)==str or i is not None else "nan" for i in age_imitation])
    style_imitation_count = Counter([i if i is not None else "nan" for i in style_imitation])
    
    print(f"Кол-во документов -- {count_doc}")
    print(f"Кол-во уникальных текстов -- {len(uniq_hash_texts)}")
    print(f"Длина текстов в токенах -- min: {min(doc_tokens_len)}, max: {max(doc_tokens_len)}, mean: {round(np.mean(doc_tokens_len),1)}")
    print(f"Длина текстов в символах -- min: {min(doc_chars_len)}, max: {max(doc_chars_len)}, mean: {round(np.mean(doc_chars_len),1)}")
    print(f"Длина текстов в предложениях -- min: {min(doc_sents_len)}, max: {max(doc_sents_len)}, mean: {round(np.mean(doc_sents_len),1)}")
    print(f"Кол-во уникальных токенов -- форм: {len(uniq_token_forms)}, лемм: {len(uniq_token_lemms)}")
    print(f"Кол-во документов написанных -- мужчинами: {genders_count['male']}, женщинами: {genders_count['female']}")
    print(f"Кол-во уникальных авторов -- {len(uniq_authors)}; мужчин: {uniq_authors_count['male']}, женщин: {uniq_authors_count['female']}")
    print(f"Возраст авторов написанных документов -- min: {min(ages)}, max: {max(ages)}, mean: {round(np.mean(ages),1)}")
    print(f"Кол-во документов по возрастным группам авторов -- 1-19: {age_groups_count['0-19']}, 20-29: {age_groups_count['20-29']}, 30-39: {age_groups_count['30-39']}, 40-49: {age_groups_count['40-49']}, 50+: {age_groups_count['50+']}")
    print(f"Кол-во документов с имитацией: {no_imitation_count['with_any_imitation']}; без имитаций: {no_imitation_count['no_any_imitation']}")
    print(f"Кол-во документов с имитацией пола: {gender_imitation_count['with_gender_imitation']}; без имитаций: {gender_imitation_count['no_gender_imitation']}; неприменимо: {gender_imitation_count['nan']}")
    print(f"Кол-во документов с имитацией возраста -- younger: {age_imitation_count['younger']}; older: {age_imitation_count['older']}; без имитаций: {age_imitation_count['no_age_imitation']}; неприменимо: {age_imitation_count['nan']}")
    print(f"Кол-во документов с имитацией стиля: {style_imitation_count['with_style_imitation']}; без имитаций: {style_imitation_count['no_style_imitation']}; неприменимо: {style_imitation_count['nan']}")

    d_res = {"Кол-во документов": count_doc,
             "Кол-во уникальных текстов": len(uniq_hash_texts),
             "Длина текстов в токенах, min": min(doc_tokens_len),
             "Длина текстов в токенах, max": max(doc_tokens_len),
             "Длина текстов в токенах, mean": round(np.mean(doc_tokens_len),1),
             "Длина текстов в символах, min": min(doc_chars_len),
             "Длина текстов в символах, max": max(doc_chars_len),
             "Длина текстов в символах, mean": round(np.mean(doc_chars_len),1),
             "Длина текстов в предложениях, min": min(doc_sents_len),
             "Длина текстов в предложениях, max": max(doc_sents_len),
             "Длина текстов в предложениях, mean": round(np.mean(doc_sents_len),1),
             "Кол-во уникальных токенов, словоформ": len(uniq_token_forms),
             "Кол-во уникальных токенов, лемм": len(uniq_token_lemms),
             "Кол-во документов написанных женщинами": genders_count['female'],
             "Кол-во документов написанных мужчинами": genders_count['male'],
             "Кол-во уникальных авторов": len(uniq_authors),
             "Кол-во уникальных авторов женщин": uniq_authors_count['female'],
             "Кол-во уникальных авторов мужчин": uniq_authors_count['male'],
             "Возраст авторов написанных документов, min": min(ages),
             "Возраст авторов написанных документов, max": max(ages),
             "Возраст авторов написанных документов, mean": round(np.mean(ages),1),
             "Кол-во документов по возрастным группам авторов, 0-19": age_groups_count['0-19'],
             "Кол-во документов по возрастным группам авторов, 20-29": age_groups_count['20-29'],
             "Кол-во документов по возрастным группам авторов, 30-39": age_groups_count['30-39'],
             "Кол-во документов по возрастным группам авторов, 40-49": age_groups_count['40-49'],
             "Кол-во документов по возрастным группам авторов, 50+": age_groups_count['50+'],
             "Кол-во документов с имитацией": no_imitation_count['with_any_imitation'],
             "Кол-во документов без имитации": no_imitation_count['no_any_imitation'],
             "Кол-во документов с имитацией пола": gender_imitation_count['with_gender_imitation'],
             "Кол-во документов без имитации пола": gender_imitation_count['no_gender_imitation'],
             "Кол-во документов с имитацией пола, неприменимо": gender_imitation_count['nan'],
             "Кол-во документов с имитацией возраста, younger": age_imitation_count['younger'],
             "Кол-во документов с имитацией возраста, older": age_imitation_count['older'],
             "Кол-во документов без имитации возраста, no_age_imitation": age_imitation_count['no_age_imitation'],
             "Кол-во документов с имитацией возраста, неприменимо": age_imitation_count['nan'],
             "Кол-во документов с имитацией стиля": style_imitation_count['with_style_imitation'],
             "Кол-во документов без имитацией стиля": style_imitation_count['no_style_imitation'],
             "Кол-во документов с имитацией стиля, неприменимо": style_imitation_count['nan']}
    return d_res


if __name__ == "__main__":
    # configs
    args_parser = ap.ArgumentParser()
    args_parser.add_argument("path", nargs="+")
    args_parser.add_argument("--res-csv", default=None, help="Путь для сохранения статистики в виде таблицы")
    args = args_parser.parse_args()
    
    l_res = []
    for inp_ds in args.path:
        # read data
        df = load_jsonl(inp_ds)

        # calc
        d_ds_stat = calc_statistics(df)
        d_ds_stat["ds"] = inp_ds
        l_res.append(d_ds_stat)

    if len(l_res) > 1 or args.res_csv is not None:
        df_res = pd.DataFrame(l_res)
        print(df_res)
        if args.res_csv is not None:
            df_res.T.to_csv(args.res_csv, sep=";")