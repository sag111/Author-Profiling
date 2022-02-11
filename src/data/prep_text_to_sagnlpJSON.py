import os
import json
import stanza
import tqdm


def load_jsonl(input_path: str) -> list:
    """
    Read list of objects from a JSON lines file.
    """
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.rstrip('\n|\r')))
    print('Loaded {} records from {}'.format(len(data), input_path))
    
    return data


def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in data:
            json_record = json.dumps(line, ensure_ascii=False)
            f.write(json_record + '\n')
    print('Wrote {} records to {}'.format(len(data), output_path))


class Stanza_prepare():
    def __init__(self, config={}):
        self.model = stanza.Pipeline(**config) # Initialize the pipeline using a configuration dict
        
    def text_to_sagnlpJSON(self, text: str) -> dict:
        doc = self.model(text).to_dict()
        
        sentences = []
        for sent in doc:
            sentence = []
            for word in sent:
                sample = {"forma": word["text"], "lemma": word["lemma"], "pos": word["upos"],
                          "posStart": word["start_char"],"len": len(word["text"]), "dom": word["head"]}
                if "feats" in word:
                    sample["grm"] = word["feats"]
                else:
                    sample["grm"] = ""
                
                sentence.append(sample)
            sentences.append(sentence)
        
        return {"text": text, "sentences": sentences}


DEFAULT_RES_DIR = os.path.split(os.path.split(os.path.split(os.path.abspath(__file__))[0])[0])[0]


if __name__ == "__main__":
    # Configs
    DATA_PATH = "{0}/data/raw/".format(DEFAULT_RES_DIR)
    SAVE_PATH = "{0}/data/preprocess/".format(DEFAULT_RES_DIR)

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    stanza_config = {
        # Comma-separated list of processors to use
        'processors': 'tokenize,pos,lemma,depparse',

        # Language code for the language to build the Pipeline in
        'lang': 'ru',

        # Use pretokenized text as input and disable tokenization
        'tokenize_pretokenized': False
    }

    stz = Stanza_prepare(config=stanza_config)

    for file_name in ["train.jsonl", "valid.jsonl", "test.jsonl"]:
        data = load_jsonl(DATA_PATH+file_name)

        with tqdm.tqdm(total=len(data)) as pbar:
            prepare_data = []
            for doc in data:
                sample = stz.text_to_sagnlpJSON(doc["text"])

                sample["meta"] = doc.get("meta", {})
                for feat in ["id", "account_id", "author_id", "age", "age_group", "gender",
                             "no_imitation", "age_imitation", "gender_imitation", "style_imitation"]:
                    sample[feat] = doc[feat]

                prepare_data.append(sample)
                pbar.update(1)

        dump_jsonl(prepare_data, SAVE_PATH+"prep_"+file_name)
