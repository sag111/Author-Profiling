# Различные способы векторизации документов

import numpy as np
from math import log

from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.base import BaseEstimator
import gensim

from src.data.corpus import WordsCorpusIterator


class OneHotMorphWordVectorizer(BaseEstimator):
    """
    Векторизатор для слова на основе морфологии.
    Возвращает бинарный вектор из 0 и 1,
    признаками векторая являются уникальные значения кждого тэга.

    Parameters
    ----------
    exclude_tags: list, tuple
        Список игнорируемых тэгов

    Examples
    --------
    Пример:
    Текст и морфологический разбор
    Я           Case=Nom|Number=Sing|Person=1
    студентка   Animacy=Anim|Case=Nom|Gender=Fem|Number=Sing
    МТУ         Animacy=Inan|Case=Gen|Gender=Neut|Number=Sing

    Векторизаця слов:
    | Признак/слова   | Я | студентка | МТУ |
    |-----------------|---|-----------|-----|
    | Case=Nom        | 1 | 1         | 0   |
    | Case=Gen        | 0 | 0         | 1   |
    | Number=Sing     | 1 | 1         | 1   |
    | Person=1        | 1 | 0         | 0   |
    | Gender=Fem      | 0 | 1         | 0   |
    | Gender=Neut     | 0 | 0         | 1   |
    | Animacy=Inan    | 0 | 0         | 1   |
    | Animacy=Anim    | 0 | 1         | 0   |
    """

    def __init__(self, exclude_tags=(), grm_sep="|", text_element=None):
        self.exclude_tags = exclude_tags
        self.grm_sep = grm_sep
        self.text_element = text_element
        self.d_tags = {None: 0}
        self.l_tags = [None, ]
        self.l_pos_inds = []
        self.dim = 0

    def _get_morph_tags(self, word):
        """
        Возвращает тэги (grm и pos) для слова в виде списка
        Parameters
        ----------
        word: dict
            Слово в формате sagnlpJSON

        Returns
        -------
        list_tags: list
            Список тэгов для слова
        """
        w_pos = [word["pos"]]
        w_grm = word["grm"].split(self.grm_sep)
        w_tags = w_pos + w_grm
        return w_tags

    def fit(self, inp):
        """
        Обучение векторизатора на корпусе.

        Parameters
        ----------
        inp: list
            Список документов в формате sagnlpJSON
        """

        set_tags = set()
        set_pos_tags = set()
        words_iter = WordsCorpusIterator(inp, text_element=self.text_element)
        for word in words_iter:
            w_tags = self._get_morph_tags(word)
            set_pos_tags.update([word["pos"], ])
            set_tags.update(w_tags)

        self.l_tags += sorted(list(set_tags))

        for tag_ind, tag in enumerate(self.l_tags):
            self.d_tags[tag] = tag_ind
            if tag in set_pos_tags:
                self.l_pos_inds.append(tag_ind)
        self.dim = len(self.d_tags)

        return self

    def transform(self, inp, pad_sequence=False, pad_value=0., pad_args=None, add_virtual_root=False):
        """
        Векторизация слов в корпусе.

        Parameters
        ----------
        inp: list
            Список документов в формате sganlpJSON

        pad_sequence: bool
            Флаг, нужно ли выравнивать все документы по длине

        pad_value: float, np.array
            Значени для заполнителя (pad_value)

        pad_args: dict
            Словарь аргументов для функции pad_sequence

        Returns
        -------
        res: list
            Список закодированных документов, каждый документ представлен как список закодированных слов
        """

        res = []
        for doc in inp:
            l_doc_words_vects = []

            if add_virtual_root:
                w_vect = np.ones((self.dim,), dtype=np.int)
                l_doc_words_vects.append(w_vect)

            words_iter = WordsCorpusIterator([doc, ], text_element=self.text_element)
            for word in words_iter:
                w_tags = self._get_morph_tags(word)
                w_tags_inds = [self.d_tags[tag] for tag in w_tags if tag in self.d_tags]
                w_vect = np.zeros((self.dim, ), dtype=np.int)
                w_vect[w_tags_inds] = 1
                l_doc_words_vects.append(w_vect)
            res.append(l_doc_words_vects)

        if pad_sequence:
            if pad_args is not None:
                d_pad_args = pad_args
            else:
                d_pad_args = {}
            res = pad_sequences(res, value=pad_value, dtype="float32", **d_pad_args)

        res = np.array([np.array(doc) for doc in res])

        return res

    def inverse_transform(self, encoded_words, empty_words=False):
        """
        Восстановление части речи и морфологических тэгов из вектора для слова.
        При восстановлении все значения в векторе больше 0.5 считаются 1, а меньше 0.

        Parameters
        ----------
        encoded_words: list
            Список документов, каждый документ представлен списком закодированных слов
        empty_words: bool
            Флаг, нужно ли включать слова без каких-либо морфологических признаков
            (все компоненты вектора для слова нулевые). Если True, то такие слова представлены: {"pos": "", "grm": ""}

        Returns
        -------
        res: list
            Список документов, каждый документ -- список слов. Слова представлены словарём с двумя ключами: pos и grm,
            аналогично оригинальному представлению в sagnlpJSON.
            Тэги в поле grm идёт в отсортированном по алфовиту порядке.
        """

        features_dim_inds = np.arange(0, self.dim)
        res = []
        for doc in encoded_words:
            l_doc_words = []
            for word in doc:
                d_word = {}
                l_grm = []
                w_features_inds = features_dim_inds[word > 0.5]
                if len(w_features_inds) > 0:
                    for val in w_features_inds:
                        if val in self.l_pos_inds:
                            d_word["pos"] = self.l_tags[val]
                        else:
                            l_grm.append(self.l_tags[val])
                    d_word["grm"] = self.grm_sep.join(l_grm)
                    l_doc_words.append(d_word)
                elif empty_words:
                    l_doc_words.append({"pos": "", "grm": ""})
            res.append(l_doc_words)

        return res


class FTVectorizer:
    def __init__(self, fasttext_model_path, model_type="gensim", text_element=None, word_field="forma", add_word_pos=False):
        """
        Векторизатор текстов на основе FastText моделей gensim

        Parameters
        ----------
        fasttext_model_path: str
            Путь до модели
        model_type: str {"gensim", "fasttext"}
            Тип модели:
            - "fasttext" -- бинарная, полученная оригинальным FastText,
            - "gensim" -- модель полученная gensim.
        text_element: str
            Текстовый элемент для анализа, появился в новой версии sagnlpJSON
        word_field: str {"lemma", "forma"}
            Поле слова, которое необходимо подавать в модель fasttext. По умолчанию forma
        add_word_pos: bool
            Добавлять ли часть речи к слову при подаче в модель fasttext. По умолчанию False
        
        """

        self.model_type = model_type
        self.fasttext_model_path = fasttext_model_path
        self.text_element = text_element
        self.word_field = word_field
        self.add_word_pos = add_word_pos

        if model_type == "gensim":
            self.model = gensim.models.fasttext.FastText.load(self.fasttext_model_path)
        elif model_type == "fasttext":
            self.model = gensim.models.fasttext.load_facebook_model(self.fasttext_model_path)
        else:
            raise Exception("Неизвестный тип модели")
        self.dim = self.model.vector_size

    def transform(self, inp, pad_sequence=False, pad_value=None, add_virtual_root=False):
        res = []
        vect_size = self.model.vector_size
        none_vector = np.zeros((vect_size, ), dtype=np.float)
        virtual_root_vector = np.ones((vect_size, ), dtype=np.float)
        for doc in inp:
            res_doc = []
            if add_virtual_root:
                res_doc.append(virtual_root_vector)

            words_iter = WordsCorpusIterator([doc, ], text_element=self.text_element)
            for word in words_iter:
                
                inp_word = word[self.word_field]
                
                if self.add_word_pos:
                    inp_word+="_"
                    inp_word+=word["pos"]
                
                try:
                    word_vect = self.model[inp_word]
                except:
                    word_vect = none_vector
                res_doc.append(word_vect)
            res_doc = np.array(res_doc)
            res.append(res_doc)

        if pad_sequence:
            if pad_value is None:
                pad_value = none_vector
            res = pad_sequences(res, dtype="float32", value=pad_value)

        res = np.array(res)

        return res


def get_synt_matrix(doc):
    n_words = 0
    for sent in doc["sentences"]:
        n_words += len(sent)
    res = np.zeros((n_words + 1, n_words + 1), dtype=np.int32)
    np.fill_diagonal(res, 1)

    word_ind = 1
    for sent in doc["sentences"]:
        sent_offset = word_ind
        for word in sent:
            word_dom = int(word["dom"])
            if word_dom > 0:
                res[word_ind, word_dom + sent_offset - 1] = 1
            else:
                res[word_ind, word_dom] = 1
            word_ind += 1
    return res


class TFIDFSyntMatrix:
    def __init__(self, use_lemma=True):
        self.use_lemma = use_lemma
        self.d_ngrams_idf = {}

    def _get_doc_ngrams(self, x):
        l_doc = []
        for sent in x["sentences"]:
            for word in sent:
                if self.use_lemma:
                    word_text = word["lemma"]
                else:
                    word_text = word["forma"]
                l_doc.append(word_text)
        res = []
        for i in range(0, len(l_doc) - 1):
            w1, w2 = l_doc[i], l_doc[i + 1]
            res_w1_w2 = (w1, w2)
            res.append(res_w1_w2)
        return res

    def fit(self, X):
        for doc_ind, doc in enumerate(X):
            doc_ngrams = self._get_doc_ngrams(doc)
            for (w1, w2) in doc_ngrams:
                n_gram_doc_inds = self.d_ngrams_idf.get((w1, w2), set())
                n_gram_doc_inds.add(doc_ind)
                self.d_ngrams_idf[(w1, w2)] = n_gram_doc_inds

        n_docs = float(len(X))
        for k, v in self.d_ngrams_idf.items():
            self.d_ngrams_idf[k] = log(n_docs/float(len(v)))

    def transform_doc(self, x):
        doc_ngrams = self._get_doc_ngrams(x)
        d_ngrams_tf = {}
        for n_gram in doc_ngrams:
            d_ngrams_tf[n_gram] = d_ngrams_tf.get(n_gram, 0.) + 1.

        l_doc_words = []
        for sent in x["sentences"]:
            for word in sent:
                if self.use_lemma:
                    word_text = word["lemma"]
                else:
                    word_text = word["forma"]
                l_doc_words.append(word_text)
        res = np.zeros((len(l_doc_words) + 1, len(l_doc_words) + 1))
        for w1_ind, w1 in enumerate(l_doc_words, 1):
            for w2_ind, w2 in enumerate(l_doc_words, 1):
                if (w1, w2) in self.d_ngrams_idf:
                    res[w1_ind, w2_ind] += d_ngrams_tf.get((w1, w2), 0.) * self.d_ngrams_idf.get((w1, w2), 0.)
        np.fill_diagonal(res, 1)
        return res

    def transform(self, X):
        res = []
        for doc in X:
            res.append(self.transform_doc(doc))
        return res
