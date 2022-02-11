# Модуль для чтения корпуса в формате sagnlpJSON в виде одной части или разбитого на несколько частей

import numpy as np


class WordsCorpusIterator:
    """
    Итератор (генератор) по словам корпуса
    """
    def __init__(self, inp_corpus, text_element=None):
        self.inp_corpus = inp_corpus
        self.text_element = text_element

    def __iter__(self):
        for doc in self.inp_corpus:
            if self.text_element is not None:
                doc = doc["text_elements"][self.text_element]
            for sent in doc["sentences"]:
                for word in sent:
                    yield word


class SentWordsCorpusIterator:
    """
    Итератор (генератор) по предложениям корпуса
    """
    def __init__(self, inp_corpus, text_element=None):
        self.inp_corpus = inp_corpus
        self.text_element = text_element

    def __iter__(self):
        for doc in self.inp_corpus:
            if self.text_element is not None:
                doc = doc["text_elements"][self.text_element]
            for sent in doc["sentences"]:
                yield sent


def balance_data(X, Y, strategy="undersampling", shuffle_data=True, random_state=None):
    """
    Балансировка множества.

    Parameters
    ----------
    X: list
        Список примеров для балансировки

    Y: list
        Список меток для примеров, используется при балансировке

    strategy: str, {"undersampling", "oversampling"}
        undersampling -- отбрасываем последние примеры в наиболее представительных классах до размера наименее представитьельного;
        oversampling -- случайным образом копируем примеры в малопредставительных классах до размера наиболее представительного.

    shuffle_data: boolean
        Если True, то перед балансировкой производится перемешивание примеров внутри класса.

    random_state: None, int
        Используется при случайном перемешивании и копировании примеров в oversampling.
    """

    d_classes_sample_inds = {}
    for sample_ind, y in enumerate(Y):
        if y not in d_classes_sample_inds:
            d_classes_sample_inds[y] = []
        d_classes_sample_inds[y].append(sample_ind)

    np_random = np.random.RandomState(random_state)
    if shuffle_data:
        for k, v in d_classes_sample_inds.items():
            v = np.array(v)
            np_random.shuffle(v)
            d_classes_sample_inds[k] = v

    class_lenses = [len(v) for v in d_classes_sample_inds.values()]
    max_class_len = max(class_lenses)
    min_class_len = min(class_lenses)

    balanced_d_classes_sample_inds = {}
    for k, v in d_classes_sample_inds.items():
        if strategy == "undersampling":
            balanced_d_classes_sample_inds[k] = v[:min_class_len]
        elif strategy == "oversampling":
            if len(v) < max_class_len:
                balanced_d_classes_sample_inds[k] = np.concatenate(
                    [v, np.array(v)[np_random.randint(0, len(v), max_class_len - len(v))]], axis=0)
            else:
                balanced_d_classes_sample_inds[k] = v

    res_x = []
    res_y = []
    for k, v in balanced_d_classes_sample_inds.items():
        for val_ind in v:
            res_x.append(X[val_ind])
            res_y.append(Y[val_ind])
    return res_x, res_y
