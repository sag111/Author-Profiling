# Функции для сохранения и загрузки векторизованного корпуса в формате hdf5
import os.path

import numpy as np
import h5py
from math import sqrt


DT_VLEN_FLOAT = h5py.vlen_dtype(np.float32)
DT_VLEN_INT = h5py.vlen_dtype(np.int32)


def save_ds(x1, x2, y, res_file_name, dtype_x1=DT_VLEN_FLOAT, dtype_x2=DT_VLEN_INT):
    """
    Сохранение корпуса в формате hdf5, сохранение произовдится в один файл,
    в ключ x1_flatten и x2_flatten сохранются массивы x1 и x2,
    при этом массив закодированных слов представлен как одномерный массив произвольной длины.
    Функция возвращает размерность для слова в массиве x1.

    Parameters
    ----------
    x1: np.ndarray
        Массив документов, каждый документ предствален массивом закодированных слов
    x2: np.ndarray
        Квадратная матрица смежности слов по синтаксическому дереву
    y: np.ndarray
        Массив метод для документов
    res_file_name: str
        Путь до файла для сохранения корпуса
    dtype_x1: h5py.dtype
        Тип данных для векторов X1, по стандарту стоит h5py.vlen_dtype(np.float32) -- для векторов переменной длины,
        для векторов фиксированной длины подходит np.float
    dtype_x2: h5py.dtype
        Тип данных для векторов X2, по стандарту стоит h5py.vlen_dtype(np.int32) -- для векторов переменной длины,
        для векторов фиксированной длины подходит np.float

    Returns
    -------
    word_dim: int
        Размерность для слова из x1
    """

    word_dim = x1[0].shape[1]
    with h5py.File(res_file_name, "w") as f:
        x1_flatten = [val.flatten() for val in x1]
        x2_flatten = [val.flatten() for val in x2]
        f.create_dataset("x1_flatten", data=x1_flatten, dtype=dtype_x1, compression="gzip")
        f.create_dataset("x2_flatten", data=x2_flatten, dtype=dtype_x2, compression="gzip")
        f.create_dataset("y", data=y)
    return word_dim


def read_ds(inp_file_name, word_dim=-1):
    """
    Считывает корпус из файла hdf5 и возвращает x1, x2, y в формате numpy массивов.

    Parameters
    ----------
    inp_file_name: str
        Путь до файла с корпусом в формате hdf5
    word_dim: int
        Размерность для слова, если -1, то вычисляется на основе квадратной матрицы из x2

    Returns
    -------
    x, y: np.ndarrays
        Возвращает массивы x, y, где каждый элемент x -- это кортеж из элемента x1 и x2
    """

    with h5py.File(os.path.normpath(inp_file_name), "r") as f:
        x1 = []
        x2 = []
        for x1_val, x2_val in zip(f["x1_flatten"], f["x2_flatten"]):
            n_words_in_doc = int(sqrt(len(x2_val)))
            x1.append(x1_val.reshape((n_words_in_doc, word_dim)))
            x2.append(x2_val.reshape((n_words_in_doc, n_words_in_doc)))
        x1 = np.array(x1)
        x2 = np.array(x2)
        y = np.array(f["y"])
    x = np.array(list(zip(x1, x2)))
    # y = np.array(y)
    return x, y