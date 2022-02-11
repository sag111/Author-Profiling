# Базовые методы для моделей
from abc import ABC, abstractmethod
from math import floor
from sklearn.base import BaseEstimator

import numpy as np
import os
import json
import tempfile
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *

from src.models.Transformer import EncoderLayer
from src.data.store import read_ds
from src.models.Data import positional_encoding
from src.models.Transformer import Encoder, EncoderAW


class BaseModel(ABC):
    @abstractmethod
    def fit(X, y, **kwargs):
        pass
    
    @abstractmethod
    def predict(X, **kwargs):
        pass
    

class DummyModel(BaseModel):
    
    name = "DummyModel"
    
    def __init__(self, test_range_int, test_range_float, test_range_int_step, test_choice_string, dummy_filler = 0, classes_nb = 2):
        '''Пример класса модели для использования с hyperopt.
        Может служить в качестве основы для абстрактного класса модели.
        На входе модели необходимы все те параметры, которые есть на выходе get_hyperparameters. 
        В данной модели они в основном служили для теста работы скриптов с гиперопт.
        Данная модель присваивает всем записям тот класс, который записан в dummy_filler.
        Для работы с нашей архитектурой модель должна иметь методы fit и predict, predict возвращает вероятности классов для каждого примера (стандартный выход нейронок).
        Для сохранения логов у модели также должно быть имя в поле name.'''
        
        self.dummy_filler = dummy_filler
        self.classes_nb = classes_nb
          
    def model_build(self):
        pass
        
    def fit(self, X, y):
        pass
        
    def predict(self, X):
        if self.dummy_filler == 0:
            return [[1,0] for cur_x in X]
        else:
            return [[0,1] for cur_x in X]
            
    
    @staticmethod
    def get_hyperparameters():
        """Функция, которая возвращает словарь с информацией о пространстве поиска. 
        Для каждого признака, входящего в пространство поиска, необходимо сформировать словарь, в котором будут следующие ключи:
        borders - list или tuple. Содержит информацию о границах пространства поиска для данного параметра.
        type - str: choice или range. 
            Если choice -- hyperopt будет выбирать из значений списка в borders.
            Если range -- hyperopt будет воспринимать borders как границы поискового пространства и рассматривать все значения между ними.
        dtype - в случае с type=range влияет на то, будут возвращены целые или действительные числа.
        step - int - в случае с type=range позволяет установить промежуточный шаг между границами.
        
        
        """
        return {"dummy_filler": {"borders": (0,1, "error"), 
                                 "type": "choice", 
                                 "dtype": int,
                                 "step": 1},
    
                "test_range_int": {"borders": (0,10), 
                                 "type": "range", 
                                 "dtype": int,
                                 "step": 1},
            
                "test_range_float": {"borders": (0.,10.), 
                                 "type": "range", 
                                 "dtype": float,
                                 "step": 0.5},
                    
                "test_range_int_step": {"borders": (16,32), 
                                 "type": "range", 
                                 "dtype": int,
                                 "step": 8},
                    
                "test_choice_string": {"borders": ["adam", "adagrad", "sgd"], 
                                 "type": "choice", 
                                 "dtype": str}}


def cyclic_learning_rate(epoch, lr, max_lr, base_lr, step_size):
    clr_step = floor(1 + epoch / step_size)
    if clr_step % 2 != 0:
        lr = lr + (max_lr - base_lr) / step_size
    else:
        lr = lr - (max_lr - base_lr) / step_size
    return lr


class GAModel(BaseEstimator):

    name = "GAModel"

    def __init__(self, word_dim, y_dim, neurons_1_multi=64, neurons_2=128, n_blocks=2, n_heads=2, dropout=0., use_residual=True,
                 add_positional_encoding=False, maximum_positions_encoding=3000, no_synt=False, x_const=False,
                 batch_size=32,  patience=30, nb_epochs=300, optimizer="Adam", loss_function="categorical_crossentropy",
                 learning_rate=0.001, use_clr=False, clr_step_size=None, clr_max_learning_rate=None,
                 lstm_reduce=False, lstm_size=32, data_unzipped=False, save_only_best_weights=False, **kwargs):
        """
        Класс для Graph Attention модели.
        Каждый пример должен быть  представлен кортежем (матрица закодированных слов, матрица смежности).
        При обучении и предсказании делается заполнение пустыми значениями всего набора данных по максимальной длине документа.
        Лучше очень длинные документы исключать из набора или разбивать на более короткии или обрезать до обучения модели.

        Parameters
        ----------
        word_dim: int
            Размерность вектора признаков для слова
        neurons_1_multi: int
            Множитель для n_heads, определет число нейронов для multi-head attention
        neurons_2: int
            Число нейронов в скрытом слое с relu блока multi-head attention
        n_blocks: int
            Число блоков в сети
        n_heads: int
            Число голов в multi-head attention
        dropout: float
            Dropout rate в каждом блоке
        use_residual: bool
            Флаг, если True -- то использовать проброс входа на выход
        y_dim: int
            Размерность выходного слоя
        add_positional_encoding: bool
            Флаг, надо ли добавлять кодирование последовательной позиции слова в тексте
        maximum_positions_encoding: int
            Максимальная длина последовательности, используется только при add_positional_encoding=True
        no_synt: bool
            Флаг, если True, то не используется синтаксис, т.е. просто MulitHeadAttention модель получается.
        x_const: bool
            Флаг, елси True, то все значения векторов слов будут заменены на константу,
            константа берётся как среднее по всем значнеями векторо слов в корпусе
        batch_size: int
            Размер батча при обучении и предсказании
        patience: int
            Число эпох до раннего останова, когда растёт ошибка на валидационном множестве
        lstm_reduce: bool
            Флаг, если True -- то перед выходом ставится LSTM, если False -- то используется классификация по первому токену
        lstm_size: int
            Размер LSTM слоя перед выходом, если он есть
        data_unzipped: bool
            Флаг, если True -- предполагается, что unzip и padding x-ов был произведён вне модели и затем x1 и x2 были объединены в кортеж.
        
        """

        self.neurons_1_multi = neurons_1_multi
        self.neurons_2 = neurons_2
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.dropout = dropout
        self.use_residual = use_residual
        self.word_dim = word_dim
        self.y_dim = y_dim
        self.add_positional_encoding = add_positional_encoding
        self.maximum_positions_encoding = maximum_positions_encoding
        self.no_synt = no_synt
        self.x_const = x_const
        self.batch_size = batch_size
        self.patience = patience
        self.nb_epochs = nb_epochs
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.use_clr = use_clr
        self.clr_step_size =clr_step_size
        self.clr_max_learning_rate = clr_max_learning_rate
        self.lstm_reduce = lstm_reduce
        self.lstm_size = lstm_size
        self.mean_x_const = None
        self.model = self._get_model()
        self.data_unzipped = data_unzipped
        self.save_only_best_weights = save_only_best_weights

    def _get_model_block(self):
        """Функция для формирования одного блока модели"""

        d_model = self.neurons_1_multi * self.n_heads
        inp_w = Input((None, d_model))
        inp_s = Input((None, None))
        if self.no_synt == False:
            mlh_synt = Dot(axes=(1, 2))([inp_w, inp_s])
            mlh_synt = Permute((2, 1))(mlh_synt)
        else:
            mlh_synt = inp_w
        mlh_mla = EncoderLayer(d_model=d_model, num_heads=self.n_heads, dff=self.neurons_2, rate=self.dropout)(mlh_synt, mask=None)

        if self.use_residual:
            mlh_out = Add()([inp_w, mlh_mla])
            mlh_out = BatchNormalization()(mlh_out)
        else:
            mlh_out = mlh_mla

        mlh_out = Dropout(self.dropout)(mlh_out)
        model = Model((inp_w, inp_s), mlh_out)
        return model

    def _get_model(self):
        """Функция для формирования модели из отдельных блоков"""

        inp_word_vects = Input((None, self.word_dim), dtype=tf.float32)
        inp_synt = Input((None, None), dtype=tf.float32)

        d_model = self.neurons_1_multi * self.n_heads
        mlh_words = TimeDistributed(Dense(d_model, activation="linear"))(inp_word_vects)

        if self.add_positional_encoding:
            seq_len = tf.shape(mlh_words)[1]
            mlh_position_encoding = positional_encoding(self.maximum_positions_encoding, d_model=d_model)
            mlh_words += mlh_position_encoding[:, :seq_len, :]

        mlh = mlh_words
        for i in range(self.n_blocks):
            model_block = self._get_model_block()
            mlh = model_block((mlh, inp_synt))

        if self.lstm_reduce:
            ml_out = Bidirectional(LSTM(self.lstm_size))(mlh)
        else:
            ml_out = Lambda(lambda x: x[:, 0])(mlh)
        ml_out = Dense(self.y_dim, activation="softmax")(ml_out)

        model = Model((inp_word_vects, inp_synt), ml_out)
        opt = tf.keras.optimizers.get({"class_name": self.optimizer, "config":{"learning_rate": self.learning_rate}})
        model.compile(opt, self.loss_function, metrics=["acc"])
        return model

    @staticmethod
    def pad_adj_matrix(inp, max_len):
        """Функция для заполнения нулями до максимальной длины синтаксической матрицы смежности"""

        res = []
        for doc in inp:
            doc_len = doc.shape[0]
            paded_len = max_len - doc_len
            padded_doc = np.pad(doc, ((paded_len, 0), (paded_len, 0)),
                                mode="constant", constant_values=0)
            res.append(padded_doc)
        res = np.array(res, dtype = np.uint8)
        return res

    def _unzip_x(self, x):
        """Функция для преобразования X в нужный для модели формат"""
        
        x1 = [val[0] for val in x]
        x2 = [val[1] for val in x]
        
        if not self.data_unzipped:
            if self.x_const:
                x1 = self._replace_x_by_mean(x1)
            x1 = np.array(x1)
            x2 = np.array(x2)
            x1 = tf.keras.preprocessing.sequence.pad_sequences(x1, dtype="float32")
            x2 = self.pad_adj_matrix(x2, len(x1[0]))
        return np.array(x1), np.array(x2)

    def _replace_x_by_mean(self, x):
        if self.mean_x_const is None:
            mean_vect = np.zeros((self.word_dim, ))
            for doc in x:
                doc_mean = np.mean(doc, axis=0)
                mean_vect += doc_mean
            mean_vect = mean_vect/float(len(x))
            self.mean_x_const = mean_vect
        else:
            mean_vect = self.mean_x_const
        res = []
        for doc in x:
            doc_mean = np.array([mean_vect, ]*len(doc))
            res.append(doc_mean)
        res = np.array(res)
        return res

    def fit(self, X, y, valid_data=None, nb_epochs=None, batch_size=None, early_stopping_patience=None, verbose=0,
            use_wandb=False, wandb_project="GraphHyperOpt", wandb_name="ga-model", wandb_additional_config=None,
            wandb_exclude_keys=None):
        """
        Обучение модели.
        В этой функции делается заполнение нулями до максимальной длины матриц X,
        максимальная длина рассчитывается по всему корпусу.

        Parameters
        ----------
        X: list
            Входной массив примеров, каждый пример представлен как кортеж из двух элементов:
            1. матрицы с векторами слов
            2. матрицы смежности слов
        y: list
            Массив меток для документов в унитарном коде (OneHot)
        valid_data: tuple
            Кортеж, валидационные данные (X_valid, y_valid)
        nb_epochs: int
            Максимальное число эпох
        batch_size: int
            Размер батчк
        early_stopping_patience: int
            Число эпох до раннего останова, после роста ошибки на валидационном множестве
        verbose: int
            Уровень выдачи логов на экран
        use_wandb: bool
            Флаг, если True, будет вестись логирование в Weights & Biases
        wandb_project: str
            Название проекта в wandb
        wandb_name: str
            Название запуска в wandb
        wandb_additional_config: dict
            Дополнительные параметры для сохранения в логи
        wandb_exclude_keys: list
            Список ключей исключений -- не для логирования

        Returns
        -------
        h: tensorflow.keras.History
            Объект History после обучения модели Keras
        """

        # Сохраняем параметры запуска в wandb
        # Мне как-то не очень нравится импортировать пакет здесь, надо подумать над этим
        if use_wandb:
            import wandb
            from wandb.keras import WandbCallback

            if wandb_exclude_keys is None:
                wandb_exclude_keys = []

            wandb_config = {}
            for k, v in self.__dict__.items():
                if k not in wandb_exclude_keys:
                    if isinstance(v, (str, int, float)):
                        wandb_config[k] = v

            if wandb_additional_config is not None:
                wandb_config.update(wandb_additional_config)
            wandb.init(project=wandb_project, name=wandb_name, config=wandb_config)

        if batch_size is None:
            batch_size = self.batch_size
        if early_stopping_patience is None:
            early_stopping_patience = self.patience
        if nb_epochs is None:
            nb_epochs = self.nb_epochs
        
        
        x1, x2 = self._unzip_x(X)

        if valid_data is not None:
            x1_valid, x2_valid = self._unzip_x(valid_data[0])
            model_valid_data = ((x1_valid, x2_valid), valid_data[1])
        else:
            model_valid_data = None

        with tempfile.NamedTemporaryFile() as f:
            model_checkpoint_tmp_file = f.name
            print("Model tmp file_name: {0}".format(model_checkpoint_tmp_file))

            if self.use_clr:
                def model_cyclic_lr(epoch, lr):
                    return cyclic_learning_rate(epoch, lr, max_lr=self.clr_max_learning_rate,
                                                base_lr=self.learning_rate, step_size=self.clr_step_size)
                clr_callback = LearningRateScheduler(model_cyclic_lr)

            callbacks = [EarlyStopping(patience=early_stopping_patience),
                         ModelCheckpoint(model_checkpoint_tmp_file, save_weights_only=True)]
            if use_wandb:
                callbacks.append(WandbCallback(save_graph=True, save_model=False))
            if self.use_clr:
                callbacks.append(clr_callback)
            h = self.model.fit((x1, x2), y, validation_data=model_valid_data, epochs=nb_epochs,
                               verbose=verbose, batch_size=batch_size,
                               callbacks=callbacks)
            self.model.load_weights(model_checkpoint_tmp_file)

        return h

    def predict(self, X, batch_size=32):
        """
        Функция для получения выхода модели,
        возвращает вектора с активностями для каждого класса
        Parameters
        ----------
        X: list
            Вход, состоит их кортежа (матрица векторов слов, матрица смежности)
        batch_size: int
            Размер батча

        Returns
        -------
        res: np.ndarray
            Двухмерный массив векторов активностей размерность:(число документов, число классов)
        """

        x1, x2 = self._unzip_x(X)
        pred = self.model.predict((x1, x2), batch_size=batch_size)
        return pred

    def save_model(self, res_dir):
        """
        Функция для сохранения модели, для дальнейшего использования.
        Parameters
        ----------
        res_dir: str
            Путь до директории для сохранения
        """

        os.makedirs(res_dir, exist_ok=True)
        res_hyperparams = {}
        for var_name, var_val in self.__dict__.items():
            if var_name == "model":
                var_val.save_weights(os.path.join(res_dir, "model.h5"))
            else:
                res_hyperparams[var_name] = var_val
        with open(os.path.join(res_dir, "hyperparams.json"), "w") as f:
            json.dump(res_hyperparams, f)

    @staticmethod
    def load_model(inp_dir):
        """
        Загрузка готовой модели.

        Parameters
        ----------
        inp_dir: str
            Путь до директории с готовой моделью

        Returns
        -------
        model: GAModel
            Возвращает загруженную модель
        """

        with open(os.path.join(inp_dir, "hyperparams.json"), "r") as f:
            hypermarams = json.load(f)
        self_obj = GAModel(**hypermarams)
        for param_name, param_val in hypermarams.items():
            self_obj.__setattr__(param_name, param_val)
        self_obj.model = self_obj._get_model()
        self_obj.model.load_weights(os.path.join(inp_dir, "model.h5"))
        return self_obj


class ModelNN(GAModel):

    name = "ModelNN"

    def __init__(self, word_dim, y_dim, n_conv_blocks=3, conv_neurons=128, conv_win=3,
                 max_pooling_win=3, lstm_neurons=128, dropout_rate=0.5, x_const=False, batch_size=32, patience=30,
                 nb_epochs=300, optimizer="Adam", loss_function="categorical_crossentropy", learning_rate=0.001,
                 use_clr=False, clr_step_size=None, clr_max_learning_rate=None, save_only_best_weights = False, **kwargs):
        self.word_dim = word_dim
        self.y_dim = y_dim
        self.n_conv_blocks = n_conv_blocks
        self.conv_neurons = conv_neurons
        self.conv_win = conv_win
        self.max_pooling_win = max_pooling_win
        self.lstm_neurons = lstm_neurons
        self.dropout_rate = dropout_rate
        self.x_const = x_const
        self.batch_size = batch_size
        self.patience = patience
        self.nb_epochs = nb_epochs
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.use_clr = use_clr
        self.clr_step_size = clr_step_size
        self.clr_max_learning_rate = clr_max_learning_rate
        self.model = self._get_model()
        self.data_unzipped = kwargs.get("data_unzipped", False)
        self.save_only_best_weights = save_only_best_weights

    def _get_conv_block(self, word_dim=None):
        inp = Input((None, word_dim), dtype=tf.float32)
        x_cb = Conv1D(self.conv_neurons, (self.conv_win,), padding="same", activation="relu")(inp)
        x_cb = MaxPooling1D((self.max_pooling_win,), padding="same")(x_cb)
        model_cnn = Model(inp, x_cb)
        return model_cnn

    def _get_model(self):
        inp_1 = Input((None, self.word_dim), dtype=tf.float32)
        inp_2 = Input((None, None), dtype=tf.float32)

        for conv_block_ind in range(self.n_conv_blocks):
            if conv_block_ind == 0:
                word_dim = self.word_dim
                ml_h_cnn = self._get_conv_block(word_dim=word_dim)
                ml_h = ml_h_cnn(inp_1)
            else:
                word_dim = self.conv_neurons
                ml_h_cnn = self._get_conv_block(word_dim=word_dim)
                ml_h = ml_h_cnn(ml_h)

        ml_h = LSTM(self.lstm_neurons)(ml_h)
        ml_h = Dropout(self.dropout_rate)(ml_h)
        ml_out = Dense(self.y_dim, activation="softmax")(ml_h)
        model = Model((inp_1, inp_2), ml_out)
        model.compile("adam", "mse", metrics=["acc"])
        return model

    @staticmethod
    def load_model(inp_dir):
        """
        Загрузка готовой модели.

        Parameters
        ----------
        inp_dir: str
            Путь до директории с готовой моделью

        Returns
        -------
        model: GAModel
            Возвращает загруженную модель
        """

        with open(os.path.join(inp_dir, "hyperparams.json"), "r") as f:
            hypermarams = json.load(f)
        self_obj = ModelNN(**hypermarams)
        for param_name, param_val in hypermarams.items():
            self_obj.__setattr__(param_name, param_val)
        self_obj.model = self_obj._get_model()
        self_obj.model.load_weights(os.path.join(inp_dir, "model.h5"))
        return self_obj


class GABERTModel(GAModel):
    def __init__(self, bert_weights, word_dim, neurons_1_multi, neurons_2, n_blocks, n_heads, dropout, use_residual,
                 y_dim, add_positional_encoding=False, maximum_positions_encoding=3000, no_synt=False, x_const=False,
                 batch_size=32,  patience=30, nb_epochs=300, optimizer="Adam", loss_function="categorical_crossentropy",
                 learning_rate=0.001, use_clr=False, clr_step_size=None, clr_max_learning_rate=None,
                 lstm_reduce=False, lstm_size=32, num_layer=12, vocab_size=119547, pad_data=False, pad_value=0):
        self.neurons_1_multi = neurons_1_multi
        self.neurons_2 = neurons_2
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.dropout = dropout
        self.use_residual = use_residual
        self.word_dim = word_dim
        self.y_dim = y_dim
        self.add_positional_encoding = add_positional_encoding
        self.maximum_positions_encoding = maximum_positions_encoding
        self.no_synt = no_synt
        self.x_const = x_const
        self.batch_size = batch_size
        self.patience = patience
        self.nb_epochs = nb_epochs
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.use_clr = use_clr
        self.clr_step_size = clr_step_size
        self.clr_max_learning_rate = clr_max_learning_rate
        self.lstm_reduce = lstm_reduce
        self.lstm_size = lstm_size

        self.bert_weights = bert_weights
        self.vocab_size = vocab_size
        self.pad_data = pad_data
        self.pad_value = pad_value
        if self.pad_data and self.pad_value is None:
            raise Exception("Если указано pad_data=True, надо указать pad_value")
        self.bert_model = Encoder({}, num_layers=num_layer, d_model=768, num_heads=12, dff=3072,
                                  maximum_position_encoding=1000,
                                  input_vocab_size=self.vocab_size, BERT_weights_dir=self.bert_weights)

        self.model = self._get_model()

    def _get_model(self):
        """Функция для формирования модели из отдельных блоков"""

        inp_word_vects = Input((None, ), dtype=tf.float32)
        inp_synt = Input((None, None), dtype=tf.float32)

        d_model = self.neurons_1_multi * self.n_heads
        mlh_words = self.bert_model(inp_word_vects, mask=None)
        mlh_words = TimeDistributed(Dense(d_model, activation="linear"))(mlh_words)

        if self.add_positional_encoding:
            seq_len = tf.shape(mlh_words)[1]
            mlh_position_encoding = positional_encoding(self.maximum_positions_encoding, d_model=d_model)
            mlh_words += mlh_position_encoding[:, :seq_len, :]

        mlh = mlh_words
        for i in range(self.n_blocks):
            model_block = self._get_model_block()
            mlh = model_block((mlh, inp_synt))

        if self.lstm_reduce:
            ml_out = Bidirectional(LSTM(self.lstm_size))(mlh)
        else:
            ml_out = Lambda(lambda x: x[:, 0])(mlh)
        ml_out = Dense(self.y_dim, activation="softmax")(ml_out)

        model = Model((inp_word_vects, inp_synt), ml_out)
        opt = tf.keras.optimizers.get({"class_name": self.optimizer, "config": {"learning_rate": self.learning_rate}})
        model.compile(opt, self.loss_function, metrics=["acc"])
        return model

    def _unzip_x(self, x):
        """Функция для преобразования X в нужный для модели формат"""

        x1 = [np.array(val[0]).reshape((-1, )) for val in x]
        x2 = [val[1] for val in x]
        if self.x_const:
            x1 = self._replace_x_by_mean(x1)
        x1 = np.array(x1)
        x2 = np.array(x2)
        if self.pad_data:
            x1 = tf.keras.preprocessing.sequence.pad_sequences(x1, dtype="float32", value=self.pad_value)
            x2 = self.pad_adj_matrix(x2, len(x1[0]))
        return x1, x2
