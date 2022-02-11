from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow_datasets as tfds
import tensorflow as tf

import os
import csv
import json
import time
import unicodedata
from selectolax.parser import HTMLParser
import numpy as np
import pandas as pd


def GetPreprocessFunctions(tokenizer, maxLen, limitStrategy="Filter"):
    def cutToMaxLen(doc, summary,  maxLen=maxLen):
        docSize = len(doc)
        summarySize = len(summary)
        return doc[:min(docSize, maxLen)], summary[:min(summarySize, maxLen)]

    def encode(doc, summary):
        
        if "BERT_tokenizer" in tokenizer.__dict__:
            doc = tokenizer.encode(doc.numpy())
            summary = tokenizer.encode(summary.numpy())
        elif type(tokenizer).__name__=="SentencePieceProcessor":
            doc = [tokenizer.bos_id()] + tokenizer.EncodeAsIds(doc.numpy()) + [tokenizer.eos_id()]
            summary = [tokenizer.bos_id()] + tokenizer.EncodeAsIds(summary.numpy()) + [tokenizer.eos_id()]
        else:
            startIdx, endIdx = tokenizer.vocab_size, tokenizer.vocab_size+1
            doc = [startIdx] + tokenizer.encode(
                    doc.numpy()) + [endIdx]

            summary = [startIdx] + tokenizer.encode(
                    summary.numpy()) + [endIdx]
        if limitStrategy == "Cut":
            doc, summary = cutToMaxLen(doc, summary)
        return doc, summary

    def filter_max_length(x, y, max_length=maxLen):
        return tf.logical_and(tf.size(x) <= max_length,
                                                    tf.size(y) <= max_length)
    return encode, filter_max_length

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                                                    np.arange(d_model)[np.newaxis, :],
                                                    d_model)
    
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
    pos_encoding = angle_rads[np.newaxis, ...]
        
    return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    
    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)
    
    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)
    
    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by 
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    
    return enc_padding_mask, combined_mask, dec_padding_mask

def loadDataset(dataConfig, loadTrain=True):
    if dataConfig["corpus"] == "gigaword":
        if loadTrain:
            train_examples = loadGigaword(dataConfig["train_x"], dataConfig["train_y"])
        dev_examples = loadGigaword(dataConfig["dev_x"], dataConfig["dev_y"])
        test_examples = loadGigaword(dataConfig["test_x"], dataConfig["test_y"])
    elif dataConfig["corpus"] == "lenta":
        if loadTrain:
            train_examples = loadLenta(dataConfig["train"])
        dev_examples = loadLenta(dataConfig["dev"])
        test_examples = loadLenta(dataConfig["test"])
    elif dataConfig["corpus"] == "ria":
        if loadTrain:
            train_examples = loadRia(dataConfig["train"], dataConfig)
        dev_examples = loadRia(dataConfig["dev"], dataConfig)
        test_examples = loadRia(dataConfig["test"], dataConfig)
    elif dataConfig["corpus"] == "gooppe_ria":
        if loadTrain:
            train_examples = loadGooppeRia(dataConfig["train"], dataConfig)
        dev_examples = loadGooppeRia(dataConfig["dev"], dataConfig)
        test_examples = loadGooppeRia(dataConfig["test"], dataConfig)
    else:
        raise ValueError("Dataset not specified")
    if loadTrain:
        return train_examples, dev_examples, test_examples
    else:
        return dev_examples, test_examples

def clear_text(text: str, rm_strong=True) -> str:
    selector = "strong"
    text = unicodedata.normalize("NFKD", text)
    text = text.replace("\n", " ")
    tree = HTMLParser(text)
    if rm_strong:
        for node in tree.css(selector):
            node.decompose()
    return tree.text().strip()

def loadRia(path, dataConfig):
    with open(path, "r") as f:
        lines = f.readlines()
    ria_json = [json.loads(x) for x in lines]
    def gen():
        for doc in ria_json:
            if dataConfig.get("clear", False):
                yield (clear_text(doc["text"]), clear_text(doc["title"]))
            else:
                yield (doc["text"], doc["title"])
    examples = tf.data.Dataset.from_generator(gen, output_shapes=((), ()), output_types=(tf.string, tf.string))
    return examples

def loadGooppeRia(path, dataConfig):
    def gen():
        with open(path) as file:
            reader = csv.reader(file, delimiter="\t")
            for line in reader:
                if dataConfig.get("clear", False):
                    yield (clear_text(line[0]), clear_text(line[1]))
                else:
                    yield (line[0], line[1])
    examples = tf.data.Dataset.from_generator(gen, output_shapes=((), ()), output_types=(tf.string, tf.string))
    return examples

def loadLenta(path):
    lentaDF = pd.read_csv(path)
    def gen():
        for ir, row in lentaDF.iterrows():
            yield (row["text"], row["title"])
    examples = tf.data.Dataset.from_generator(gen, output_shapes=((), ()), output_types=(tf.string, tf.string))
    return examples


def loadGigaword(path_src, path_tgt):
    with open(path_src, "r") as f:
        docs = f.readlines()
    with open(path_tgt, "r") as f:
        summs = f.readlines()
    docs = [s.strip() for s in docs]
    summs = [s.strip() for s in summs]

    def gen():
        for doc, summ in zip(docs, summs):
            yield (doc, summ)
    examples = tf.data.Dataset.from_generator(gen, output_shapes=((), ()), output_types=(tf.string, tf.string))
    return examples