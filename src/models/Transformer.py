from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf

from itertools import cycle

DISTRIBUTED_LAYERS = False


def gelu(x):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
    x: float Tensor to perform activation.

    Returns:
    `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead) 
    but it must be broadcastable for addition.
    
    Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable 
                    to (..., seq_len_q, seq_len_k). Defaults to None.
        
    Returns:
        output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    
    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


def print_out(q, k, v):
    temp_out, temp_attn = scaled_dot_product_attention(
            q, k, v, None)
    print ('Attention weights are:')
    print (temp_attn)
    print ('Output is:')
    print (temp_out)


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, wq_weights=None, wk_weights=None, wv_weights=None, w_dense=None):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        # Загрузка предобученных весов

        # Default
        # kernel_initializer='glorot_uniform',
        # bias_initializer='zeros'

        if wq_weights is not None:
            wq_k = tf.initializers.Constant(wq_weights[0])
            wq_b = tf.initializers.Constant(wq_weights[1])
        else:
            wq_k = 'glorot_uniform'
            wq_b = 'zeros'

        if wk_weights is not None:
            wk_k = tf.initializers.Constant(wk_weights[0])
            wk_b = tf.initializers.Constant(wk_weights[1])
        else:
            wk_k = 'glorot_uniform'
            wk_b = 'zeros'

        if wv_weights is not None:
            wv_k = tf.initializers.Constant(wv_weights[0])
            wv_b = tf.initializers.Constant(wv_weights[1])
        else:
            wv_k = 'glorot_uniform'
            wv_b = 'zeros'

        if w_dense is not None:
            wd_k = tf.initializers.Constant(w_dense[0])
            wd_b = tf.initializers.Constant(w_dense[1])
        else:
            wd_k = 'glorot_uniform'
            wd_b = 'zeros'

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model, kernel_initializer=wq_k, bias_initializer=wq_b)
        self.wk = tf.keras.layers.Dense(d_model, kernel_initializer=wk_k, bias_initializer=wk_b)
        self.wv = tf.keras.layers.Dense(d_model, kernel_initializer=wv_k, bias_initializer=wv_b)

        self.dense = tf.keras.layers.Dense(d_model, kernel_initializer=wd_k, bias_initializer=wd_b)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff, l1_weights=None, l2_weights=None, inner_activation_type="relu"):
    if l1_weights is not None:
        l1_k = tf.initializers.Constant(l1_weights[0])
        l1_b = tf.initializers.Constant(l1_weights[1])
    else:
        l1_k = "glorot_uniform"
        l1_b = "zeros"

    if l2_weights is not None:
        l2_k = tf.initializers.Constant(l2_weights[0])
        l2_b = tf.initializers.Constant(l2_weights[1])
    else:
        l2_k = "glorot_uniform"
        l2_b = "zeros"

    if inner_activation_type == "relu":
        activation_inner = "relu"
    elif inner_activation_type == "gelu":
        activation_inner = gelu

    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation=activation_inner, kernel_initializer=l1_k, bias_initializer=l1_b),
        # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model, kernel_initializer=l2_k, bias_initializer=l2_b)  # (batch_size, seq_len, d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, BERT_layer_ind=None, BERT_weights_dir=None):
        super(EncoderLayer, self).__init__()

        if BERT_layer_ind is not None:
            wq_weights, wk_weights, wv_weights, w_dense = load_BERT_qkv_weights(BERT_layer_ind, BERT_weights_dir)
            l1_weights, l2_weights = load_BERT_feed_worward_weights(BERT_layer_ind, BERT_weights_dir)
            norm_weights = load_BERT_normalization_weigths(BERT_layer_ind, BERT_weights_dir)
            inner_activation_type = "gelu"
        else:
            wq_weights = None
            wk_weights = None
            wv_weights = None
            w_dense = None
            l1_weights = None
            l2_weights = None
            norm_weights = None
            inner_activation_type = "relu"

        self.mha = MultiHeadAttention(d_model, num_heads,
                                      wq_weights=wq_weights, wk_weights=wk_weights, wv_weights=wv_weights,
                                      w_dense=w_dense)
        self.ffn = point_wise_feed_forward_network(d_model, dff,
                                                   l1_weights=l1_weights, l2_weights=l2_weights,
                                                   inner_activation_type=inner_activation_type)

        if norm_weights is not None:
            w1 = norm_weights[0]
            wn_1_b = tf.initializers.Constant(w1[0])
            wn_1_g = tf.initializers.Constant(w1[1])
            w2 = norm_weights[1]
            wn_2_b = tf.initializers.Constant(w2[0])
            wn_2_g = tf.initializers.Constant(w2[1])
        else:
            wn_1_b = "zeros"
            wn_1_g = "ones"
            wn_2_b = "zeros"
            wn_2_g = "ones"

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, beta_initializer=wn_1_b,
                                                             gamma_initializer=wn_1_g)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, beta_initializer=wn_2_b,
                                                             gamma_initializer=wn_2_g)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)
 
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)
        
        
    def call(self, x, enc_output, training, 
                     look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
        
        attn2, attn_weights_block2 = self.mha2(
                enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)
        
        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)
        
        return out3, attn_weights_block1, attn_weights_block2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, encoderConfig, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1, BERT_weights_dir=None, trainable=True, vocab_inds=None):
        super(Encoder, self).__init__(trainable=trainable)

        self.BERT_weights_dir = BERT_weights_dir

        self.d_model = d_model
        self.num_layers = num_layers     

        if self.BERT_weights_dir is not None:
            if vocab_inds is None:
                we_tokens = tf.initializers.Constant(
                    np.load("{0}/module_bert_embeddings_word_embeddings:0.npy".format(self.BERT_weights_dir)))
            else:
                we_tokens = tf.initializers.Constant(
                    np.load("{0}/module_bert_embeddings_word_embeddings:0.npy".format(self.BERT_weights_dir))[vocab_inds, :])
            we_tokens_type = tf.initializers.Constant(
                np.load("{0}/module_bert_embeddings_token_type_embeddings:0.npy".format(self.BERT_weights_dir)))
            we_pos = tf.initializers.Constant(
                np.load("{0}/module_bert_embeddings_position_embeddings:0.npy".format(self.BERT_weights_dir)))
            we_norm = [tf.initializers.Constant(
                np.load("{0}/module_bert_embeddings_LayerNorm_beta:0.npy".format(self.BERT_weights_dir))),
                       tf.initializers.Constant(
                           np.load("{0}/module_bert_embeddings_LayerNorm_gamma:0.npy".format(self.BERT_weights_dir)))]

            self.tokens_embedding = tf.keras.layers.Embedding(input_vocab_size, d_model,
                                                              embeddings_initializer=we_tokens, trainable=trainable)
            self.tokens_type_embedding = tf.keras.layers.Embedding(2, d_model, embeddings_initializer=we_tokens_type,
                                                                   trainable=trainable)
            self.pos_embedding = tf.keras.layers.Embedding(512, d_model, embeddings_initializer=we_pos,
                                                           trainable=trainable)
            self.tokens_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, beta_initializer=we_norm[0],
                                                                  gamma_initializer=we_norm[1], trainable=trainable)

            self.enc_layers = []
            for layer_ind in range(0, num_layers):
                enc_layer = EncoderLayer(d_model, num_heads, dff, rate,
                                                    BERT_layer_ind=layer_ind, BERT_weights_dir=self.BERT_weights_dir)
                enc_layer.trainable = trainable
                self.enc_layers.append(enc_layer)
        else:
            if encoderConfig.get("pretrained_emb_path", None) is not None:
                self.embedding = tf.keras.layers.Embedding(input_vocab_size, encoderConfig["pretrained_emb_size"], embeddings_initializer=tf.initializers.Constant(np.load(encoderConfig["pretrained_emb_path"])))
            else:
                self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)

            self.pos_encoding = Data.positional_encoding(maximum_position_encoding, self.d_model)

            self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                               for _ in range(num_layers)]

        if encoderConfig.get("dense_emb", False):
            self.dense_emb = tf.keras.layers.Dense(d_model)
        else:
            self.dense_emb = None

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask, tokens_type_inds=None):

        seq_len = tf.shape(x)[1]
        batch_size = tf.shape(x)[0]

        if self.BERT_weights_dir is not None:
            x = self.tokens_embedding(x)

            if tokens_type_inds is None:
                tokens_type_inds = tf.zeros(shape=(batch_size, seq_len), dtype=tf.int32)
            tokens_type = self.tokens_type_embedding(tokens_type_inds)
            x += tokens_type

            seq_inds = tf.range(0, seq_len, dtype=tf.int32)
            seq_inds = tf.broadcast_to(seq_inds, (batch_size, seq_len))
            pos = self.pos_embedding(seq_inds)
            x += pos

            x = self.tokens_norm(x)

        else:

            # adding embedding and position encoding.
            x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
            if self.dense_emb is not None:
                x = self.dense_emb(x)
            x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
            x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)
        
        gpus = cycle(tf.config.experimental.list_physical_devices('GPU'))
        for i in range(self.num_layers):
            if DISTRIBUTED_LAYERS:
                gpuName = next(gpus).name.replace("/physical_device:", "")
                with tf.device(gpuName):
                    x = self.enc_layers[i](x, training, mask)
            else:
                x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)

    def save_weight_to_dir(self, dir):
        tokens_embedding_path = "{0}/module_bert_embeddings_word_embeddings:0.npy".format(dir)
        np.save(tokens_embedding_path, self.tokens_embedding.get_weights())
        tokens_type_embedding_path = "{0}/module_bert_embeddings_token_type_embeddings:0.npy".format(dir)
        np.save(tokens_type_embedding_path, self.tokens_type_embedding.get_weights())
        pos_embedding_name = "{0}/module_bert_embeddings_position_embeddings:0.npy".format(dir)
        np.save(pos_embedding_name, self.pos_embedding.get_weights())
        tokens_norm_beta_path = "{0}/module_bert_embeddings_LayerNorm_beta:0.npy".format(dir)
        tokens_norm_gamma_path = "{0}/module_bert_embeddings_LayerNorm_gamma:0.npy".format(dir)
        np.save(tokens_norm_beta_path, self.tokens_norm.beta.numpy())
        np.save(tokens_norm_gamma_path, self.tokens_norm.gamma.numpy())

        for layer_ind, layer in enumerate(self.enc_layers):
            np.save("{0}/module_bert_encoder_layer_{1}_attention_self_query_kernel:0.npy".format(dir, layer_ind),
                    layer.mha.wq.kernel.numpy())
            np.save("{0}/module_bert_encoder_layer_{1}_attention_self_query_bias:0.npy".format(dir, layer_ind),
                    layer.mha.wq.bias.numpy())
            np.save("{0}/module_bert_encoder_layer_{1}_attention_self_key_kernel:0.npy".format(dir, layer_ind),
                    layer.mha.wk.kernel.numpy())
            np.save("{0}/module_bert_encoder_layer_{1}_attention_self_key_bias:0.npy".format(dir, layer_ind),
                    layer.mha.wk.bias.numpy())
            np.save("{0}/module_bert_encoder_layer_{1}_attention_self_value_kernel:0.npy".format(dir, layer_ind),
                    layer.mha.wv.kernel.numpy())
            np.save("{0}/module_bert_encoder_layer_{1}_attention_self_value_bias:0.npy".format(dir, layer_ind),
                    layer.mha.wv.bias.numpy())

            np.save("{0}/module_bert_encoder_layer_{1}_attention_output_dense_kernel:0.npy".format(dir, layer_ind),
                    layer.mha.dense.kernel.numpy())
            np.save("{0}/module_bert_encoder_layer_{1}_attention_output_dense_bias:0.npy".format(dir, layer_ind),
                    layer.mha.dense.bias.numpy())

            np.save("{0}/module_bert_encoder_layer_{1}_intermediate_dense_kernel:0.npy".format(dir, layer_ind),
                    layer.ffn.layers[0].kernel.numpy())
            np.save("{0}/module_bert_encoder_layer_{1}_intermediate_dense_bias:0.npy".format(dir, layer_ind),
                    layer.ffn.layers[0].bias.numpy())
            np.save("{0}/module_bert_encoder_layer_{1}_output_dense_kernel:0.npy".format(dir, layer_ind),
                    layer.ffn.layers[1].kernel.numpy())
            np.save("{0}/module_bert_encoder_layer_{1}_output_dense_bias:0.npy".format(dir, layer_ind),
                    layer.ffn.layers[1].bias.numpy())

            np.save("{0}/module_bert_encoder_layer_{1}_attention_output_LayerNorm_beta:0.npy".format(dir, layer_ind),
                    layer.layernorm1.beta.numpy())
            np.save("{0}/module_bert_encoder_layer_{1}_attention_output_LayerNorm_gamma:0.npy".format(dir, layer_ind),
                    layer.layernorm1.gamma.numpy())
            np.save("{0}/module_bert_encoder_layer_{1}_output_LayerNorm_beta:0.npy".format(dir, layer_ind),
                    layer.layernorm2.beta.numpy())
            np.save("{0}/module_bert_encoder_layer_{1}_output_LayerNorm_gamma:0.npy".format(dir, layer_ind),
                    layer.layernorm2.gamma.numpy())


class EncoderLayerAW(EncoderLayer):
    def call(self, x, training, mask):
        attn_output, attn_weights = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2, attn_weights


class EncoderAW(Encoder):
    def __init__(self, encoderConfig, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1, BERT_weights_dir=None, trainable=True, vocab_inds=None):
        super(Encoder, self).__init__(trainable=trainable)

        self.BERT_weights_dir = BERT_weights_dir

        self.d_model = d_model
        self.num_layers = num_layers

        if self.BERT_weights_dir is not None:
            if vocab_inds is None:
                we_tokens = tf.initializers.Constant(
                    np.load("{0}/module_bert_embeddings_word_embeddings:0.npy".format(self.BERT_weights_dir)))
            else:
                we_tokens = tf.initializers.Constant(
                    np.load("{0}/module_bert_embeddings_word_embeddings:0.npy".format(self.BERT_weights_dir))[vocab_inds, :])
            we_tokens_type = tf.initializers.Constant(
                np.load("{0}/module_bert_embeddings_token_type_embeddings:0.npy".format(self.BERT_weights_dir)))
            we_pos = tf.initializers.Constant(
                np.load("{0}/module_bert_embeddings_position_embeddings:0.npy".format(self.BERT_weights_dir)))
            we_norm = [tf.initializers.Constant(
                np.load("{0}/module_bert_embeddings_LayerNorm_beta:0.npy".format(self.BERT_weights_dir))),
                       tf.initializers.Constant(
                           np.load("{0}/module_bert_embeddings_LayerNorm_gamma:0.npy".format(self.BERT_weights_dir)))]

            self.tokens_embedding = tf.keras.layers.Embedding(input_vocab_size, d_model,
                                                              embeddings_initializer=we_tokens, trainable=trainable)
            self.tokens_type_embedding = tf.keras.layers.Embedding(2, d_model, embeddings_initializer=we_tokens_type,
                                                                   trainable=trainable)
            self.pos_embedding = tf.keras.layers.Embedding(512, d_model, embeddings_initializer=we_pos,
                                                           trainable=trainable)
            self.tokens_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, beta_initializer=we_norm[0],
                                                                  gamma_initializer=we_norm[1], trainable=trainable)

            self.enc_layers = []
            for layer_ind in range(0, num_layers):
                enc_layer = EncoderLayerAW(d_model, num_heads, dff, rate,
                                                    BERT_layer_ind=layer_ind, BERT_weights_dir=self.BERT_weights_dir)
                enc_layer.trainable = trainable
                self.enc_layers.append(enc_layer)
        else:
            if encoderConfig.get("pretrained_emb_path", None) is not None:
                self.embedding = tf.keras.layers.Embedding(input_vocab_size, encoderConfig["pretrained_emb_size"], embeddings_initializer=tf.initializers.Constant(np.load(encoderConfig["pretrained_emb_path"])))
            else:
                self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)

            self.pos_encoding = Data.positional_encoding(maximum_position_encoding, self.d_model)

            self.enc_layers = [EncoderLayerAW(d_model, num_heads, dff, rate)
                               for _ in range(num_layers)]

        if encoderConfig.get("dense_emb", False):
            self.dense_emb = tf.keras.layers.Dense(d_model)
        else:
            self.dense_emb = None

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask, tokens_type_inds=None):

        seq_len = tf.shape(x)[1]
        batch_size = tf.shape(x)[0]

        if self.BERT_weights_dir is not None:
            x = self.tokens_embedding(x)

            if tokens_type_inds is None:
                tokens_type_inds = tf.zeros(shape=(batch_size, seq_len), dtype=tf.int32)
            tokens_type = self.tokens_type_embedding(tokens_type_inds)
            x += tokens_type

            seq_inds = tf.range(0, seq_len, dtype=tf.int32)
            seq_inds = tf.broadcast_to(seq_inds, (batch_size, seq_len))
            pos = self.pos_embedding(seq_inds)
            x += pos

            x = self.tokens_norm(x)

        else:

            # adding embedding and position encoding.
            x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
            if self.dense_emb is not None:
                x = self.dense_emb(x)
            x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
            x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        attention_weights = {}
        gpus = cycle(tf.config.experimental.list_physical_devices('GPU'))
        for i in range(self.num_layers):
            if DISTRIBUTED_LAYERS:
                gpuName = next(gpus).name.replace("/physical_device:", "")
                with tf.device(gpuName):
                    x, x_attn_weights = self.enc_layers[i](x, training, mask)
            else:
                x, x_attn_weights = self.enc_layers[i](x, training, mask)
            attention_weights["layer_{0}".format(i)] = x_attn_weights

        return x, attention_weights  # (batch_size, input_seq_len, d_model)


class EncoderBERTSumLayersOut(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1, BERT_weights_dir=None, trainable=False, vocab_inds=None):

        """
        Класс для кодировщика Encoder и Transformer, он же модель BERT.
        Пока может работать только с предобученными весами BERT.
        Этот класс почти аналогичен классу Encoder, есть два отличия:
        1. Выходом является взвешенное среднее активностей всех слоёв, коэфициенты для слоёв -- обучаемые переменные.
           Выходом сети будет sum(x_i*W_i)/sum(W_i) по всем i, i от 0 до num_layers + 2 (эмбединги слов и положений),
           x_i -- i-й слой сети, W_i -- коэфициент для i-того слоя,
           W_i от 0 до 1 W_i = sigmoid(W_linear_i), где W_linear_i -- обучаемые веса модели.
        2. Добавлен флаг trainable, если он True -- то идёт обучение весов BERT,
           если False -- то веса BERT заморожены и обучаеются только веса для коэфициентов взвешенного среднего.

        Parameters
        ----------
        num_layers: int
            Число слоёв, кроме embedding для слов и позиций.
            Можно указать меньше чем в модели BERT, тогда загрузится только n первых слоёв.

        d_model: int
            Размерность выходного слоя в блоках

        num_heads: int
            Число голов в MultiHeadAttention
        dff: int
            Размерность скрытого слоя в блоках

        input_vocab_size: int
            Размер словаря

        maximum_position_encoding: int
            *Максимальная длина последовательности, размерность слоя позиционного кодирования.
            *Сейчас не используется.

        rate: float
            Dropout rate от 0 до 1, 0 -- не применяется, 1 -- 100% нейронов отбрасывается.

        BERT_weights_dir: str
            Путь до весов BERT

        trainable: bool
            Флаг, являются ли веса BERT обучаемыми

        vocab_inds: str
            Путь до файла с соотвествием индексов новому маленькому словарю
        """

        super(EncoderBERTSumLayersOut, self).__init__()

        self.BERT_weights_dir = BERT_weights_dir

        self.d_model = d_model
        self.num_layers = num_layers

        if vocab_inds is None:
            we_tokens = tf.initializers.Constant(
                np.load("{0}/module_bert_embeddings_word_embeddings:0.npy".format(self.BERT_weights_dir)))
        else:
            we_tokens = tf.initializers.Constant(
                np.load("{0}/module_bert_embeddings_word_embeddings:0.npy".format(self.BERT_weights_dir))[vocab_inds,
                :])
        we_tokens_type = tf.initializers.Constant(
            np.load("{0}/module_bert_embeddings_token_type_embeddings:0.npy".format(self.BERT_weights_dir)))
        we_pos = tf.initializers.Constant(
            np.load("{0}/module_bert_embeddings_position_embeddings:0.npy".format(self.BERT_weights_dir)))
        we_norm = [tf.initializers.Constant(
            np.load("{0}/module_bert_embeddings_LayerNorm_beta:0.npy".format(self.BERT_weights_dir))),
            tf.initializers.Constant(
                np.load("{0}/module_bert_embeddings_LayerNorm_gamma:0.npy".format(self.BERT_weights_dir)))]
        self.tokens_embedding = tf.keras.layers.Embedding(input_vocab_size, d_model,
                                                          embeddings_initializer=we_tokens, trainable=trainable)
        self.tokens_type_embedding = tf.keras.layers.Embedding(2, d_model, embeddings_initializer=we_tokens_type,
                                                               trainable=trainable)
        self.pos_embedding = tf.keras.layers.Embedding(512, d_model, embeddings_initializer=we_pos,
                                                       trainable=trainable)
        self.tokens_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6, beta_initializer=we_norm[0],
                                                              gamma_initializer=we_norm[1], trainable=trainable)

        self.enc_layers = []
        for layer_ind in range(0, num_layers):
            enc_layer = EncoderLayer(d_model, num_heads, dff, rate,
                                    BERT_layer_ind=layer_ind, BERT_weights_dir=self.BERT_weights_dir)
            enc_layer.trainable = trainable
            self.enc_layers.append(enc_layer)

        self.dropout = tf.keras.layers.Dropout(rate)

    def build(self, input_shape):
        self.W_layers_sum_coef = self.add_weight("W_layers_sum_coef", shape=[self.num_layers + 2, ])

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        batch_size = tf.shape(x)[0]

        l_x = []  # Список выходов всех слоёв
        x = self.tokens_embedding(x)
        l_x.append(x)

        tokens_type_inds = tf.zeros(shape=(batch_size, seq_len), dtype=tf.int32)
        tokens_type = self.tokens_type_embedding(tokens_type_inds)
        x += tokens_type

        seq_inds = tf.range(0, seq_len, dtype=tf.int32)
        seq_inds = tf.broadcast_to(seq_inds, (batch_size, seq_len))
        pos = self.pos_embedding(seq_inds)
        x += pos

        x = self.tokens_norm(x)
        l_x.append(x)

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
            l_x.append(x)

        l_out = []
        for w_ind, layer_out in enumerate(l_x):
            l_out.append(layer_out * tf.sigmoid(self.W_layers_sum_coef[w_ind]))
        x = tf.reduce_sum(l_out, axis=0)
        x = x / tf.reduce_sum(tf.sigmoid(self.W_layers_sum_coef))

        return x  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                             maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = Data.positional_encoding(maximum_position_encoding, d_model)
        
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) 
                                             for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
        
    def call(self, x, enc_output, training, 
                     look_ahead_mask, padding_mask):

        seq_len = tf.shape(x)[1]
        attention_weights = {}
        
        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        
        x = self.dropout(x, training=training)
        
        gpus = cycle(tf.config.experimental.list_physical_devices('GPU'))
        for i in range(self.num_layers):
            if DISTRIBUTED_LAYERS:
                gpuName = next(gpus).name.replace("/physical_device:", "")
                with tf.device(gpuName):
                    x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                                                                 look_ahead_mask, padding_mask)

                    attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
                    attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
            else:
                x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                                             look_ahead_mask, padding_mask)

                attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
                attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
        
        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


class Transformer(tf.keras.Model):
    def __init__(self, modelConfig, input_vocab_size,
                             target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()

        if modelConfig["Encoder"].get("SumEmbedding", False):
            self.encoder = EncoderBERTSumLayersOut(modelConfig["Encoder"]["num_layers"], modelConfig["Encoder"]["d_model"], modelConfig["Encoder"]["num_heads"], modelConfig["Encoder"]["dff"], input_vocab_size, pe_input, rate=modelConfig["Encoder"]["dropout_rate"], BERT_weights_dir=modelConfig["Encoder"].get("BERT_weights", None), trainable=not modelConfig["Encoder"].get("freeze_weights", False))
        else:
            self.encoder = Encoder(modelConfig["Encoder"], modelConfig["Encoder"]["num_layers"], modelConfig["Encoder"]["d_model"], modelConfig["Encoder"]["num_heads"], modelConfig["Encoder"]["dff"], 
                                                     input_vocab_size, pe_input, modelConfig["Encoder"]["dropout_rate"], BERT_weights_dir=modelConfig["Encoder"].get("BERT_weights", None))

        self.decoder = Decoder(modelConfig["Decoder"]["num_layers"], modelConfig["Decoder"]["d_model"], modelConfig["Decoder"]["num_heads"], modelConfig["Decoder"]["dff"], 
                                                     target_vocab_size, pe_target, modelConfig["Decoder"]["dropout_rate"])

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
        
    def call(self, inp, tar, training, enc_padding_mask, 
                     look_ahead_mask, dec_padding_mask):

        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
        
        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
                tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        
        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        
        return final_output, attention_weights


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
    
        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
    
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def evaluate(inp_sentence, tokenizer, transformer, max_length, usingKeras=False):
    if "BERT_tokenizer" in tokenizer.__dict__:
        start_token = [tokenizer.BERT_tokenizer.vocab["[CLS]"]]
        end_token = [tokenizer.BERT_tokenizer.vocab["[SEP]"]]
    else:
        if type(tokenizer).__name__=="SentencePieceProcessor":
            start_token = [tokenizer.bos_id()]
            end_token = [tokenizer.eos_id()]
        else:
            start_token = [tokenizer.vocab_size]
            end_token = [tokenizer.vocab_size + 1]
        
    if type(inp_sentence)==bytes:        
        if type(tokenizer).__name__=="SentencePieceProcessor":
            inp_sentence = start_token + tokenizer.EncodeAsIds(inp_sentence) + end_token
        else:
            inp_sentence = start_token + tokenizer.encode(inp_sentence) + end_token


    encoder_input = tf.expand_dims(inp_sentence, 0)
    encoder_input = encoder_input[:,:max_length]
    #encoder_input = tf.slice(encoder_input, [0, 0], [tf.shape(encoder_input)[0], max_length])
    #encoder_input = tf.slice(encoder_input, [0, 0], [encoder_input.shape[0], max_length])
    
    # as the target is english, the first word to the transformer should be the
    # english start token.
    decoder_input = start_token
    output = tf.expand_dims(decoder_input, 0)
      
    for i in range(max_length):
        enc_padding_mask, combined_mask, dec_padding_mask = Data.create_masks(
            encoder_input, output)
      
        # predictions.shape == (batch_size, seq_len, vocab_size)
        if usingKeras:
            predictions, attention_weights = transformer((encoder_input,
                                                     output,
                                                     enc_padding_mask,
                                                     combined_mask,
                                                     dec_padding_mask),
                                                     False)
        else:
            predictions, attention_weights = transformer(encoder_input,
                                                        output,
                                                        False,
                                                        enc_padding_mask,
                                                        combined_mask,
                                                        dec_padding_mask)
        
        # select the last word from the seq_len dimension
        predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)
      
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        
        # return the result if the predicted_id is equal to the end token
        if predicted_id == end_token[0]:
          return tf.squeeze(output, axis=0), attention_weights
        
        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)
   
    return tf.squeeze(output, axis=0), attention_weights


class Transformer_keras(tf.keras.Model):
    def __init__(self, modelConfig, input_vocab_size,
                             target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer_keras, self).__init__()

        if modelConfig["Encoder"].get("SumEmbedding", False):
            self.encoder = EncoderBERTSumLayersOut(modelConfig["Encoder"]["num_layers"], modelConfig["Encoder"]["d_model"], modelConfig["Encoder"]["num_heads"], modelConfig["Encoder"]["dff"],
                                                   input_vocab_size, pe_input, rate=modelConfig["Encoder"]["dropout_rate"], BERT_weights_dir=modelConfig["Encoder"].get("BERT_weights", None),
                                                   trainable=not modelConfig["Encoder"].get("freeze_weights", False))
        else:
            self.encoder = Encoder(modelConfig["Encoder"], modelConfig["Encoder"]["num_layers"], modelConfig["Encoder"]["d_model"], modelConfig["Encoder"]["num_heads"], modelConfig["Encoder"]["dff"],
                                                     input_vocab_size, pe_input, modelConfig["Encoder"]["dropout_rate"], BERT_weights_dir=modelConfig["Encoder"].get("BERT_weights", None),
                                                     trainable=not modelConfig["Encoder"].get("freeze_weights", False))

        self.decoder = Decoder(modelConfig["Decoder"]["num_layers"], modelConfig["Decoder"]["d_model"], modelConfig["Decoder"]["num_heads"], modelConfig["Decoder"]["dff"],
                                                     target_vocab_size, pe_target, modelConfig["Decoder"]["dropout_rate"])

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs, training):
        inp, tar, enc_padding_mask, look_ahead_mask, dec_padding_mask = inputs
        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
                tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        # 
        return final_output

class Transformer_keras_predict(Transformer_keras):

    def call(self, inputs, training):
        inp, tar, enc_padding_mask, look_ahead_mask, dec_padding_mask = inputs
        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
                tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        # 
        return final_output, attention_weights


def evaluate_batch(encoder_input, tokenizer, transformer, max_length, usingKeras=False):
    if "BERT_tokenizer" in tokenizer.__dict__:
        start_token = [tokenizer.BERT_tokenizer.vocab["[CLS]"]]
        end_token = [tokenizer.BERT_tokenizer.vocab["[SEP]"]]
    else:
        if type(tokenizer).__name__=="SentencePieceProcessor":
            start_token = [tokenizer.bos_id()]
            end_token = [tokenizer.eos_id()]
        else:
            start_token = [tokenizer.vocab_size]
            end_token = [tokenizer.vocab_size + 1]

    decoder_input = np.tile(np.array(start_token, np.int32),  encoder_input.shape[0])
    output = tf.expand_dims(decoder_input, 1)
      
    for i in range(max_length):
        enc_padding_mask, combined_mask, dec_padding_mask = Data.create_masks(
            encoder_input, output)
      
        # predictions.shape == (batch_size, seq_len, vocab_size)
        if usingKeras:
            predictions, attention_weights = transformer((encoder_input,
                                                     output,
                                                     enc_padding_mask,
                                                     combined_mask,
                                                     dec_padding_mask),
                                                     False)
        else:
            predictions = transformer(encoder_input,
                                                         output,
                                                         False,
                                                         enc_padding_mask,
                                                         combined_mask,
                                                         dec_padding_mask)
        
        # select the last word from the seq_len dimension
        predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)
    
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        
        # return the result if the predicted_id is equal to the end token
        if (predicted_id == end_token[0]).numpy().all():
            return output, attention_weights
        
        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)
    return output, attention_weights
