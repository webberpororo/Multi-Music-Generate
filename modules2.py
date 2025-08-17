# import os
# os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
import tensorflow as tf
import numpy as np

from tensorflow.keras.mixed_precision import set_global_policy

# set_global_policy('mixed_float16')
np.set_printoptions(threshold=np.inf)
tf.config.optimizer.set_jit(True)

tf.random.set_seed(42)
# tf.keras.backend.clear_session()

class TransformerModel(tf.keras.Model):
    def __init__(self, n_token, n_layer, d_model, d_embed, n_head, d_head, d_inner,
                 dropout, dropatt, initializer, proj_initializer=None, mem_len=None,
                 same_length=False, clamp_len=-1, untie_r=False, proj_same_dim=True):
        super(TransformerModel, self).__init__()
        self.n_token = n_token
        self.n_layer = n_layer
        self.d_model = d_model
        self.d_embed = d_embed
        self.n_head = n_head
        self.d_head = d_head
        self.d_inner = d_inner
        self.dropout = dropout
        self.dropatt = dropatt
        self.initializer = initializer
        self.proj_initializer = proj_initializer
        self.mem_len = mem_len
        self.same_length = same_length
        self.clamp_len = clamp_len
        self.untie_r = untie_r
        self.proj_same_dim = proj_same_dim

        # Embedding and projection
        self.embedding = tf.keras.layers.Embedding(n_token, d_embed)

        # with tf.device('/GPU:0'):
        #     embedding_weight = torch.randn(n_token, d_embed) * 0.05
        #     embedding_weight_np = embedding_weight.detach().cpu().numpy().astype('float32')
        #     self.embedding = tf.keras.layers.Embedding(
        #         n_token, d_embed, embeddings_initializer=tf.keras.initializers.Constant(embedding_weight_np)
        #     )

        if d_embed != d_model:
            self.proj = tf.keras.layers.Dense(d_model, kernel_initializer=proj_initializer)
        else:
            self.proj = None

        # Relative biases
        if untie_r:
            self.r_w_bias = [tf.Variable(initializer([n_head, d_head])) for _ in range(n_layer)]
            self.r_r_bias = [tf.Variable(initializer([n_head, d_head])) for _ in range(n_layer)]
        else:
            self.r_w_bias = tf.Variable(initializer([n_head, d_head]))
            self.r_r_bias = tf.Variable(initializer([n_head, d_head]))

        # Transformer layers
        self.layer = [TransformerLayer(d_model, d_inner, n_head, d_head, dropout, dropatt, initializer)
                       for _ in range(n_layer)]

        # Output layer
        self.before_batch1 = tf.keras.layers.BatchNormalization()
        self.before_out1 = tf.keras.layers.Dense(1024, kernel_initializer=initializer)
        self.before_batch2 = tf.keras.layers.BatchNormalization()
        self.before_out2 = tf.keras.layers.Dense(2048, kernel_initializer=initializer)
        self.before_batch3 = tf.keras.layers.BatchNormalization()
        self.before_out3 = tf.keras.layers.Dense(4096, kernel_initializer=initializer)
        self.before_batch4 = tf.keras.layers.BatchNormalization()
        self.before_out4 = tf.keras.layers.Dense(8192, kernel_initializer=initializer)
        self.before_batch5 = tf.keras.layers.BatchNormalization()
        self.output_layer = tf.keras.layers.Dense(n_token, kernel_initializer=initializer)
        self.before_batch6 = tf.keras.layers.BatchNormalization()

    def call(self, dec_inp, target, mems=None, training=False):
        # print(dec_inp.shape)
        # print(target.shape)
        dec_inp = tf.transpose(dec_inp, perm=[2, 0, 1])
        target = tf.transpose(target, perm=[2, 0, 1])
        # print(dec_inp.shape)
        # print(target.shape)
        qlen = tf.shape(dec_inp)[0]
        mlen = tf.shape(mems[0])[0] if mems is not None else 0
        klen = qlen + mlen
        # print(qlen)
        # print(mlen)
        # print(klen)

        # Embedding lookup

        # embeddings, shared_params = self.normal_embedding_lookup(
        #     x=dec_inp,
        #     n_token=self.n_token,
        #     d_embed=self.d_embed,
        #     d_proj=self.d_model,
        #     initializer=self.initializer,
        #     proj_initializer=self.proj_initializer)

        embeddings = self.embedding(dec_inp)
        if self.proj is not None:
            embeddings = self.proj(embeddings)
        emb_scale = self.d_model ** 0.5
        embeddings *= emb_scale
        # noise_scale = tf.Variable(initial_value=tf.ones([1]) * 0.1, trainable=training)
        # embeddings = embeddings + tf.random.normal(tf.shape(embeddings)) * noise_scale
        # embeddings = embeddings + tf.random.normal(tf.shape(embeddings))
        # print('word_emb', embeddings[511])

        # Positional embedding
        pos_seq = tf.range(klen - 1, -1, -1.0)
        pos_seq = tf.reshape(pos_seq, [klen, 1])
        pos_seq = tf.concat([pos_seq, pos_seq], -1)
        # print(pos_seq.shape)
        if self.clamp_len > 0:
            pos_seq = tf.minimum(pos_seq, self.clamp_len)
        inv_freq = 1 / (10000 ** (tf.range(0, self.d_model // 2 * 2, 2.0) / self.d_model)) # 頻率分布
        pos_emb = self.positional_embedding(pos_seq, inv_freq)

        # Dropout
        output = tf.keras.layers.Dropout(self.dropout)(embeddings, training=training)
        # print('output', output[511])
        pos_emb = tf.keras.layers.Dropout(self.dropout)(pos_emb, training=training)
        # print(output.shape)
        # print('pos', pos_emb.shape)

        # Memory
        if mems is None:
            mems = [None] * self.n_layer

        new_mems = []
        for i in range(self.n_layer):
            # Cache new mems
            new_mems.append(self._cache_mem(output, mems[i], self.mem_len))

            # Multi-head attention
            output = self.layer[i](output, pos_emb,
                                   self.r_w_bias[i] if self.untie_r else self.r_w_bias,
                                   self.r_r_bias[i] if self.untie_r else self.r_r_bias, mems[i], training=training)

        # output = tf.transpose(output, perm=[0, 2, 1, 3])
        # print('output', output.shape)
        output = tf.keras.layers.Dropout(self.dropout)(output, training=training)
        # print('output', output.shape)

        # output = self.before_out1(output)
        # output = self.before_batch1(output)

        # output = self.before_out2(output)
        # output = self.before_batch2(output)

        # output = self.before_out3(output)
        # output = self.before_batch3(output)

        # output = self.before_out4(output)
        # output = self.before_batch4(output)

        logits = self.output_layer(output)
        logits = self.before_batch5(logits)
        if logits.shape[0] != target.shape[0]:
            # print(logits.shape)
            # print(target.shape)
            logits = tf.transpose(logits, perm = [3, 1, 2, 0])
        # print('logitss', logits.shape)

        # print(logits.shape)
        # print(target.shape)

        # print('output', logits[511])
        # print('target', target[511])

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=logits)
        # print('loss', loss[511])
        loss = tf.reduce_mean(loss)
        # print(logits.shape)

        return loss, logits, new_mems

    def positional_embedding(self, pos_seq, inv_freq, bsz=None):
        # print(pos_seq.shape)
        # print(inv_freq.shape)
        sinusoid_inp = tf.einsum('in,j->inj', pos_seq, inv_freq)
        pos_emb = tf.concat([tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)], -1)
        # print('position', pos_emb.shape)
        if bsz is not None:
            return tf.tile(pos_emb[:, :, None, :], [1, 1, bsz, 1]) # [seq_len, 1, d_model]->[seq_len, bsz, d_model]
        else:
            return pos_emb[:, :, None, :]

    def _cache_mem(self, curr_out, prev_mem, mem_len=None):
        if mem_len is None or prev_mem is None:
            return curr_out
        elif mem_len == 0:
            return prev_mem
        else:
            # Ensure curr_out has shape [seq_len, batch_size, d_model]
            # print(curr_out.shape)
            # if len(curr_out.shape) == 3 and curr_out.shape[0] != self.mem_len:
            #     curr_out = tf.transpose(curr_out, perm=[1, 0, 2])  # Transpose to [seq_len, batch_size, d_model]
            prev_mem = tf.cast(prev_mem, dtype=tf.float32)
            if prev_mem.shape[1] != curr_out.shape[1]:
                prev_mem = tf.transpose(prev_mem, perm=[0, 2, 1, 3])
            curr_out = tf.cast(curr_out, dtype=tf.float32)
            # print(prev_mem.shape)
            # print(curr_out.shape)
            new_mem = tf.concat([prev_mem, curr_out], 0)[-mem_len:]
        return tf.stop_gradient(new_mem)

    # def _compress_mem(self, curr_out, prev_mem, mem_len, compress_dim=512):
    #     if prev_mem is None:
    #         return curr_out
    #     compressor = tf.keras.layers.Dense(compress_dim)
    #     prev_mem_compressed = compressor(prev_mem)
    #     new_mem = tf.concat([prev_mem_compressed, curr_out], axis=0)[-mem_len:]
    #     return new_mem

class TransformerLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, d_inner, n_head, d_head, dropout, dropatt, initializer):
        super(TransformerLayer, self).__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.n_head = n_head
        self.d_head = d_head
        self.dropout = dropout
        self.dropatt = dropatt
        self.initializer = initializer

        # Multi-head attention
        self.attn = RelMultiHeadAttn(d_model, n_head, d_head, dropout, dropatt, initializer)
        # Position-wise feed-forward
        self.ff = PositionwiseFF(d_model, d_inner, dropout, initializer)

    def call(self, inp, pos_emb, r_w_bias, r_r_bias, mems=None, training=True):
        # Multi-head attention
        output = self.attn(inp, pos_emb, r_w_bias, r_r_bias, mems, training=training)
        # Position-wise feed-forward
        output = self.ff(output, training=training)
        return output


class RelMultiHeadAttn(tf.keras.layers.Layer):
    def __init__(self, d_model, n_head, d_head, dropout, dropatt, initializer):
        super(RelMultiHeadAttn, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.dropout = dropout
        self.dropatt = dropatt
        self.initializer = initializer
        self.mem_len = 512
        self.same_length = False

        # Linear layers
        self.qkv = tf.keras.layers.Dense(d_model, use_bias=False, kernel_initializer=initializer)
        # self.qkv_c0 = tf.keras.layers.Conv2D(d_model, (3, 3), activation='relu', padding='same',
        #                                     use_bias=False, kernel_initializer=initializer)
        self.qkv_c = tf.keras.layers.Conv2D(3 * n_head * d_head // 4, (2, 2), activation='relu', padding='same',
                                            use_bias=False, kernel_initializer=initializer) # 3 * n_head * d_head // 2 or d_model
        self.r = tf.keras.layers.Dense(d_model, use_bias=False, kernel_initializer=initializer)
        # self.r_c0 = tf.keras.layers.Conv2D(d_model, (3, 3), activation='relu', padding='same', use_bias=False,
        #                                   kernel_initializer=initializer)
        self.r_c = tf.keras.layers.Conv2D(n_head * d_head, (2, 2), activation='relu', padding='same', use_bias=False,
                                            kernel_initializer=initializer) # n_head * d_head or d_model
        self.o = tf.keras.layers.Dense(d_model, use_bias=False, kernel_initializer=initializer)
        # self.o_c0 = tf.keras.layers.Conv2D(d_model, (3, 3), activation='relu', padding='same',
        #                                   use_bias=False, kernel_initializer=initializer)
        self.o_c = tf.keras.layers.Conv2D(d_model, (3, 3), activation='relu', padding='same',
                                            use_bias=False, kernel_initializer=initializer)

    def call(self, w, r, r_w_bias, r_r_bias, mems=None, training=True):
        # Ensure w has shape [seq_len, batch_size, d_model]
        # if len(w.shape) == 3 and w.shape[0] != self.mem_len:
        #     w = tf.transpose(w, perm=[1, 0, 2])  # Transpose to [seq_len, batch_size, d_model]
        # print(w.shape)
        # print(r.shape)
        # print(mems.shape)
        qlen = tf.shape(w)[0]
        rlen = tf.shape(r)[0]
        bsz = tf.shape(w)[1]

        # Concatenate memory if exists
        if mems.shape[1] != w.shape[1]:
            mems = tf.transpose(mems, perm=[0, 2, 1, 3])
        if mems is not None and mems.shape.ndims > 1:
            mems = tf.cast(mems, dtype=tf.float32)
            w = tf.cast(w, dtype=tf.float32)
            cat = tf.concat([mems, w], 0)
        else:
            cat = w

        # Linear projections
        # print(cat.shape)

        cat = self.qkv(cat)

        # print(cat[:5])
        w_heads = self.qkv_c(cat)
        # print('r', r.shape)

        # print(r.shape)

        r = self.r(r)

        # print(r[:5])
        r_head_k = self.r_c(r)
        # print(w_heads.shape)
        # print(r_head_k.shape)

        # Split into query, key, value
        w_head_q, w_head_k, w_head_v = tf.split(w_heads, 3, -1)
        w_head_q = w_head_q[-qlen:]
        # print(w_head_q.shape)
        # print(qlen)
        # print(bsz)
        # print(self.n_head)
        # print(self.d_head)
        # print(r_head_k.shape)
        # print(rlen)

        # Reshape
        w_head_q = tf.reshape(w_head_q, [qlen, bsz, self.n_head, self.d_head])
        w_head_k = tf.reshape(w_head_k, [tf.shape(w_head_k)[0], bsz, self.n_head, self.d_head])
        w_head_v = tf.reshape(w_head_v, [tf.shape(w_head_v)[0], bsz, self.n_head, self.d_head])
        # 相對位置編碼的 Key 矩陣
        r_head_k = tf.reshape(r_head_k, [rlen, self.n_head, 2, self.d_head])
        # print(w_head_q.shape)
        # print(r_w_bias.shape)
        # print(w_head_k.shape)
        # print(r_r_bias.shape)
        # print(r_head_k.shape)

        # Compute attention scores
        # AC 注意力分數的第一部分，由 Query 和 Key 的點積計算，r_w_bias 在計算注意力分數時，為 Query 添加額外的位置信息(positional bias)
        AC = tf.einsum('ibnd,jbnd->ijbn', tf.cast(w_head_q, dtype=tf.float32) + tf.cast(r_w_bias, dtype=tf.float32), tf.cast(w_head_k, dtype=tf.float32))
        # BD 注意力分數的第二部分，由 Query 和相對位置編碼的 Key 計算，r_r_bias 相對位置編碼的偏差
        BD = tf.einsum('ibnd,jned->ijbn', tf.cast(w_head_q, dtype=tf.float32) + tf.cast(r_r_bias, dtype=tf.float32), tf.cast(r_head_k, dtype=tf.float32))
        BD = self.rel_shift(BD)
        # print(AC.shape)
        # print(BD.shape)

        if AC.shape != BD.shape:
            raise ValueError(f"Shapes of AC and BD must match. AC: {AC.shape}, BD: {BD.shape}")

        # Scale and mask
        attn_score = (AC + BD) * (1 / (self.d_head ** 0.5))
        attn_mask = self._create_mask(qlen, mems[0].shape[2], self.same_length)
        # print(attn_mask.shape)
        # print(attn_score.shape)
        attn_score = attn_score * (1 - attn_mask) - 1e30 * attn_mask

        # Attention probabilities
        attn_prob = tf.nn.softmax(attn_score, 1)
        attn_prob = tf.keras.layers.Dropout(self.dropatt)(attn_prob, training=training)

        # Attention output
        # print(attn_prob.shape)
        # print(w_head_v.shape)
        attn_vec = tf.einsum('ijbn,jbnd->ibnd', attn_prob, w_head_v)
        # print(attn_vec.shape)
        attn_vec = tf.reshape(attn_vec, [qlen, 1, bsz, self.n_head * self.d_head])

        # Output projection
        # print(attn_vec.shape)
        attn_out = self.o(attn_vec)
        # attn_out = self.o_c(attn_vec)
        attn_out = tf.keras.layers.Dropout(self.dropout)(attn_out, training=training)
        # res_scale = tf.Variable(initial_value=tf.ones([1]), trainable=training)
        attn_out = tf.transpose(attn_out, perm=[0, 2, 1, 3])
        # print(attn_out.shape)
        # print(w.shape)
        output = tf.keras.layers.LayerNormalization(axis=-1)(attn_out + w)
        # output = tf.transpose(output, perm=[0, 2, 1, 3])
        # print(output.shape)

        return output

    def rel_shift(self, x):
        x_size = tf.shape(x)
        x = tf.pad(x, [[0, 0], [1, 0], [0, 0], [0, 0]])  # Pad on the second dimension
        x = tf.reshape(x, [x_size[1] + 1, x_size[0], x_size[2], x_size[3]])  # Reshape
        x = tf.slice(x, [1, 0, 0, 0], [-1, -1, -1, -1])  # Slice
        x = tf.reshape(x, x_size)  # Reshape back to original shape
        return x

    def _create_mask(self, qlen, mlen, same_length=False):
        # print(qlen)
        # print(mlen)
        attn_mask = tf.ones([qlen, qlen])
        mask_u = tf.linalg.band_part(attn_mask, 0, -1)  # Upper triangular part
        mask_dia = tf.linalg.band_part(attn_mask, 0, 0)  # Diagonal
        attn_mask_pad = tf.zeros([qlen, mlen])
        ret = tf.concat([attn_mask_pad, mask_u - mask_dia], 1)
        if same_length:
            mask_l = tf.linalg.band_part(attn_mask, -1, 0)  # Lower triangular part
            ret = tf.concat([ret[:, :qlen] + mask_l - mask_dia, ret[:, qlen:]], 1)
        # Expand dimensions to match attn_score shape [qlen, klen, batch_size, n_head]
        ret = ret[:, :, tf.newaxis, tf.newaxis]  # Shape: [qlen, klen, 1, 1]
        return ret


class PositionwiseFF(tf.keras.layers.Layer):
    def __init__(self, d_model, d_inner, dropout, initializer):
        super(PositionwiseFF, self).__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout
        self.initializer = initializer
        self.gate = self.ff1 = tf.keras.layers.Dense(d_inner, kernel_initializer=initializer)
        # self.gate_c = tf.keras.layers.Conv2D(d_inner, (3, 3), padding='same',
        #                                     use_bias=False, kernel_initializer=initializer)

        # Linear layers
        self.ff1 = tf.keras.layers.Dense(d_inner, kernel_initializer=initializer)
        # self.ff1_c = tf.keras.layers.Conv2D(d_inner, (3, 3), padding='same',
        #                                     use_bias=False, kernel_initializer=initializer)
        self.ff2 = tf.keras.layers.Dense(d_model, kernel_initializer=initializer)
        # self.ff2_c = tf.keras.layers.Conv2D(d_model, (3, 3), padding='same',
        #                                     use_bias=False, kernel_initializer=initializer)
        # self.ff2 = tf.keras.layers.Dense(d_inner, kernel_initializer=initializer)
        # self.ff3 = tf.keras.layers.Dense(d_model, kernel_initializer=initializer)

    def call(self, inp, training=True):
        output = self.ff1(inp)
        output = tf.nn.leaky_relu(output, alpha=0.1)# * tf.nn.sigmoid(self.gate(inp))
        output = tf.keras.layers.Dropout(self.dropout)(output, training=training)

        output = self.ff2(output)
        output = tf.nn.leaky_relu(output, alpha=0.1)
        output = tf.keras.layers.Dropout(self.dropout)(output, training=training)

        # output = self.ff3(output)
        # print(output.shape)
        # print(inp.shape)
        output = tf.keras.layers.LayerNormalization(axis=-1)(output + inp)
        return output