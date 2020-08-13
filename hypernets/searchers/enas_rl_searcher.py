# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTMCell, RNN, Dense, Embedding
from tensorflow.keras.losses import SparseCategoricalCrossentropy, Reduction
import tensorflow as tf
from hypernets.core.search_space import MultipleChoice, Choice


class RnnController(Model):
    def __init__(self, search_space_fn, lstm_size=100, num_ops=5, entropy_reduction='sum', **kwargs):
        super(RnnController, self).__init__(**kwargs)

        self.search_space_fn = search_space_fn
        self.lstm_rnn = RNN(
            cell=LSTMCell(units=lstm_size, use_bias=False, name='lstm_cell'),
            name='lstm_rnn',
            stateful=True)
        self.g_emb = tf.random.uniform(shape=(1, 1, lstm_size))
        self.anchor = Dense(lstm_size, use_bias=False)
        self.q = Dense(lstm_size, use_bias=False)
        self.v = Dense(1, use_bias=False)

        self.sample_log_prob = 0
        self.sample_entropy = 0
        self.num_ops = num_ops

        self.embedding = Embedding(self.num_ops + 1, lstm_size)
        self.soft = Dense(self.num_ops, activation='softmax', use_bias=False)
        self.entropy_reduction = tf.reduce_sum if entropy_reduction == 'sum' else tf.reduce_mean
        self.cross_entropy_loss = SparseCategoricalCrossentropy(from_logits=True, reduction=Reduction.NONE)

    def reset(self):
        self._inputs = self.g_emb
        self.attn_2 = {}
        self.sample_log_prob = 0
        self.sample_entropy = 0

    def _op_choice(self):
        logit = self.soft(self.lstm_rnn(self._inputs))
        op_id = tf.reshape(tf.random.categorical(tf.math.log(logit), num_samples=1), [1])
        log_prob = self.cross_entropy_loss(op_id, logit)
        self.sample_log_prob += self.entropy_reduction(log_prob)
        entropy = log_prob * tf.math.exp(-log_prob)
        self.sample_entropy += self.entropy_reduction(entropy)
        self._inputs = tf.reshape(self.embedding(op_id), [1, 1, -1])
        op_id = op_id.numpy()[0]
        return op_id

    def _input_choice(self, n_inputs):
        q, anchors = [], []
        for label in range(n_inputs):
            if label not in self.attn_2:
                self.attn_2[label] = self.lstm_rnn(self._inputs)
            q.append(self.anchor(self.attn_2[label]))
            anchors.append(self.attn_2[label])
        q = tf.concat(q, axis=0)
        q = tf.tanh(q + self.q(anchors[-1]))
        q = self.v(q)
        logit = tf.reshape(q, [1, -1])
        softmax_logit = tf.math.log(tf.nn.softmax(logit, axis=-1))
        input_id = tf.random.categorical(softmax_logit, 1).numpy()[0][0]

        log_prob = self.cross_entropy_loss(logit, q.numpy())
        self._inputs = tf.reshape(anchors[input_id], [1, 1, -1])

        self.sample_log_prob += self.entropy_reduction(log_prob)
        entropy = log_prob * tf.exp(-log_prob)
        self.sample_entropy += self.entropy_reduction(entropy)
        return [input_id]

    def sample(self, ):
        space_sample = self.search_space_fn()
        for hp in space_sample.params_iterator:
            if not isinstance(hp, (MultipleChoice, Choice)):
                raise ValueError('Does not support parameter spaces other than `MultipleChoice` and `Choice`')
            if not hp.is_mutable:
                hp.random_sample()
                continue
            if isinstance(hp, MultipleChoice):  # Param of InputChoice
                n_inputs = len(hp.options)
                input_id = self._input_choice(n_inputs)
                hp.assign(input_id)
            else:  # Param of ModuleChoice
                n_ops = len(hp.options)
                if n_ops != self.num_ops:
                    raise ValueError('')
                op_id = self._op_choice()
                hp.assign(op_id)
        return space_sample
