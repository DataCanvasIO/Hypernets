# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTMCell, RNN, Dense, Embedding
from tensorflow.python.ops import clip_ops
import tensorflow as tf
from hypernets.core.search_space import MultipleChoice, Choice
from hypernets.core.searcher import Searcher, OptimizeDirection


class RnnController(Model):
    def __init__(self, search_space_fn, lstm_size=100, num_ops=5, temperature=None, tanh_constant=1.5, optimizer=None,
                 baseline=0., baseline_decay=0.999, entropy_weight=None, **kwargs):
        super(RnnController, self).__init__(**kwargs)

        self.search_space_fn = search_space_fn
        self.num_ops = num_ops
        self.lstm_size = lstm_size
        self.tanh_constant = tanh_constant
        self.temperature = temperature
        self.lstm_rnn = RNN(
            cell=LSTMCell(units=lstm_size, use_bias=False, name='lstm_cell'),
            name='lstm_rnn',
            stateful=True)
        self.g_emb = tf.random.uniform(shape=(1, 1, lstm_size))
        self.anchor = Dense(lstm_size, use_bias=False)
        self.q = Dense(lstm_size, use_bias=False)
        self.v = Dense(1, use_bias=False)

        self.log_prob = 0
        self.entropy = 0
        self.embedding = Embedding(num_ops + 1, lstm_size)
        self.dense_ops = Dense(num_ops, use_bias=False)

        self.baseline = baseline
        self.baseline_decay = baseline_decay
        self.entropy_weight = entropy_weight
        self.optimizer = optimizer if optimizer is not None else Adam(learning_rate=1e-3)
        self.clipnorm = 5.0

    def reset(self):
        self._inputs = self.g_emb
        self.attn_2 = {}
        self.log_prob = 0
        self.entropy = 0

    def _op_choice(self, op_id_true=None):
        logits = self.dense_ops(self.lstm_rnn(self._inputs))
        if self.temperature is not None:
            logits /= self.temperature
        if self.tanh_constant is not None:
            logits = self.tanh_constant * tf.tanh(logits)

        op_id = tf.reshape(tf.random.categorical(logits, num_samples=1), [1])
        if op_id_true:
            op_id_true = tf.reshape(op_id_true, [1])
            cur_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=op_id_true)
        else:
            if op_id[0] == 5:
                print(op_id)
            cur_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=op_id)

        self.log_prob += cur_log_prob
        cur_entropy = tf.stop_gradient(cur_log_prob * tf.exp(-cur_log_prob))
        self.entropy += cur_entropy
        self._inputs = tf.reshape(self.embedding(op_id), [1, 1, -1])
        op_id = op_id.numpy()[0]
        return op_id

    def _input_choice(self, n_inputs, choice_id='ID_default', index_true=None):
        q, anchors = [], []
        cell_type = '_'.join(choice_id.split('_')[:2])
        for id in range(n_inputs):
            key = f'{cell_type}_{id}'
            if key not in self.attn_2:
                self.attn_2[key] = self.lstm_rnn(self._inputs)
            q.append(self.anchor(self.attn_2[key]))
            anchors.append(self.attn_2[key])
        q = tf.concat(q, axis=0)
        q = tf.tanh(q + self.q(anchors[-1]))
        q = self.v(q)
        if self.temperature is not None:
            q /= self.temperature
        if self.tanh_constant is not None:
            q = self.tanh_constant * tf.tanh(q)
        logits = tf.reshape(q, [1, -1])
        softmax_logits = tf.math.log(tf.nn.softmax(logits, axis=-1))
        index = tf.reshape(tf.random.categorical(softmax_logits, 1), [1])

        if index_true:
            index_true = tf.reshape(index_true, [1])
            cur_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=index_true)
        else:
            cur_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=index)
        self.log_prob += cur_log_prob
        cur_entropy = tf.stop_gradient(cur_log_prob * tf.exp(-cur_log_prob))
        self.entropy += cur_entropy
        self._inputs = tf.reshape(anchors[tf.reduce_sum(index)], [1, 1, -1])

        return list(index.numpy())

    def sample(self):
        self.reset()
        space_sample = self.search_space_fn()
        for hp in space_sample.params_iterator:
            if not isinstance(hp, (MultipleChoice, Choice)):
                raise ValueError('Does not support ParameterSpace other than `MultipleChoice` and `Choice` in ENAS.')
            if not hp.is_mutable:
                hp.random_sample()
                continue
            if isinstance(hp, MultipleChoice):  # Param of InputChoice
                n_inputs = len(hp.options)
                input_id = self._input_choice(n_inputs, hp.id)
                hp.assign(input_id)
            else:  # Param of ModuleChoice
                n_ops = len(hp.options)
                if n_ops != self.num_ops:
                    raise ValueError('The number of modules in ModuleChoice is not equal to `num_ops`.')
                op_id = self._op_choice()
                hp.assign(op_id)
        return space_sample

    def calc_log_prob(self, space_sample):
        self.reset()
        for hp in space_sample.assigned_params_stack:
            if not isinstance(hp, (MultipleChoice, Choice)):
                raise ValueError('Does not support ParameterSpace other than `MultipleChoice` and `Choice` in ENAS.')
            if not hp.is_mutable:
                hp.random_sample()
                continue
            if isinstance(hp, MultipleChoice):  # Param of InputChoice
                n_inputs = len(hp.options)
                self._input_choice(n_inputs, hp.id, hp.value[0])
            else:  # Param of ModuleChoice
                n_ops = len(hp.options)
                if n_ops != self.num_ops:
                    raise ValueError('The number of modules in ModuleChoice is not equal to `num_ops`.')
                self._op_choice(hp.value)

    def train_one_sample(self, space_sample, reward):
        self.reset()
        with tf.GradientTape() as tape:
            self.reset()
            self.calc_log_prob(space_sample)
            if self.entropy_weight is not None:
                self.reward += self.entropy_weight * self.entropy
            self.baseline = self.baseline * self.baseline_decay + reward * (1 - self.baseline_decay)
            loss = self.log_prob * (reward - self.baseline)
            print(f'Reward: {reward}, Loss: {loss}')
            # loss += skip_weight * self.sample_skip_penalty
        grads = tape.gradient(loss, self.trainable_variables)

        if hasattr(self, "clipnorm"):
            grads = [clip_ops.clip_by_norm(g, self.clipnorm) for g in grads]
        if hasattr(self, "clipvalue"):
            grads = [
                clip_ops.clip_by_value(g, -self.clipvalue, self.clipvalue)
                for g in grads
            ]
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss


class EnasSearcher(Searcher):
    def __init__(self, space_fn, lstm_size=100, temperature=None, tanh_constant=1.5, optimizer=None,
                 baseline=0., baseline_decay=0.999, entropy_weight=None, optimize_direction=OptimizeDirection.Minimize,
                 use_meta_learner=True):
        Searcher.__init__(self, space_fn=space_fn, optimize_direction=optimize_direction,
                          use_meta_learner=use_meta_learner)
        num_ops = self._get_num_ops(space_fn)
        self.controller = RnnController(space_fn, lstm_size, num_ops, temperature, tanh_constant, optimizer, baseline,
                                        baseline_decay, entropy_weight)

    @property
    def parallelizable(self):
        return False

    def sample(self):
        return self.controller.sample()

    def update_result(self, space_sample, result):
        return self.controller.train_one_sample(space_sample, result)

    def _get_num_ops(self, space_fn):
        space_sample = space_fn()
        num_ops = None
        for hp in space_sample.params_iterator:
            if not isinstance(hp, (MultipleChoice, Choice)):
                raise ValueError('Does not support ParameterSpace other than `MultipleChoice` and `Choice` in ENAS.')
            if not hp.is_mutable:
                hp.random_sample()
                continue
            if isinstance(hp, MultipleChoice):  # Param of InputChoice
                hp.random_sample()
            else:  # Param of ModuleChoice
                n_ops = len(hp.options)
                if num_ops is None:
                    num_ops = n_ops
                else:
                    if num_ops != n_ops:
                        raise ValueError('The number of modules in all ModuleChoice must be the same.')
        if num_ops is None:
            num_ops = 5
        return num_ops
