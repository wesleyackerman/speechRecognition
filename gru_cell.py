import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import RNNCell


class GruCell(RNNCell):
    def __init__(self, s_dim, v_size, reuse=None, name=None):
        super(GruCell, self).__init__()
        self.state_dim = s_dim
        self.vocab_size = v_size
        self._name = name

    @property
    def state_size(self):
        return self.state_dim

    @property
    def output_size(self):
        return self.state_dim

    def __call__(self, inputs, state, scope=None):
        input_dim = inputs.shape[-1]
        w_xr = tf.get_variable(self._name + 'w_xr', [input_dim, self.state_dim], dtype=tf.float32,
                               initializer=tf.random_normal_initializer)
        w_hr = tf.get_variable(self._name + 'w_hr', [self.state_dim, self.state_dim], dtype=tf.float32,
                               initializer=tf.random_normal_initializer)
        b_r = tf.get_variable(self._name + 'b_r', [self.state_dim], dtype=tf.float32,
                              initializer=tf.random_normal_initializer)

        w_xz = tf.get_variable(self._name + 'w_xz', [input_dim, self.state_dim], dtype=tf.float32,
                               initializer=tf.random_normal_initializer)
        w_hz = tf.get_variable(self._name + 'w_hz', [self.state_dim, self.state_dim], dtype=tf.float32,
                               initializer=tf.random_normal_initializer)
        b_z = tf.get_variable(self._name + 'b_z', [self.state_dim], dtype=tf.float32,
                              initializer=tf.random_normal_initializer)

        w_xh = tf.get_variable(self._name + 'w_xh', [input_dim, self.state_dim], dtype=tf.float32,
                               initializer=tf.random_normal_initializer)
        w_hh = tf.get_variable(self._name + 'w_hh', [self.state_dim, self.state_dim], dtype=tf.float32,
                               initializer=tf.random_normal_initializer)
        b_h = tf.get_variable(self._name + 'b_h', [self.state_dim], dtype=tf.float32,
                              initializer=tf.random_normal_initializer)

        r_t = tf.sigmoid(tf.matmul(inputs, w_xr) + tf.matmul(state, w_hr) + b_r, name=self._name + 'r_sigmoid')
        z_t = tf.sigmoid(tf.matmul(inputs, w_xz) + tf.matmul(state, w_hz) + b_z, name=self._name + 'z_sigmoid')
        h_candidate = tf.tanh(tf.matmul(inputs, w_xh) + tf.matmul(r_t * state, w_hh) + b_h, name=self._name + 'h_tanh')

        new_state = z_t * state + (1 - z_t) * h_candidate
        return new_state, new_state
