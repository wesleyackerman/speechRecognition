#! /usr/bin/python
import tensorflow as tf
import numpy as np
from textloader import TextLoader
from tensorflow.contrib.rnn import BasicLSTMCell, MultiRNNCell
from tensorflow.contrib.legacy_seq2seq import rnn_decoder, sequence_loss
from tensorflow.contrib.rnn import RNNCell


class GRU(RNNCell):
    def __init__(self, num_units):
        self._num_units = num_units

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        initializer = tf.contrib.layers.variance_scaling_initializer
        input_size = inputs.get_shape().as_list()[-1]

        Wz = tf.get_variable('Wz', [input_size, self._num_units], initializer=initializer(), reuse=True)
        Uz = tf.get_variable('Uz', [self._num_units, self._num_units], initializer=initializer(), reuse=True)
        bz = tf.get_variable('bz', [self._num_units], initializer=initializer(), reuse=True)

        Wr = tf.get_variable('Wr', [input_size, self._num_units], initializer=initializer(), reuse=True)
        Ur = tf.get_variable('Ur', [self._num_units, self._num_units], initializer=initializer(), reuse=True)
        br = tf.get_variable('br', [self._num_units], initializer=initializer(), reuse=True)

        Wh = tf.get_variable('Wh', [input_size, self._num_units], initializer=initializer(), reuse=True)
        Uh = tf.get_variable('Uh', [self._num_units, self._num_units], initializer=initializer(), reuse=True)
        bh = tf.get_variable('bh', [self._num_units], initializer=initializer(), reuse=True)

        zt = tf.sigmoid(tf.matmul(inputs, Wz) + tf.matmul(state, Uz) + bz)
        rt = tf.sigmoid(tf.matmul(inputs, Wr) + tf.matmul(state, Ur) + br)
        ht = zt * state + (1 - zt) * tf.tanh(tf.matmul(inputs, Wh) + tf.matmul((rt * state), Uh) + bh)
        return ht, ht

#
# -------------------------------------------
#
# Global variables

batch_size = 50
sequence_length = 50

data_loader = TextLoader(".", batch_size, sequence_length)

vocab_size = data_loader.vocab_size  # dimension of one-hot encodings
state_dim = 128

num_layers = 2

tf.reset_default_graph()

#
# ==================================================================
# ==================================================================
# ==================================================================
#

# define placeholders for our inputs.
# in_ph is assumed to be [batch_size,sequence_length]
# targ_ph is assumed to be [batch_size,sequence_length]

in_ph = tf.placeholder(tf.int32, [ batch_size, sequence_length ], name='inputs')
targ_ph = tf.placeholder(tf.int32, [ batch_size, sequence_length ], name='targets')
in_onehot = tf.one_hot(in_ph, vocab_size, name="input_onehot")

input = tf.split(in_onehot, sequence_length, axis=1)
input = [tf.squeeze(input_, [1]) for input_ in input]
targets = tf.split(targ_ph, sequence_length, axis=1)

# at this point, inputs is a list of length sequence_length
# each element of inputs is [batch_size,vocab_size]

# targets is a list of length sequence_length
# each element of targets is a 1D vector of length batch_size

# ------------------
# YOUR COMPUTATION GRAPH HERE
# create a BasicLSTMCell
#   use it to create a MultiRNNCell
#   use it to create an initial_state
#     note that initial_state will be a *list* of tensors!

# call seq2seq.rnn_decoder

# transform the list of state outputs to a list of logits.
# use a linear transformation.

# call seq2seq.sequence_loss

# create a training op using the Adam optimizer

# cells = [BasicLSTMCell(state_dim) for _ in range(num_layers)]
cells = [GRU(state_dim) for _ in range(num_layers)]
rnnCell = MultiRNNCell(cells)
initial_state = rnnCell.zero_state(batch_size, tf.float32)

outputs, final_state = rnn_decoder(input, initial_state, rnnCell)

initializer = tf.contrib.layers.variance_scaling_initializer
W = tf.get_variable('W', [state_dim, vocab_size], initializer=initializer())
b = tf.get_variable('b', [vocab_size], initializer=initializer())
logits = [tf.matmul(output, W) + b for output in outputs]

loss = sequence_loss(logits, targets, sequence_length * [1.0])
train_step = tf.train.AdamOptimizer().minimize(loss)

# ------------------
# YOUR SAMPLER GRAPH HERE

# place your sampler graph here it will look a lot like your
# computation graph, except with a "batch_size" of 1.

# remember, we want to reuse the parameters of the cell and whatever
# parameters you used to transform state outputs to logits!

s_input = tf.placeholder(tf.int32, [1], name='s_inputs')
s_input_onehot = tf.one_hot(s_input, vocab_size, name='s_input_onehot')
s_initial_state = rnnCell.zero_state(1, tf.float32)
s_outputs, s_final_state = rnn_decoder([s_input_onehot], s_initial_state, rnnCell)

s_logits = [tf.matmul(s_output, W) + b for s_output in s_outputs]

#
# ==================================================================
# ==================================================================
# ==================================================================
#


def sample(num=200, prime='ab'):
    # prime the pump
    # generate an initial state. this will be a list of states, one for
    # each layer in the multicell.
    s_state = sess.run(s_initial_state)

    # for each character, feed it into the sampler graph and
    # update the state.
    for char in prime[:-1]:
        x = np.ravel(data_loader.vocab[char]).astype('int32')
        feed = {s_input: x}
        for i, s in enumerate(s_initial_state):
            feed[s] = s_state[i]
        s_state = sess.run(s_final_state, feed_dict=feed)

    # now we have a primed state vector; we need to start sampling.
    ret = prime
    char = prime[-1]
    for n in range(num):
        x = np.ravel(data_loader.vocab[char]).astype('int32')

        # plug the most recent character in...
        feed = {s_input: x}
        for i, s in enumerate(s_initial_state):
            feed[s] = s_state[i]
        ops = [s_logits]
        ops.extend(list(s_final_state))

        retval = sess.run(ops, feed_dict=feed)

        s_probsv = retval[0]
        s_state = retval[1:]

        # ...and get a vector of probabilities out!

        # now sample (or pick the argmax)
        sample = np.argmax(s_probsv[0])
        # sample = np.random.choice(vocab_size, p=s_probsv[0])

        pred = data_loader.chars[sample]
        ret += pred
        char = pred
    return ret

#
# ==================================================================
# ==================================================================
# ==================================================================
#

sess = tf.Session()
sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter("./tf_logs", graph=sess.graph)

lts = []

print "FOUND %d BATCHES" % data_loader.num_batches
for j in range(1000):
    state = sess.run(initial_state)
    data_loader.reset_batch_pointer()

    for i in range(data_loader.num_batches):
        x, y = data_loader.next_batch()

        # we have to feed in the individual states of the MultiRNN cell
        feed = {in_ph: x, targ_ph: y}
        for k, s in enumerate(initial_state):
            feed[s] = state[k]

        ops = [train_step, loss]
        ops.extend(list(final_state))

        # retval will have at least 3 entries:
        # 0 is None (triggered by the train_step op)
        # 1 is the loss
        # 2+ are the new final states of the MultiRNN cell
        retval = sess.run(ops, feed_dict=feed)

        lt = retval[1]
        state = retval[2:]

        if i % 1000 == 0:
            print "%d %d\t%.4f" % (j, i, lt)
            lts.append(lt)

    # print sample(num=60, prime="And ")
    print sample(num=60, prime="Alice ")
    # print sample(num=60, prime="Quoth ")
    # print sample(num=60, prime="ababab")
    # print sample(num=60, prime="foo ba")
    # print sample(num=60, prime="abcdab")

summary_writer.close()

#
# ==================================================================
# ==================================================================
# ==================================================================
#

#import matplotlib
#import matplotlib.pyplot as plt
#plt.plot(lts)
#plt.show()
