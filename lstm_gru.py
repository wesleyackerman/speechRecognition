

import tensorflow as tf
import numpy as np
from textloader import TextLoader
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.contrib.rnn import RNNCell


class GruCell(RNNCell):
    def __init__(self, s_dim, v_size, reuse=None, name=None):
        super(GruCell, self).__init__(_reuse=reuse, name=name)
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
        w_xr = tf.get_variable(self._name + 'w_xr', [input_dim, self.state_dim], initializer=tf.variance_scaling_initializer)
        w_hr = tf.get_variable(self._name + 'w_hr', [self.state_dim, self.state_dim], initializer=tf.variance_scaling_initializer)
        b_r = tf.get_variable(self._name + 'b_r', [self.state_dim], initializer=tf.variance_scaling_initializer)

        w_xz = tf.get_variable(self._name + 'w_xz', [input_dim, self.state_dim], initializer=tf.variance_scaling_initializer)
        w_hz = tf.get_variable(self._name + 'w_hz', [self.state_dim, self.state_dim], initializer=tf.variance_scaling_initializer)
        b_z = tf.get_variable(self._name + 'b_z', [self.state_dim])

        w_xh = tf.get_variable(self._name + 'w_xh', [input_dim, self.state_dim], initializer=tf.variance_scaling_initializer)
        w_hh = tf.get_variable(self._name + 'w_hh', [self.state_dim, self.state_dim], initializer=tf.variance_scaling_initializer)
        b_h = tf.get_variable(self._name + 'b_h', [self.state_dim], initializer=tf.variance_scaling_initializer)

        r_t = tf.sigmoid(tf.matmul(inputs, w_xr) + tf.matmul(state, w_hr) + b_r, name=self._name + 'r_sigmoid')
        z_t = tf.sigmoid(tf.matmul(inputs, w_xz) + tf.matmul(state, w_hz) + b_z, name=self._name + 'z_sigmoid')
        h_candidate = tf.tanh(tf.matmul(inputs, w_xh) + tf.matmul(r_t * state, w_hh) + b_h, name=self._name + 'h_tanh')

        new_state = z_t * state + (1 - z_t) * h_candidate
        return new_state, new_state

#
# -------------------------------------------
#
# Global variables

batch_size = 50
sequence_length = 50

data_loader = TextLoader( ".", batch_size, sequence_length )

vocab_size = data_loader.vocab_size  # dimension of one-hot encodings
state_dim = 128

LR = 0.01
num_layers = 2

tf.reset_default_graph()

#
# ==================================================================
# ==================================================================
# ==================================================================
#
with tf.variable_scope("rnn_vars") as scope:

    # define placeholders for our inputs.
    # in_ph is assumed to be [batch_size,sequence_length]
    # targ_ph is assumed to be [batch_size,sequence_length]

    in_ph = tf.placeholder( tf.int32, [ batch_size, sequence_length ], name='inputs' )
    targ_ph = tf.placeholder( tf.int32, [ batch_size, sequence_length ], name='targets' )
    in_onehot = tf.one_hot( in_ph, vocab_size, name="input_onehot" )

    inputs = tf.split( in_onehot, sequence_length, axis=1 )
    inputs = [ tf.squeeze(input_, [1]) for input_ in inputs ]
    targets = tf.split( targ_ph, sequence_length, axis=1 )

    # at this point, inputs is a list of length sequence_length
    # each element of inputs is [batch_size,vocab_size]

    # targets is a list of length sequence_length
    # each element of targets is a 1D vector of length batch_size

    # ------------------
    # YOUR COMPUTATION GRAPH HERE

    cells = []
    for i in range(num_layers):
        cells.append(GruCell(state_dim, vocab_size, name='cell' + str(i)))
        #cells.append(BasicLSTMCell(state_dim))
    mrnn_cell = tf.contrib.rnn.MultiRNNCell(cells)
    init_state = mrnn_cell.zero_state(batch_size, tf.float32)
    # create a BasicLSTMCell
    #   use it to create a MultiRNNCell
    #   use it to create an initial_state
    #     note that initial_state will be a *list* of tensors!
    outputs, final_state = tf.contrib.legacy_seq2seq.rnn_decoder(inputs, init_state, mrnn_cell)
    outputs = tf.reshape(tf.convert_to_tensor(outputs), [-1, state_dim])

    dense_w = tf.get_variable("dense_w", [state_dim, vocab_size])
    dense_b = tf.get_variable("dense_b", [vocab_size])

    dense = tf.matmul(outputs, dense_w) + dense_b
    probs = tf.nn.softmax(dense)
    # call seq2seq.rnn_decoder

    # loss = tf.contrib.legacy_seq2seq.sequence_loss([dense], [targets], [np.ones(batch_size * sequence_length, np.float32)])
    # train_op = tf.train.AdamOptimizer(LR).minimize(loss)
    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([dense], [targets], [tf.ones([batch_size * sequence_length])])
    loss = tf.reduce_sum(loss) / batch_size / sequence_length
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),
                                      1.0)
    optimizer = tf.train.AdamOptimizer(LR)
    train_op = optimizer.apply_gradients(zip(grads, tvars))

    # transform the list of state outputs to a list of logits.
    # use a linear transformation.

    # call seq2seq.sequence_loss

    # create a training op using the Adam optimizer

    # ------------------
    # YOUR SAMPLER GRAPH HERE
    scope.reuse_variables()
    s_in_ph = tf.placeholder(tf.int32, [1], name='s_inputs')
    s_in_onehot = tf.one_hot(s_in_ph, vocab_size, name="s_input_onehot")

    s_inputs = [s_in_onehot]
    #s_inputs = tf.split(s_in_onehot, 1, axis=1)
    #s_inputs = [tf.squeeze(input_, [1]) for input_ in s_inputs]

    s_init_state = mrnn_cell.zero_state(1, tf.float32)
    # create a BasicLSTMCell
    #   use it to create a MultiRNNCell
    #   use it to create an initial_state
    #     note that initial_state will be a *list* of tensors!
    s_outputs, s_final_state = tf.contrib.legacy_seq2seq.rnn_decoder(s_inputs, s_init_state, mrnn_cell)
    s_outputs = tf.reshape(tf.concat(s_outputs, 1), [-1, state_dim])

    s_dense_w = tf.get_variable("dense_w", [state_dim, vocab_size])
    s_dense_b = tf.get_variable("dense_b", [vocab_size])

    s_dense = tf.matmul(s_outputs, s_dense_w) + s_dense_b
    s_probs = tf.nn.softmax(s_dense)

    # place your sampler graph here it will look a lot like your
    # computation graph, except with a "batch_size" of 1.

    # remember, we want to reuse the parameters of the cell and whatever
    # parameters you used to transform state outputs to logits!

#
# ==================================================================
# ==================================================================
# ==================================================================


def sample( num=200, prime='ab' ):

    # prime the pump 

    # generate an initial state. this will be a list of states, one for
    # each layer in the multicell.
    s_state = sess.run( s_init_state )

    # for each character, feed it into the sampler graph and
    # update the state.
    for char in prime[:-1]:
        x = np.ravel( data_loader.vocab[char] ).astype('int32')
        feed = { s_in_ph:x }
        for i, s in enumerate( s_init_state ):
            feed[s] = s_state[i]
        s_state = sess.run( s_final_state, feed_dict=feed )

    # now we have a primed state vector; we need to start sampling.
    ret = prime
    char = prime[-1]
    for n in range(num):
        x = np.ravel( data_loader.vocab[char] ).astype('int32')

        # plug the most recent character in...
        feed = { s_in_ph:x }
        for i, s in enumerate( s_init_state ):
            feed[s] = s_state[i]
        ops = [s_probs]
        ops.extend( list(s_final_state) )

        retval = sess.run( ops, feed_dict=feed )

        s_probsv = retval[0]
        s_state = retval[1:]

        # ...and get a vector of probabilities out!

        # now sample (or pick the argmax)
        # sample = np.argmax( s_probsv[0] )
        sample = np.random.choice( vocab_size, p=s_probsv[0] )

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
sess.run( tf.global_variables_initializer() )
summary_writer = tf.summary.FileWriter( "./tf_logs", graph=sess.graph )

lts = []

print("FOUND %d BATCHES" % data_loader.num_batches)

for j in range(150):

    state = sess.run( init_state )
    data_loader.reset_batch_pointer()

    for i in range( data_loader.num_batches ):
        
        x,y = data_loader.next_batch()

        # we have to feed in the individual states of the MultiRNN cell
        feed = { in_ph: x, targ_ph: y }
        for k, s in enumerate( init_state ):
            feed[s] = state[k]

        ops = [train_op,loss]
        ops.extend( list(final_state) )

        # retval will have at least 3 entries:
        # 0 is None (triggered by the optim op)
        # 1 is the loss
        # 2+ are the new final states of the MultiRNN cell
        retval = sess.run( ops, feed_dict=feed )

        lt = retval[1]
        state = retval[2:]

        if i%1000==0:
            print("%d %d\t%.4f" % ( j, i, lt ))
            lts.append( lt )

    print(sample( num=100, prime=np.random.choice(['And', 'The', 'There', '\'You', '\'That', '\'The'])))
    if j % 10 == 0:
        with open("output_two_towers_gru.txt", 'w') as file:
            for i in range(40):
                file.write(sample( num=100, prime=np.random.choice(['And', 'The', 'There', '\'You', '\'That', '\'The'])))
                file.write('\n')
#    print(sample( num=60, prime="ababab" ))
#    print(sample( num=60, prime="foo ba" ))
#    print(sample( num=60, prime="abcdab" ))


summary_writer.close()

#
# ==================================================================
# ==================================================================
# ==================================================================
#

#import matplotlib
#import matplotlib.pyplot as plt
#plt.plot( lts )
#plt.show()