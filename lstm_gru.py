import tensorflow as tf
import numpy as np
from textloader import TextLoader
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.contrib.rnn import RNNCell
from gru_cell import GruCell

# Run stack of cells
def run_cells(inputs, rnn_cell, init_st):
    outputs = []
    state = init_st
    for input in inputs:
        # for cell in rnn_cells:
        output, state = rnn_cell(input, state)
        outputs.append(output)

    return outputs

batch_size = 50
sequence_length = 50

data_loader = TextLoader( ".", batch_size, sequence_length )

vocab_size = data_loader.vocab_size  # dimension of one-hot encodings
state_dim = 128

LR = 0.01
num_layers = 2

tf.reset_default_graph()

with tf.variable_scope("rnn_vars") as scope:
    in_ph = tf.placeholder( tf.int32, [ batch_size, sequence_length ], name='inputs' )
    targ_ph = tf.placeholder( tf.int32, [ batch_size, sequence_length ], name='targets' )
    in_onehot = tf.one_hot( in_ph, vocab_size, name="input_onehot" )

    inputs = tf.split( in_onehot, sequence_length, axis=1 )
    inputs = [ tf.squeeze(input_, [1]) for input_ in inputs ]
    targets = tf.split( targ_ph, sequence_length, axis=1 )

    cells = []
    for i in range(num_layers):
        cells.append(GruCell(state_dim, vocab_size, name='cell' + str(i)))
    mrnn_cell = tf.contrib.rnn.MultiRNNCell(cells)
    init_state = mrnn_cell.zero_state(batch_size, tf.float32)
    outputs, final_state = tf.contrib.legacy_seq2seq.rnn_decoder(inputs, init_state, mrnn_cell)
    outputs = tf.reshape(tf.convert_to_tensor(outputs), [-1, state_dim])

    dense_w = tf.get_variable("dense_w", [state_dim, vocab_size])
    dense_b = tf.get_variable("dense_b", [vocab_size])

    dense = tf.matmul(outputs, dense_w) + dense_b
    probs = tf.nn.softmax(dense)
    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([dense], [targets], [tf.ones([batch_size * sequence_length])])
    loss = tf.reduce_sum(loss) / batch_size / sequence_length
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),
                                      1.0)
    optimizer = tf.train.AdamOptimizer(LR)
    train_op = optimizer.apply_gradients(zip(grads, tvars))

    scope.reuse_variables()
    s_in_ph = tf.placeholder(tf.int32, [1], name='s_inputs')
    s_in_onehot = tf.one_hot(s_in_ph, vocab_size, name="s_input_onehot")

    s_inputs = [s_in_onehot]
    s_init_state = mrnn_cell.zero_state(1, tf.float32)

    s_outputs, s_final_state = tf.contrib.legacy_seq2seq.rnn_decoder(s_inputs, s_init_state, mrnn_cell)
    s_outputs = tf.reshape(tf.concat(s_outputs, 1), [-1, state_dim])

    s_dense_w = tf.get_variable("dense_w", [state_dim, vocab_size])
    s_dense_b = tf.get_variable("dense_b", [vocab_size])

    s_dense = tf.matmul(s_outputs, s_dense_w) + s_dense_b
    s_probs = tf.nn.softmax(s_dense)

def sample( num=200, prime='ab' ):
    s_state = sess.run( s_init_state )
    for char in prime[:-1]:
        x = np.ravel( data_loader.vocab[char] ).astype('int32')
        feed = { s_in_ph:x }
        for i, s in enumerate( s_init_state ):
            feed[s] = s_state[i]
        s_state = sess.run( s_final_state, feed_dict=feed )

    ret = prime
    char = prime[-1]
    for n in range(num):
        x = np.ravel( data_loader.vocab[char] ).astype('int32')

        feed = { s_in_ph:x }
        for i, s in enumerate( s_init_state ):
            feed[s] = s_state[i]
        ops = [s_probs]
        ops.extend( list(s_final_state) )

        retval = sess.run( ops, feed_dict=feed )

        s_probsv = retval[0]
        s_state = retval[1:]

        sample = np.random.choice( vocab_size, p=s_probsv[0] )

        pred = data_loader.chars[sample]
        ret += pred
        char = pred

    return ret

sess = tf.Session()
sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter("./tf_logs", graph=sess.graph)

lts = []

print("FOUND %d BATCHES" % data_loader.num_batches)

for j in range(150):
    state = sess.run( init_state )
    data_loader.reset_batch_pointer()

    for i in range( data_loader.num_batches ):
        x,y = data_loader.next_batch()

        feed = { in_ph: x, targ_ph: y }
        for k, s in enumerate( init_state ):
            feed[s] = state[k]

        ops = [train_op,loss]
        ops.extend( list(final_state) )

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

summary_writer.close()
