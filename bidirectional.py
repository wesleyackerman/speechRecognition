import h5py
import tensorflow as tf
import numpy as np
from gru_cell import GruCell
import matplotlib.pyplot as plt
import time
import soundfile as sf


prefix = 'example_prefix_'


# Run stack of cells
def run_cells(inpts, rnn_cells, init_st):
    outpts = []
    st = init_st
    with tf.variable_scope("run_cells_scope"):
        for c, inpt in enumerate(inpts):
            new_st = []
            outpt = None
            for d, cell in enumerate(rnn_cells):
                outpt, temp = cell(inpt, st[d])
                new_st.append(temp)
                inpt = outpt
            outpts.append(outpt)
            st = tuple(new_st)
            tf.get_variable_scope().reuse_variables()
    return outpts, st


def create_init_state(b_size, st_dim, n_layers):
    init_st = []
    for i in range(n_layers):
        init_st.append(tf.zeros((b_size, st_dim)))
    return tuple(init_st)


def sequence_loss(logits, targets, vocab_size):
    targets = tf.convert_to_tensor(targets)
    logits = tf.reshape(logits, [targets.shape[0], targets.shape[1], vocab_size])
    norm_probs = tf.nn.softmax(logits, dim=-1)
    one_hot_targets = tf.squeeze(tf.one_hot(targets, vocab_size))
    return tf.reduce_mean(-tf.reduce_sum(one_hot_targets * tf.log(norm_probs), reduction_indices=[-1]))
    #return tf.reduce_sum(tf.abs(one_hot_targets - norm_probs))

def get_time_str():
    return time.strftime("%d%b%Y-%H:%M:%S", time.gmtime())


batch_size = 50
n_epochs = 80
data_file = "spokenverbs/db.dog.hdf5"

f = h5py.File(data_file, 'r')
data = f['data'].value
labels = f['labels'].value
n_instances = data.shape[0]
n_batches = n_instances // batch_size
n_classes = np.unique(labels).shape[0]
sequence_length = data.shape[1]
n_values = data.shape[2]

state_dim = 128

LR = 0.01
num_layers = 2

tf.reset_default_graph()

with tf.variable_scope("rnn_vars") as scope:
    in_ph = tf.placeholder(tf.float32, [batch_size, sequence_length, n_values], name='inputs')
    targ_ph = tf.placeholder(tf.int32, [batch_size, 1], name='targets')
    # in_onehot = tf.one_hot(in_ph, n_classes, name="input_onehot")

    # Change dims to sequence length, then batch, then n_values so we can iterate over sequence items
    inputs = tf.transpose(in_ph, perm=[1, 0, 2])
    inputs = tf.split(inputs, sequence_length)
    inputs = [tf.squeeze(input_, [0]) for input_ in inputs]
    # inputs = tf.split(in_ph, sequence_length, axis=1)
    # inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
    # targets = tf.split(targ_ph, sequence_length, axis=1)

    cells = []
    for i in range(num_layers):
        cells.append(GruCell(state_dim, n_classes, name='cell' + str(i)))
    init_state = create_init_state(batch_size, state_dim, num_layers)
    outputs, final_state = run_cells(inputs, cells, init_state)
    outputs = tf.transpose(tf.convert_to_tensor(outputs), perm=[1, 0, 2])
    outputs = tf.reshape(outputs, [batch_size, -1])

    dense_w = tf.get_variable("dense_w", [state_dim * sequence_length, n_classes])
    dense_b = tf.get_variable("dense_b", [n_classes])

    dense = tf.matmul(outputs, dense_w) + dense_b
    probs = tf.nn.softmax(dense)
    loss = sequence_loss(dense, targ_ph, n_classes)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),
                                      1.0)
    optimizer = tf.train.AdamOptimizer(LR)
    train_op = optimizer.apply_gradients(zip(grads, tvars))

    scope.reuse_variables()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter("./tf_logs", graph=sess.graph)

lts = []

for j in range(n_epochs):
    state = sess.run(init_state)

    for i in range(n_batches):
        x, y = data[i*batch_size:(i*batch_size) + batch_size], labels[i*batch_size:(i*batch_size) + batch_size]

        feed = {in_ph: x, targ_ph: np.expand_dims(y, -1)}
        for k, s in enumerate(init_state):
            feed[s] = state[k]

        ops = [train_op, loss]
        ops.extend(list(final_state))

        retval = sess.run(ops, feed_dict=feed)

        lt = retval[1]
        state = retval[2:]
        print lt

fig = plt.figure(1, figsize=(6, 6))
x_values = np.arange(j + 1) + 1
plt.plot(x_values, np.array(lts))
plt.ylabel("Sequence Loss")
plt.xlabel("Epoch")
plt.title("Sequence loss across epochs")
plt.savefig(prefix + get_time_str() + "_loss_graph.png", bbox_inches='tight')

summary_writer.close()
