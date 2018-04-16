import tensorflow as tf
import numpy as np
from textloader import TextLoader
from gru_cell import GruCell
import matplotlib.pyplot as plt
import time


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
    logits = tf.reshape(logits, [int(targets.shape[0]), int(targets.shape[1]), vocab_size])
    norm_probs = tf.nn.softmax(logits, dim=-1)
    one_hot_targets = tf.squeeze(tf.one_hot(targets, vocab_size))
    return tf.reduce_mean(-tf.reduce_sum(one_hot_targets * tf.log(norm_probs), reduction_indices=[-1]))
    #return tf.reduce_sum(tf.abs(one_hot_targets - norm_probs))

def get_time_str():
    return time.strftime("%d%b%Y-%H:%M:%S", time.gmtime())


prefix = 'example_prefix_'

batch_size = 50
sequence_length = 50
n_epochs = 80

data_loader = TextLoader(".", batch_size, sequence_length)

vocab_size = data_loader.vocab_size  # dimension of one-hot encodings
state_dim = 128

LR = 0.01
num_layers = 2

seed_words = ['And', 'The', 'There', 'With' 'He', 'She', 'A', '\'And', '\'The', '\'There', '\'We']
write_filename = prefix + "output_test_loss.txt"

tf.reset_default_graph()

with tf.variable_scope("rnn_vars") as scope:
    in_ph = tf.placeholder(tf.int32, [batch_size, sequence_length], name='inputs')
    targ_ph = tf.placeholder(tf.int32, [batch_size, sequence_length], name='targets')
    in_onehot = tf.one_hot(in_ph, vocab_size, name="input_onehot")

    inputs = tf.split(in_onehot, sequence_length, axis=1)
    inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
    targets = tf.split(targ_ph, sequence_length, axis=1)

    cells = []
    for i in range(num_layers):
        cells.append(GruCell(state_dim, vocab_size, name='cell' + str(i)))
    # mrnn_cell = tf.contrib.rnn.MultiRNNCell(cells)
    init_state = create_init_state(batch_size, state_dim, num_layers)
    # outputs, final_state = tf.contrib.legacy_seq2seq.rnn_decoder(inputs, init_state, mrnn_cell)
    outputs, final_state = run_cells(inputs, cells, init_state)
    outputs = tf.reshape(tf.convert_to_tensor(outputs), [-1, state_dim])

    dense_w = tf.get_variable("dense_w", [state_dim, vocab_size])
    dense_b = tf.get_variable("dense_b", [vocab_size])

    dense = tf.matmul(outputs, dense_w) + dense_b
    probs = tf.nn.softmax(dense)
    loss = sequence_loss(dense, targets, vocab_size)
    #loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([dense], [targets], [tf.ones([batch_size * sequence_length])])
    #loss = tf.reduce_sum(loss) / batch_size / sequence_length
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),
                                      1.0)
    optimizer = tf.train.AdamOptimizer(LR)
    train_op = optimizer.apply_gradients(zip(grads, tvars))

    scope.reuse_variables()
    s_in_ph = tf.placeholder(tf.int32, [1], name='s_inputs')
    s_in_onehot = tf.one_hot(s_in_ph, vocab_size, name="s_input_onehot")

    s_inputs = [s_in_onehot]
    # s_init_state = mrnn_cell.zero_state(1, tf.float32)
    s_init_state = create_init_state(1, state_dim, num_layers)
    # s_outputs, s_final_state = tf.contrib.legacy_seq2seq.rnn_decoder(s_inputs, s_init_state, mrnn_cell)
    s_outputs, s_final_state = run_cells(s_inputs, cells, s_init_state)
    s_outputs = tf.reshape(tf.concat(s_outputs, 1), [-1, state_dim])

    s_dense_w = tf.get_variable("dense_w", [state_dim, vocab_size])
    s_dense_b = tf.get_variable("dense_b", [vocab_size])

    s_dense = tf.matmul(s_outputs, s_dense_w) + s_dense_b
    s_probs = tf.nn.softmax(s_dense)


def sample(num=200, prime='ab'):
    s_state = sess.run(s_init_state)
    for char in prime[:-1]:
        x = np.ravel(data_loader.vocab[char]).astype('int32')
        feed = {s_in_ph:x}
        for i, s in enumerate(s_init_state):
            feed[s] = s_state[i]
        s_state = sess.run(s_final_state, feed_dict=feed)

    ret = prime
    char = prime[-1]
    for n in range(num):
        x = np.ravel(data_loader.vocab[char]).astype('int32')

        feed = {s_in_ph: x}
        for i, s in enumerate(s_init_state):
            feed[s] = s_state[i]
        ops = [s_probs]
        ops.extend(list(s_final_state))

        retval = sess.run(ops, feed_dict=feed)

        s_probsv = retval[0]
        s_state = retval[1:]

        if ret[-1] == " ":
            sample = np.random.choice(vocab_size, p=s_probsv[0])
        else:
            sample = np.argmax(s_probsv[0])

        pred = data_loader.chars[sample]
        ret += pred
        char = pred

    return ret


sess = tf.Session()
sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter("./tf_logs", graph=sess.graph)

lts = []
with open(write_filename, 'w') as file:
    file.write("")

print("FOUND %d BATCHES" % data_loader.num_batches)

for j in range(n_epochs):
    state = sess.run(init_state)
    data_loader.reset_batch_pointer()

    for i in range(data_loader.num_batches):
        x, y = data_loader.next_batch()

        feed = {in_ph: x, targ_ph: y}
        for k, s in enumerate(init_state):
            feed[s] = state[k]

        ops = [train_op, loss]
        ops.extend(list(final_state))

        retval = sess.run(ops, feed_dict=feed)

        lt = retval[1]
        state = retval[2:]

        if i % 1000 == 0:
            print("%d %d\t%.4f" % (j, i, lt))
            lts.append(lt)

    print(sample(num=300, prime=np.random.choice(seed_words).encode("utf8")))
    if j % 5 == 0:
        with open(write_filename, 'a') as file:
            file.write("epoch " + str(j) + "\n")
            for i in range(5):
                file.write(sample(num=300, prime=np.random.choice(seed_words)).encode("utf8"))
                file.write('\n')

fig = plt.figure(1, figsize=(6, 6))
x_values = np.arange(len(lts)) + 1
plt.plot(x_values, np.array(lts))
plt.ylabel("Sequence Loss")
plt.xlabel("Epoch")
plt.title("Sequence loss across epochs")
plt.savefig(prefix + get_time_str() + "_loss_graph.png", bbox_inches='tight')

summary_writer.close()
