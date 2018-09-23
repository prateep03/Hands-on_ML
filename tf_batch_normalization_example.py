import numpy as np
import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
from functools import partial

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed=seed)
    np.random.seed(seed)

# mnist = input_data.read_data_sets("datasets/mnist/")
reset_graph()

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10
n_epochs = 40
batch_size = 50
learning_rate = 0.01

# Set up Graph
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

'''
 Leaky Rectified Linear Unit(LRelu)
    lrelu(z) = max(0.01 * z, z)
'''
def lrelu(z, name=None):
    return tf.maximum(0.01 * z, z, name=name)

'''
 Expoential Linear Unit(ELU)
    - elu(z) = (alpha * (exp(z) - 1)) [z < 0] + z [z>=0]
'''
def elu(z, alpha=1):
    return np.where(z < 0, alpha * (np.exp(z) - 1), z)

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.elu, name="hidden1")
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.elu, name="hidden2")
    logits = tf.layers.dense(hidden2, n_outputs, name="output")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, n_inputs) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, n_inputs) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

print("Basic tf training\n")
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size=batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y:y_batch})

        if epoch % 5 == 0:
            acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_valid = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
            print(epoch, "Batch accuracy:", acc_batch, "Validation accuracy:", acc_valid)


"""
Accuracy measures without batch normalization
-----------------------------------------------------
0 Batch accuracy: 0.88 Validation accuracy: 0.9036
5 Batch accuracy: 0.94 Validation accuracy: 0.9338
10 Batch accuracy: 0.9 Validation accuracy: 0.9532
15 Batch accuracy: 0.9 Validation accuracy: 0.9622
20 Batch accuracy: 1.0 Validation accuracy: 0.9656
25 Batch accuracy: 1.0 Validation accuracy: 0.9698
30 Batch accuracy: 0.98 Validation accuracy: 0.9698
35 Batch accuracy: 0.98 Validation accuracy: 0.9726
"""

reset_graph()

# Set up Graph
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

training = tf.placeholder_with_default(False, shape=(), name="training")

with tf.name_scope("dnn-bn"):
    my_batch_norm_layer = partial(tf.layers.batch_normalization, training=training, momentum=0.9)

    hidden1 = tf.layers.dense(X, n_hidden1, activation=lrelu, name="hidden1")
    bn1 = my_batch_norm_layer(hidden1)
    bn1_act = tf.nn.elu(bn1)

    hidden2 = tf.layers.dense(bn1_act, n_hidden2, activation=lrelu, name="hidden2")
    bn2 = my_batch_norm_layer(hidden2)
    bn2_act = tf.nn.elu(bn2)

    logits_before_bn = tf.layers.dense(bn2_act, n_outputs, name="output")
    logits = my_batch_norm_layer(logits_before_bn)

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

'''
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)
'''

"""
Gradient clipping
"""
with tf.name_scope("train-gc"):
    threshold = 1.0
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    grads_and_vars = optimizer.compute_gradients(loss)
    capped_gvs = [(tf.clip_by_value(grad, -threshold, threshold), var) for grad, var in grads_and_vars]
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

print("Training with batch normalization\n")
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size=batch_size):
            sess.run([training_op, extra_update_ops], feed_dict={training: True, X: X_batch, y:y_batch})

        if epoch % 5 == 0:
            acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_valid = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
            print(epoch, "Batch accuracy:", acc_batch, "Validation accuracy:", acc_valid)

        if epoch % 10 == 0:
            accuracy_test = accuracy.eval(feed_dict={training: False, X: X_test, y: y_test})
            print(epoch, "Test accuracy:", accuracy_test)

"""
Accuracy measures with batch normalization
-----------------------------------------------------
0 Batch accuracy: 0.9 Validation accuracy: 0.906
5 Batch accuracy: 0.94 Validation accuracy: 0.9458
10 Batch accuracy: 0.94 Validation accuracy: 0.9654
15 Batch accuracy: 0.92 Validation accuracy: 0.9714
20 Batch accuracy: 1.0 Validation accuracy: 0.975
25 Batch accuracy: 0.98 Validation accuracy: 0.9766
30 Batch accuracy: 1.0 Validation accuracy: 0.9772
35 Batch accuracy: 1.0 Validation accuracy: 0.9762 <-- better
"""

"""
Accuracy measures with batch normalization, lrelu
-----------------------------------------------------
0 Batch accuracy: 0.88 Validation accuracy: 0.9116
5 Batch accuracy: 0.94 Validation accuracy: 0.9608
10 Batch accuracy: 0.96 Validation accuracy: 0.9714
15 Batch accuracy: 0.96 Validation accuracy: 0.9756
20 Batch accuracy: 1.0 Validation accuracy: 0.9768
25 Batch accuracy: 1.0 Validation accuracy: 0.9788
30 Batch accuracy: 1.0 Validation accuracy: 0.9788
35 Batch accuracy: 1.0 Validation accuracy: 0.979
"""
