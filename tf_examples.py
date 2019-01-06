import numpy as np
from sklearn.datasets import fetch_california_housing
import tensorflow as tf

housing = fetch_california_housing()

"""
Linear regression with tensorflow and numpy
"""
def linear_regression(housing):
    m, n = housing.data.shape
    housing_data_plus_bias = np.c_[np.ones(shape=(m, 1)), housing.data]

    X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
    y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
    XT = tf.transpose(X)
    theta = tf.matmul( tf.matmul( tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

    print("In tensorflow\n")
    with tf.Session() as sess:
        theta_value = theta.eval()
        print(theta_value)

    print("In numpy\n")
    X = housing_data_plus_bias
    y = housing.target.reshape(-1, 1)
    theta_numpy = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    print(theta_numpy)

# linear_regression(housing)

def manually_compute_gradients(housing):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_housing_data = scaler.fit_transform(housing.data)
    m, n = scaled_housing_data.shape
    scaled_housing_data_plus_bias = np.c_[np.ones(shape=(m, 1)), scaled_housing_data]

    n_epochs = 1000
    learning_rate = 0.01

    X = tf.constant(value=scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
    y = tf.constant(value=housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
    theta = tf.Variable(tf.random_uniform(shape=[n + 1, 1], minval=-1.0, maxval=1.0), name="theta")
    y_pred = tf.matmul(X, theta, name="predictions")
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")
    # gradients = 2 / m * tf.matmul(tf.transpose(X), error)
    # for auto-computing gradients
    # gradients = tf.gradients(mse, [theta])[0]
    # training_op = tf.assign(theta, theta - learning_rate * gradients)

    ## Using gradient-descent optimizer
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    training_op = optimizer.minimize(mse)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(n_epochs):
            if epoch % 100 == 0:
                print("Epoch", epoch, "MSE =", mse.eval())
            sess.run(training_op)

        best_theta = theta.eval()
        print("Best theta =", best_theta)


manually_compute_gradients(housing)