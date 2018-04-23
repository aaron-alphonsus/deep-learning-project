# ------------------------Download and read MNIST data--------------------------

from tensorflow.examples.tutorials.mnist import input_data
# mnist is a lightweight class that stores training, validation and testing sets
#   as NumPy arrays. Also provides function for iterating through data 
#   minibatches
mnist = input_data.read_data_sets('./../MNIST_data', one_hot=True) 

# ---------------------Start TensorFlow InteractiveSession----------------------

import tensorflow as tf
sess = tf.InteractiveSession() # Allows more flexible structuring of code

# -------------------------Placeholders and Variables---------------------------

# Create placeholder nodes for input images and target classes
x = tf.placeholder(tf.float32, shape=[None, 784]) # 28*28 image linearized
y_ = tf.placeholder(tf.float32, shape=[None, 10]) # digit class (0 to 9)

# Define weights W and biases b (provides initial values, tensors full of zeros)
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Creates the Saver object whose restore() function we will use later
saver = tf.train.Saver()

# Initialize all variables (W and B here) before using them within session
sess.run(tf.global_variables_initializer())

# Implement regression model (output = input images * weights + bias)
y = tf.matmul(x, W) + b

# restore the previously saved model. #notrainingrequired
saver.restore(sess, './savefiles/my-model')

# -----------------------------Evaluating the Model-----------------------------

# Checks whether the output y matches the target y_. Gives a list of booleans.
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# Convert booleans to floats and take the mean to get accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Test accuracy %g%%" % (accuracy.eval(feed_dict={
    x: mnist.test.images, y_:mnist.test.labels})*100))
