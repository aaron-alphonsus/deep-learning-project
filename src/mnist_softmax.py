# ------------------------Download and read MNIST data--------------------------

from tensorflow.examples.tutorials.mnist import input_data
# mnist is a lightweight class that stores training, validation and testing sets
# as NumPy arrays. Also provides function for iterating through data minibatches
mnist = input_data.read_data_sets('MNIST_data', one_hot=True) # one_hot = binary

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

# Initialize all variables (W and B here) before using them within session
sess.run(tf.global_variables_initializer())

# Implement regression model (output = input images * weights + bias)
y = tf.matmul(x, W) + b

# Define the loss function (cross-entropy between target and softmax activation)
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# -----------------------------Training the model-------------------------------

# Steepest gradient descent, step size = 0.5
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# repeatedly running train_step trains the model
for _ in range(1000):
    # 100 training examples in each iteration
    batch = mnist.train.next_batch(100)
    # replace placeholders with actual training examples
    train_step.run(feed_dict = {x: batch[0], y_: batch[1]})

# -----------------------------Evaluating the Model-----------------------------

# Checks whether the output y matches the target y_. Gives a list of booleans.
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# Convert booleans to floats and take the mean to get accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_:mnist.test.labels}))
