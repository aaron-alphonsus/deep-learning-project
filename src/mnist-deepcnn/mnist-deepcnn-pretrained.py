# ------------------------Download and read MNIST data--------------------------
 
from tensorflow.examples.tutorials.mnist import input_data
# mnist is a lightweight class that stores training, validation and testing sets
# as NumPy arrays. Also provides function for iterating through data minibatches
mnist = input_data.read_data_sets('./../MNIST_data', one_hot=True) 
 
# ---------------------Start TensorFlow InteractiveSession----------------------
 
import tensorflow as tf
sess = tf.InteractiveSession() # Allows more flexible structuring of code
 
# -------------------------Placeholders and Variables---------------------------
 
# Create placeholder nodes for input images and target classes
x = tf.placeholder(tf.float32, shape=[None, 784]) # 28*28 image linearized
y_ = tf.placeholder(tf.float32, shape=[None, 10]) # digit class (0 to 9)

# -----------------------------Weight Initialization----------------------------

def weight_variable(shape):
    """
    Takes shape of weight tensor and returning it after initializing it with a   
    little noise. This is done to achieve 'symmetry breaking' and to prevent     
    gradients of 0. 
    """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """
    Initializing the bias variable as slightly positive to prevent ReLU neurons
    from going dead.
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# ---------------------------Convolution and Pooling----------------------------

def conv2d(x, W):
    """
    Simple convolution with a stride of one. Convolutions are zero padded to
    make sure output is the same size as the input.
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    """Plain max pooling over 2x2 blocks."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], 
                          padding='SAME')

# --------------------------First Convolutional Layer---------------------------

# Convolves input image with the weight tensor, adds the bias, applies ReLU, and 
#   applies max pooling. The image size is reduced to 14x14 after the            
#   operations. 32 features are computed for each 5x5 patch.                     
                                                                                 
# The first two dimensions of the weight tensor correspond to the 5x5 patch      
#   size. The next two refer to the 1 input channel and 32 output channels. The  
#   bias vector has a component for each output channel.  
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# Input vector is reshaped into 4D tensor. The first dimension is inferred, the
#   next two refer to the 28x28 image size, and the last refers to the number of
#   color channels, 1.
x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# -------------------------Second Convolutional Layer---------------------------

# Does convolution again, reducing the image size to 7x7. 64 features are        
#   computed for each 5x5 patch. 

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# ----------------------------Densely Connected Layer---------------------------

# For our densely connected layer of 1024 neurons, we flatten our image input,   
#   multiply by the weight matrix, add the bias and apply ReLU. 

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# -----------------------------------Dropout------------------------------------

# Applying dropout to reduce overfitting as explained in JMLR 2014 paper by
#   Srivastava et al.
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# ---------------------------------Readout layer--------------------------------

# Simple linear regression layer. 

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# ------------------------------Train and Evaluate------------------------------

# Creates the Saver object whose restore() function we will use later 
saver = tf.train.Saver()

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

# Restore the trained model
saver.restore(sess, './savefiles/my-model')

print("Test accuracy %g%%" % (accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})*100))
