
from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib as mpl
rng = np.random

#trX = np.linspace(-1, 1, 101)
#print(trX)


# create a y value which is approximately linear but with some random noise

#trY = 2 * trX + 4+np.random.randn(*trX.shape) * 0.033

dat = np.genfromtxt('housing.txt')

#x1=dat[:,4]
#print(x1)
#x2=x1[0:250]
#x3=dat[:,14]
#x4=x1[0:250]
trX=dat[0:250,12]
trY=dat[0:250,13]
test_X=dat[251:506,12]
test_Y=dat[251:506,13]




# create symbolic variables

X = tf.placeholder("float")

Y = tf.placeholder("float")


# create a shared variable for the weight matrix

w = tf.Variable(rng.randn(), name="weights")

b = tf.Variable(rng.randn(), name="bias")

# prediction function
y_model = tf.add(tf.mul(X, w), b)


# Mean squared error

cost = tf.reduce_sum(tf.pow(y_model-Y, 2))/(2*100)

# construct an optimizer to minimize cost and fit line to my data

train_op = tf.train.GradientDescentOptimizer(0.5).minimize(cost)


# Launch the graph in a session
sess = tf.Session()

# Initializing the variables

init = tf.initialize_all_variables()


# you need to initialize variables
sess.run(init)


for i in range(100):
    for (x, y) in zip(trX, trY):
        sess.run(train_op, feed_dict={X: x, Y: y})

print("Optimization Finished!")
training_cost = sess.run(cost, feed_dict={X: trX, Y: trY})

print("Training cost=", training_cost, "W=", sess.run(w), "b=", sess.run(b), '\n')

# Testing or Inference
#test_X = np.asarray([rng.randn(),rng.randn()])
#test_Y = 2*test_X + 4

print("Testing... (Mean square loss Comparison)")

testing_cost = sess.run(
    tf.reduce_sum(tf.pow(y_model - Y, 2)) / (2 * test_X.shape[0]),
    feed_dict={X: test_X, Y: test_Y})  # same function as cost above
print("Testing cost=", testing_cost)
print("Absolute mean square loss difference:", abs(
    training_cost - testing_cost))
