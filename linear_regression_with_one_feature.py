import numpy as np
import tensorflow as tf

#Model parameters

theta0 = tf.Variable([-.3], dtype=tf.float32)
theta1 = tf.Variable([.3], dtype = tf.float32)

#input
x = tf.placeholder(tf.float32)
#Output
y = tf.placeholder(tf.float32)
#Model/Hypothesis function
linear_model = theta0 + theta1*x
#cost
cost = tf.reduce_sum(tf.square(linear_model - y)) #least square sum method
#optimizer
alpha = 0.01
optimizer = tf.train.GradientDescentOptimizer(alpha)
train = optimizer.minimize(cost)
#training data
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]
#training loop
init = tf.global_variables_initializer()
sees = tf.Session()
sees.run(init)
for i in range(1000):
        sees.run(train,{x:x_train,y:y_train})

#evaluate training accuracy
curr_theta0,curr_theta1,curr_cost = sees.run([theta0,theta1,cost],{x:x_train,y:y_train})
print("theta0 : %s; theta1 : %s; cost:%s"%(curr_theta0,curr_theta1,curr_cost))
