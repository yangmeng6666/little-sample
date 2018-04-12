# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 08:08:55 2018

@author: ym
"""

import tensorflow as tf
from read_data import Dataset
import numpy
import time
from matplotlib import pyplot as plt
way=496
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
    strides=[1, 2, 2, 1], padding='SAME')
def conv2d(x, W,b,pa):
    if pa==1:
       return tf.nn.sigmoid(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')+b)
    else:
       return tf.nn.relu(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')+b)
def variable_summaries(var,name):
    with tf.name_scope(name+'_summaries'):
      # 计算参数的均值，并使用tf.summary.scaler记录
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)

      # 计算参数的标准差
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      # 使用tf.summary.scaler记录记录下标准差，最大值，最小值
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      # 用直方图记录参数的分布
      tf.summary.histogram('histogram', var)   
if __name__=='__main__':
    data=Dataset()
    print('read data done!!')
    y_= tf.placeholder("float", [None,way])
    w_= tf.Variable(tf.random_normal(shape=[5,5,1,128]),name='W_1')
    b_= tf.Variable(tf.random_normal(shape=[128]),name='b_1')
    w__= tf.Variable(tf.random_normal(shape=[3,3,128,64]),name='W_2')
    b__= tf.Variable(tf.random_normal(shape=[64]),name='b_2')
    w1__= tf.Variable(tf.random_normal(shape=[5,5,64,16]),name='W_3')
    b1__= tf.Variable(tf.random_normal(shape=[16]),name='b_3')
    wc1= tf.Variable(tf.random_normal(shape=[7*7*16,256]),name='W_4')
    bc1= tf.Variable(tf.random_normal(shape=[256]),name='b_4')
    wc2= tf.Variable(tf.random_normal(shape=[256,32]),name='W_4')
    bc2= tf.Variable(tf.random_normal(shape=[32]),name='b_4')
    x_std=tf.placeholder(tf.float32, [None, 784])
    x = tf.placeholder(tf.float32, [None, 784])
    c1=conv2d(tf.reshape(x, [-1,28,28,1]),w_,b_,1)
    c1_std=conv2d(tf.reshape(x_std, [-1,28,28,1]),w_,b_,1)
    c2=conv2d(max_pool_2x2(conv2d(max_pool_2x2(c1),w__,b__,1)),w1__,b1__,pa=1)
    c2_std=conv2d(max_pool_2x2(conv2d(max_pool_2x2(c1_std),w__,b__,1)),w1__,b1__,pa=1)
    c2_flat=tf.reshape(c2, [-1, 7*7*16])  
    c2_std_flat=tf.reshape(c2_std, [-1, 7*7*16])
    fc1=tf.nn.relu(tf.matmul(c2_flat,wc1) + bc1)
    fc2=tf.nn.relu(tf.matmul(fc1,wc2) + bc2)
    fc1_std=tf.nn.relu(tf.matmul(c2_std_flat,wc1) + bc1)
    fc2_std=tf.nn.relu(tf.matmul(fc1_std,wc2) + bc2)
    x_all=tf.square(fc2_std-tf.tile(fc2,(way*5,1)))
    h2=tf.reduce_sum(tf.reshape(tf.reduce_sum(x_all,1),[-1,5]),1)
    cross_entropy=tf.reduce_sum(y_*h2+(1-y_)*tf.maximum(0.0,250.0-h2))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
    #correct_prediction = tf.equal(tf.argmax(y_), tf.argmax(y))
    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init)
    saver = tf.train.Saver()
    saver.restore(sess,'model/model1_cov.ckpt')
    #merged_summary_op = tf.summary.merge_all()
    #summary_writer = tf.summary.FileWriter('/tmp/little', sess.graph)
    for i in range(10000):
        start=time.time()
        if i<20000:
           train,test,remainder= data.random_sample(way,5,11,0)
        else:
           train,test,remainder= data.random_sample(way,1,18,1)
        ce=0.0
        for item in test:
            put=numpy.zeros(way)
            for i0 in range(way):
                if item[1].argmax()==train[1][i0*5].argmax():
                    put[i0]=1
            feed={x_std:train[0].reshape(-1,784), y_:put.reshape((1,way)),
                 x:item[0].reshape(-1,784)}
            #
            #print(cross_entropy.eval(feed_dict=feed))
            #print((y).eval(feed_dict=feed),put)
            sess.run(train_step, feed_dict=feed)
        j=0.0
        #train,__,remainder= data.random_sample(way,1,15,1)
        if i%2==0:
            for item in remainder:
                put=numpy.zeros(way)
                for i0 in range(way):
                    if item[1].argmax()==train[1][i0*5].argmax():
                        put[i0]=1
                feed={x_std:train[0].reshape(-1,784), y_:put.reshape((1,way)),
                     x:item[0].reshape(-1,784)}
                #summary_str = sess.run(merged_summary_op,feed_dict=feed)
                #summary_writer.add_summary(summary_str, i*len(remainder)+j)
                ce+=sess.run(cross_entropy,feed_dict=feed)
                #t=tf.reduce_max(h2).eval(feed_dict=feed)
                print(numpy.array(h2.eval(feed_dict=feed)),put)
                if numpy.array(sess.run(h2,feed_dict=feed)).argmin()==put.argmax():
                   j+=1
                '''else:
                   plt.imshow(item[0]),plt.show()
                   plt.imshow(train[0][numpy.array(sess.run(y,feed_dict=feed))[0].argmax()])
                   plt.show()
                   print(j)'''
            print(i,ce,j/len(remainder),time.time()-start,'s')
    saver.save(sess,'model/model1_cov.ckpt')
    sess.close()
    