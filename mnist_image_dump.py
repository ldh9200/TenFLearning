# -*- coding: euc-kr -*-
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
# Dump mnist image & Label & Predict 

import argparse 
import sys
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
FLAGS = None

def main(_): 
  mnist = input_data.read_data_sets(FLAGS.data_dir) 
  x = tf.placeholder(tf.float32, [None, 784]) 
  W = tf.Variable(tf.zeros([784, 10])) 
  b = tf.Variable(tf.zeros([10])) 
  y = tf.matmul(x, W) + b
  y_ = tf.placeholder(tf.int64, [None])

  cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y) 
  train_step    = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(y, 1), y_) 
  accuracy           = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  
  x_image4print   = tf.reshape(x, [-1, 28, 28])   #-->input-image,인쇄용 2D변환
  y_label4print   = y_                            #-->input-label
  y_Predict4print = tf.argmax(y, 1)               #-->predict_y

  with tf.Session() as sess:  
    sess.run(tf.global_variables_initializer()) 
    for i in range(30):      
       batch_xs, batch_ys = mnist.train.next_batch(1000)  
       train_step4print,x4print,y4print = sess.run([train_step,x_image4print,y_label4print], feed_dict={x: batch_xs, y_: batch_ys})
       if i % 10 == 0:
           train_accuracy = sess.run(accuracy, feed_dict={ x: mnist.test.images, y_: mnist.test.labels })
           print("epoch1, i=%d, accuracy=" % i, train_accuracy)
   
    accuracy4print,x4print,y4print,yPredict4print = sess.run([accuracy,x_image4print,y_label4print,y_Predict4print],feed_dict={ x: mnist.test.images, y_: mnist.test.labels})   
    print("final accuracy=", accuracy4print)  
  
  # 테스트 코드 시작 ################################################ (( 출처: http://mazdah.tistory.com/813 [꿈을 위한 단상] ))
  fig, ax = plt.subplots(5, 5, figsize=(10, 10)) 
  if True:
    idx = -1 
    for i in range(5): 
      for j in range(5):
        idx = idx + 1   
        ax[i][j].set_axis_off()
        ax[i][j].imshow(x4print[idx])
        ax[i][j].annotate("label"+ "[" + str(y4print[idx]) +"]" + " Predict->" + str(yPredict4print[idx]) +"  Image.No#" + str(idx), xy=(0, 0), xytext=(0, -1), arrowprops=None, fontsize=6)
    dir = os.path.abspath("image") 
    plt.savefig(dir + "/" + "test_image") 
  ###########################################################

if __name__ == '__main__': 
  parser = argparse.ArgumentParser() 
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',  help='Directory for storing input data') 
  FLAGS, unparsed = parser.parse_known_args() 
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed) 
