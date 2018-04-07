# -*- coding: euc-kr -*- 
'''
Created on 2018. 4. 1.
 - MINIST  : �ձ۾� ���� �ν� 
 - 28 X 28 ũ���� ȸ���� �̹��� ( 1ä��)    RGB 3ä��    ,  ���̺�� ����
 - �Է���    :  28 X 28  ��źȭ = 1 * 784 ��ķ� ����  H(X) = wX + b 
 - ������� :  10�� ( 0 ~ 9 ) ,  one hot encoding
                    softmax  : �з�����  0 ~ 1 ������ ������ ǥ�� , ��°��� �� ���� 1 , ��� ���� Ȯ���� ����
                                        Ȱ���� ǥ�ð� �ƴҰ��  ���� �ϱ⵵ ��
@author: ldh92
'''
import tensorflow as tf
import argparse 
import numpy as np

from minist import input_data #����Ÿ �ε� 
from PIL  import Image
#from test.CoreMuLiValueLinear import optimizer

parser = argparse.ArgumentParser()
 
parser.add_argument('--down_load_dir' , type=str, default='c://tmp/Download/minist' , help='DownloadPath for  IMAGE FILES of MINIST') 
parser.add_argument('--model_dir'        , type=str, default='c://tmp/LRLS' , help='Base directory for the model.')
parser.add_argument('--batch_size'       , type=np.float32, default=100 , help='batch_size')
parser.add_argument('--epochs'           , type=np.float32, default=25 , help='epochs')
FLAGS, unparsed = parser.parse_known_args()

# ����ȭ ó���� Feature , lable 
# Convert from [0, 255] -> [0.0, 1.0].
mnist = input_data.read_data_sets(FLAGS.down_load_dir, one_hot=True ) #X :  (1, 784)  , Y :(1, 10)
#mnist = input_data.read_data_sets(FLAGS.down_load_dir, one_hot=False )  #X :  (1, 784)  , Y :(1)
X = tf.placeholder(tf.float32, [None, 784])
W1 = tf.Variable(tf.random_normal([784, 256]))
W3 = tf.Variable(tf.random_normal([256, 10]))

B1 = tf.Variable(tf.random_normal([256]))
B3 = tf.Variable(tf.random_normal([10]))
Y = tf.placeholder(tf.int64 ,[None,10] )

# Construct model
L1 = tf.nn.relu(tf.add(tf.matmul(X,W1),B1))
hypothesis = tf.add(tf.matmul(L1, W3), B3) # No need to use softmax here
#cost
cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2( labels =Y , logits=hypothesis ))
#
#train_step = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cross_entropy) 

init = tf.initialize_all_variables()
with tf.Session() as sess :
    sess.run(init)
    total_batch = int(mnist.train.num_examples/FLAGS.batch_size)
    print("��ü�Ǽ� :" , mnist.train.num_examples)
    print("��ġ����Ƚ�� ��ü�� / ��ġ������ :" , total_batch)
    for epoch in range( FLAGS.epochs) :
        avg_cost = 0.
        for step in range(total_batch) :
            batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
            _,_,vh=  sess.run([train_step , cross_entropy ,hypothesis]  , feed_dict={X :batch_xs , Y:batch_ys })
            avg_cost += sess.run(cross_entropy, feed_dict={X :batch_xs , Y:batch_ys })/total_batch
        if epoch % 1 == 0:
            print ("Epoch:", '%04d' %(epoch+1), "cost=", "{:.9f}".format(avg_cost))
    print ("Optimization Finished!")
    # Test model
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print ("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))
    batch_xs, batch_ys = mnist.train.next_batch(2)
    _,_,vh=  sess.run([train_step , cross_entropy ,hypothesis]  , feed_dict={X :batch_xs , Y:batch_ys })
    print ("������ one hot :", batch_ys)
    print ("������ :", sess.run(tf.argmax(vh, 1)))
