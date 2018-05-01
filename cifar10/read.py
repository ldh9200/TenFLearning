
# -*- coding: euc-kr -*- 
'''
Created on 2018. 4. 22.

@author: ldh92
'''
import tensorflow as tf


import os
import matplotlib.pyplot as plt  #이미지DUMP용 

 
class CifarImgShow:
    
    '''
    classdocs
    '''
     
 
    def __init__(self):
        '''
        Constructor
        '''
        self.fig ,self.ax=  plt.subplots(15, 6, figsize=(6,15))
        self.i =-1 
        self.j =-1
         
    def insertRow( self, img , label) :
        self.i+=1
        self.j =0
        print(self.i , self.j)
        self.ax[self.i][self.j].set_axis_off() 
        self.ax[self.i][self.j].imshow(img) 
        self.ax[self.i][self.j].annotate(str(label) + str( self.i) + "," + str(   self.j ), xy=(0, 0), xytext=(0, -1), arrowprops=None, fontsize=6)
         
         
         
    def insertCell( self, img , label) :
        self.j+=1
        self.ax[self.i][self.j].set_axis_off() 
        self.ax[self.i][self.j].imshow(img)
        self.ax[self.i][self.j].annotate(str(label) , xy=(0, 0), xytext=(0, -1), arrowprops=None, fontsize=6)
 
        
'''
   Cifar10.py 를 통해 다운받은 파일 이미지와 라벨확인 
'''
def readCifar10File():
    FLAGS = tf.app.flags.FLAGS

    #파일은 다운 받았다 치고
    tf.app.flags.DEFINE_integer('batch_size'    , 128, """Number of images to process in a batch.""")
    tf.app.flags.DEFINE_string('data_dir'       , '/tmp/cifar10_data',"""Path to the CIFAR-10 data directory.""")
    tf.app.flags.DEFINE_boolean('use_fp16'  , False,"""Train the model using fp16.""")

    print( "data_dir" , FLAGS.data_dir)
    print( "batch_size" , FLAGS.batch_size)
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    print( "data_dir" , data_dir)
    
    label_bytes = 1  # 2 for CIFAR-100
    height = 32
    width = 32
    depth = 3
    image_bytes = height * width * depth

    #읽어들일 목록구성 Queue Runner ,  Default shuffle true
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in range(1, 6)]    
   #filenames = ['C:\\Users\ldh92\\.keras\datasets\\cifar-10-batches-py\\data_batch_1']
    print(filenames )

    filename_queue = tf.train.string_input_producer(filenames)

    # 읽어야 하는 라인당 사이즈
    record_bytes = label_bytes + image_bytes
   
    #데이타 읽기  FixedLengthRecordReader
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    # filename_queue[] 데이타 읽기 
    key, value = reader.read(filename_queue)
    # value  tf.uint8 으로 변환
    record_bytes = tf.decode_raw(value , tf.uint8)
    #첫자리 라벨분리 
    label = tf.cast( tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)
  
  
    depth_major = tf.reshape(

      tf.strided_slice(record_bytes, [label_bytes],

                       [label_bytes + image_bytes]),

      [depth, height, width]
      
      #[height, width, depth]
      )

  # Convert from [depth, height, width] to [height, width, depth].

    uint8image = tf.transpose(depth_major, [1, 2, 0])
  
    fig ,ax=  plt.subplots(10, 10, figsize=(10, 10)) 
    
    with tf.Session() as sess :
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess =sess , coord =coord)
        for i in range(10):  
             for j in range(10): 
                 print(sess.run(depth_major)) 
                 vl,vk,vv ,img= sess.run([label , key ,value ,uint8image])
                 ax[i][j].set_axis_off() 
                 ax[i][j].imshow(img) 
                 ax[i][j].annotate(str(vl) + str(i) + "," + str(j), xy=(0, 0), xytext=(0, -1), arrowprops=None, fontsize=6)
        coord.request_stop()
        coord.join(threads)
        plt.show()
        


'''
   keras datasets.cifar10 를 통해 다운받은 파일 이미지와 라벨확인 
'''
def readCifar10Keras():

   
    import numpy as np
    from tensorflow.python.keras._impl.keras.datasets.cifar10 import load_data
    from tensorflow.python.keras._impl.keras.utils.data_utils import get_file

    dirname = 'cifar-10-batches-py'
    origin = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    path = get_file(dirname, origin=origin, untar=True)
    print(path)
    (x_train, y_train), (x_test, y_test) = load_data()
    
    fig ,ax=  plt.subplots(10, 10, figsize=(10, 10))     
    index =0
    for i in  range(10) :
        for j in range(10):
            ax[i][j].imshow(x_train[index]) 
            ax[i][j].annotate(str(y_train[index]) + str(i) + "," + str(j), xy=(0, 0), xytext=(0, -1), arrowprops=None, fontsize=8)
            index+=1
         
    plt.show()



def shuffle(float_image, label):
        images, label_batch = tf.train.shuffle_batch(
        [float_image, label],
        batch_size=128,
        num_threads=2,
        capacity=2 + 3 * 128,
        min_after_dequeue=2)
        
        return images, tf.reshape(label_batch, [128])
    
'''
   Cifar10.py 를 통해 다운받은 파일 이미지와 라벨확인 
'''
def readCifar10File2():
    FLAGS = tf.app.flags.FLAGS

    #파일은 다운 받았다 치고
    tf.app.flags.DEFINE_integer('batch_size'    , 128, """Number of images to process in a batch.""")
    tf.app.flags.DEFINE_string('data_dir'       , '/tmp/cifar10_data',"""Path to the CIFAR-10 data directory.""")
    tf.app.flags.DEFINE_boolean('use_fp16'  , False,"""Train the model using fp16.""")

    print( "data_dir" , FLAGS.data_dir)
    print( "batch_size" , FLAGS.batch_size)
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    print( "data_dir" , data_dir)
    
    label_bytes = 1  # 2 for CIFAR-100
    height = 32
    width = 32
    depth = 3
    image_bytes = height * width * depth

    #읽어들일 목록구성 Queue Runner ,  Default shuffle true
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in range(1, 6)]    
   #filenames = ['C:\\Users\ldh92\\.keras\datasets\\cifar-10-batches-py\\data_batch_1']
    print(filenames )

    filename_queue = tf.train.string_input_producer(filenames)

    # 읽어야 하는 라인당 사이즈
    record_bytes = label_bytes + image_bytes
   
    #데이타 읽기  FixedLengthRecordReader
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    # filename_queue[] 데이타 읽기 
    key, value = reader.read(filename_queue)
    # value  tf.uint8 으로 변환
    record_bytes = tf.decode_raw(value , tf.uint8)
    #첫자리 라벨분리 및 shape 지정 
    label = tf.cast( tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)
    label.set_shape([1])
    
    depth_major = tf.reshape(

      tf.strided_slice(record_bytes, [label_bytes],

                       [label_bytes + image_bytes]),

      [depth, height, width]
      
      #[height, width, depth]
      )
    print(depth_major)
    #Tensor("Reshape:0", shape=(3, 32, 32), dtype=uint8)
    uint8image = tf.transpose(depth_major, [1, 2, 0])
    print(uint8image)    
    #Tensor("transpose:0", shape=(32, 32, 3), dtype=uint8)
    #uint8 이미지를 float32로 변환
    with tf.name_scope('data_augmentation'):
        reshaped_image = tf.cast(uint8image, tf.float32)
    
        # 24 * 24 * 3  
        distorted_image = tf.random_crop(reshaped_image, [24, 24, 3] , name="distorted_image" )
        #좌우뒤집기
        distorted_image2 = tf.image.random_flip_left_right(distorted_image)
        #밝기 조절
        distorted_image3 = tf.image.random_brightness(distorted_image2, max_delta=63)
        #콘트라스( 대비 )
        distorted_image4 = tf.image.random_contrast( distorted_image3, lower=0.2, upper=1.8)
      
        float_image = tf.image.per_image_standardization(distorted_image4)
    
    float_image.set_shape([24, 24, 3])
    #print("label",label)

    #print(float_image)        

    
    #images, label_batch = shuffle(reshaped_image, label)
    
    images, label_batch = tf.train.shuffle_batch(
      [float_image, label],
      batch_size=128,
      num_threads=2,
      capacity=3 ,   #최대수용 ???
      min_after_dequeue=2)


    e = CifarImgShow()
    #fig , ax=  plt.subplots(10, 10, figsize=(10, 10))
    with tf.Session() as sess :
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess =sess , coord =coord)
        for i in range(15):  
        
            vl,vk,vv ,img,c_img,f_img,b_img, cn_img ,float_img  = sess.run([label , key ,value ,uint8image,  tf.cast(distorted_image, tf.uint8)  , tf.cast(distorted_image2, tf.uint8) , tf.cast(distorted_image3, tf.uint8), tf.cast(distorted_image4, tf.uint8),tf.cast(float_image, tf.uint8)])
            e.insertRow(img, vl)
            e.insertCell(c_img, 'crop')
            e.insertCell(f_img, 'flip')
            e.insertCell(b_img,  'bri')
            e.insertCell(cn_img, "con")
            e.insertCell(float_img, "float")
        coord.request_stop()
        coord.join(threads)
        plt.show()
        
        
        
        
        
def cast():
     
     a = tf.Variable(1)

     b = tf.cast(a , tf.float32)
     with tf.Session() as sess :
        sess.run( tf.global_variables_initializer())
        print( sess.run( a))
        print( sess.run( b))

         
#cast()         
readCifar10File2()
 