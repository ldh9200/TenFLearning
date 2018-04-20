# =========CIFAR10_INPUT_DUMP.PY =========
# TOTORIAL의 CIFAR10_INPUT.PY에서 TRAIN데이타 100개이미지를 PNG파일도 DUMP하는 예제만 추가한것.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from six.moves import xrange
import tensorflow as tf
import matplotlib.pyplot as plt  #이미지DUMP용

IMAGE_SIZE = 24

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES                      = 10        # 10개의 이미지를 분류한다.
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000     # 한번의 훈련epoch에 읽어들일 입력사진 갯수
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL  = 10000     # 한번의 평가epoch에서 읽어들일 입력사진 갯수


def read_cifar10(filename_queue):

  class CIFAR10Record(object):
    pass
  result = CIFAR10Record()

  # input format.
  label_bytes   = 1 
  result.height = 32
  result.width  = 32
  result.depth  = 3
  image_bytes   = result.height * result.width * result.depth
  record_bytes  = label_bytes + image_bytes

  reader            = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  result.key, value = reader.read(filename_queue)
  record_bytes      = tf.decode_raw(value, tf.uint8)
  result.label      = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

  depth_major       = tf.reshape(tf.strided_slice(record_bytes, [label_bytes],[label_bytes + image_bytes])
                     ,[result.depth, result.height, result.width])
  result.uint8image = tf.transpose(depth_major, [1, 2, 0]) # Convert from [depth, height, width] to [height, width, depth].

  # 아래 코드를 싫행하면 총 100개의 이미지가 10 X 10 형태로 배열된 1개의 이미지가 # 만들어지며, label, key, value 값을 확인할 수 있다. 
  # 이 코드를 사용하려면 matplotlib.pyplot을 import해야 한다. 
  fig, ax = plt.subplots(10, 10, figsize=(10, 10)) 
  with tf.Session() as sess: 
    coord = tf.train.Coordinator() 
    threads = tf.train.start_queue_runners(coord=coord, sess=sess) 
    for i in range(10): 
      for j in range(10): 
        xxlabel = str(sess.run(result.label))
        print(sess.run(result.label), sess.run(result.key), sess.run(value)) 
        img = sess.run(result.uint8image)
        ax[i][j].set_axis_off() 
        ax[i][j].imshow(img)
        ax[i][j].annotate(xxlabel + str(i) + "," + str(j), xy=(0, 0), xytext=(0, -1), arrowprops=None, fontsize=6)
        
    dir = os.path.abspath("cifar10_image") 
    plt.savefig(dir + "/" + "image") 
    print(dir) 
    coord.request_stop() 
    coord.join(threads) 
  ###########################################

  return result


def _generate_image_and_label_batch(image, label, min_queue_examples, batch_size, shuffle):
  num_preprocess_threads = 16
  if shuffle:
    images, label_batch = tf.train.shuffle_batch([image, label],
                                   batch_size=batch_size,
                                   num_threads=num_preprocess_threads,
                                   capacity=min_queue_examples + 3 * batch_size,
                                   min_after_dequeue=min_queue_examples
                                   )
  else:
    images, label_batch = tf.train.batch        ([image, label],
                                    batch_size=batch_size,
                                    num_threads=num_preprocess_threads,
                                    capacity=min_queue_examples + 3 * batch_size
                                    )
  tf.summary.image('images', images)
  return images, tf.reshape(label_batch, [batch_size])


def distorted_inputs(data_dir, batch_size):

  filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in xrange(1, 6)]
  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  filename_queue = tf.train.string_input_producer(filenames)

  with tf.name_scope('data_augmentation'):
    read_input     = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)
    height         = IMAGE_SIZE  #-->24
    width          = IMAGE_SIZE  #-->24

    #image비틀기
    distorted_image = tf.random_crop(reshaped_image, [height, width, 3])
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    distorted_image = tf.image.random_brightness(distorted_image,  max_delta=63)
    distorted_image = tf.image.random_contrast  (distorted_image,  lower=0.2, upper=1.8)
    float_image     = tf.image.per_image_standardization(distorted_image)

    # Set the shapes of tensors.
    float_image.set_shape([height, width, 3])
    read_input.label.set_shape([1])

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)
    print ('Filling queue with %d CIFAR images before starting to train. ' 'This will take a few minutes.' % min_queue_examples)
  
  return _generate_image_and_label_batch(float_image, read_input.label, min_queue_examples, batch_size, shuffle=True)


def inputs(eval_data, data_dir, batch_size):

  if not eval_data:
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in xrange(1, 6)]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  else:
    filenames = [os.path.join(data_dir, 'test_batch.bin')]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(filenames)
    read_input     = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)
    height         = IMAGE_SIZE
    width          = IMAGE_SIZE
    resized_image  = tf.image.resize_image_with_crop_or_pad(reshaped_image,height, width)
    float_image    = tf.image.per_image_standardization(resized_image)

    float_image.set_shape([height, width, 3])
    read_input.label.set_shape([1])

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch * min_fraction_of_examples_in_queue)

  return _generate_image_and_label_batch(float_image, read_input.label, min_queue_examples, batch_size, shuffle=False)
