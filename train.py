import tensorflow as tf
import os
import numpy as np
from datetime import datetime


asian_path = 'images/train/asian/'
white_path = 'images/train/white/'
black_path = 'images/train/black/'

train_log_path = 'train/logs/'

img_files = []
label_images = []

for file in os.listdir(asian_path):
    img_files.append(asian_path + file)
    label_images.append(0)

for file in os.listdir(white_path):
    img_files.append(white_path + file)
    label_images.append(1)

for file in os.listdir(black_path):
    img_files.append(black_path + file)
    label_images.append(2)

img_files = np.hstack((img_files))
label_images = np.hstack((label_images))


temp = np.array([img_files,label_images])
temp = temp.transpose()

img_files = list(temp[:,0])
label_images = list(temp[:,1])
label_images = [int(i) for i in label_images]

number_of_files = len(img_files)

img_files = tf.cast(img_files, tf.string)
label_images = tf.cast(label_images, tf.int32)
input_queue = tf.train.slice_input_producer([img_files, label_images])

label_images = input_queue[1]
image_content = tf.read_file(input_queue[0])
img_files = tf.image.decode_jpeg(image_content,channels=3)
img_files = tf.image.resize_image_with_crop_or_pad(img_files,208,208)

img_files = tf.image.per_image_standardization(img_files)

image_batch,label_batch = tf.train.batch([img_files,label_images], batch_size= 16, num_threads=32, capacity=2000)

label_batch = tf.reshape(label_batch,[16])
image_batch = tf.cast(image_batch,tf.float32)

keep_prob = tf.placeholder("float")

W1 = tf.Variable(tf.random_normal([3,3,3,16],stddev=0.01))
L1 = tf.nn.conv2d(image_batch,W1,strides=[1,1,1,1],padding = 'SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1,3,3,1],strides=[1,2,2,1], padding='SAME')
L1 = tf.nn.dropout(L1,keep_prob=keep_prob)

W2 = tf.Variable(tf.random_normal([3,3,16,16],stddev=0.01))
L2 = tf.nn.conv2d(L1,W2,strides=[1,1,1,1],padding = 'SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1,3,3,1],strides=[1,1,1,1], padding='SAME')
L2 = tf.nn.dropout(L2,keep_prob=keep_prob)
L2 = tf.reshape(L2,shape=[16,-1])

dim = L2.get_shape()[1].value
W3 = tf.get_variable("W4",shape=[dim,128],initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([128]))
L3 = tf.nn.relu(tf.matmul(L2,W3)+b)
L3 = tf.nn.dropout(L3,keep_prob=keep_prob)

W5 = tf.get_variable("W5",shape=[128,3],initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([3]))
hypothesis = tf.matmul(L3,W5)+b2

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits\
                        (logits=hypothesis, labels=label_batch, name='xentropy_per_example')
cost = tf.reduce_mean(cross_entropy, name='loss')
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

correct = tf.nn.in_top_k(hypothesis, label_batch, 1)
correct = tf.cast(correct, tf.float16)
accuracy = tf.reduce_mean(correct)

saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for step in np.arange(1000):
    if coord.should_stop():
        break
    _, tra_loss, tra_acc = sess.run([optimizer, cost, accuracy],feed_dict={keep_prob: 0.7})
    if step % 50 == 0:
        print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))

save_path = saver.save(sess, "./model_save/model_saved.ckpt")
