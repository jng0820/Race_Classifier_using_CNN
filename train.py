import tensorflow as tf
import os
import numpy as np
import Layer_class
from datetime import datetime

batch_size = 64
n_classes = 3

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
np.random.shuffle(temp)


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

image_batch,label_batch = tf.train.batch([img_files,label_images], batch_size= batch_size, num_threads=64, capacity=2000)

label_batch = tf.reshape(label_batch,[batch_size])
image_batch = tf.cast(image_batch,tf.float32)



sess = tf.Session()
sess.run(tf.global_variables_initializer())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)


x = tf.placeholder(tf.float32,shape=[batch_size,208,208,3])
y = tf.placeholder(tf.int32,shape=[batch_size])
keep_prob = tf.placeholder(tf.float32)

logits = Layer_class._build_net(x, batch_size, n_classes,keep_prob)

cost = Layer_class.cost(logits,y)
optimizer = Layer_class.optimizer(cost,0.001)
accuracy = Layer_class.accuaracy(logits,y)

with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess,coord)

    summary_op = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(train_log_path,sess.graph)
    val_writer = tf.summary.FileWriter(train_log_path,sess.graph)

    for step in np.arange(1500):
        if coord.should_stop():
            break

        tra_images, tra_labels = sess.run([image_batch, label_batch])
        _, train_loss,train_accuracy = sess.run([optimizer, cost, accuracy],feed_dict={x:tra_images,y:tra_labels,keep_prob:0.7})

        if step % 50 == 0:
            print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, train_loss, train_accuracy * 100.0))
            #summary_str = sess.run(summary_op)
            #train_writer.add_summary(summary_str, step)

    checkpoint_path = os.path.join(train_log_path, 'model.ckpt')
    saver.save(sess, checkpoint_path)
    print('Model training & saving finished.')
    coord.request_stop()
    coord.join(threads)
    print("Image Name : ",tra_images)
    print("Prediction : ", sess.run(tf.argmax(logits,1),feed_dict={x:tra_images,keep_prob:1}))
    print("Real Value : ",tra_labels)
