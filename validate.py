import tensorflow as tf
import numpy as np
import train
import Layer_class

BATCH_SIZE = 1
N_CLASSES = 2

val_src = "images/validate/"
filename = input()
image_array = []
image_array.append(val_src+filename)


image = tf.cast(image_array, tf.float32)
image = tf.image.per_image_standardization(image)
image = tf.reshape(image, [1, 208, 208, 3])

logit = model1.inference(image, BATCH_SIZE, N_CLASSES)

model1 = Layer_class.Model()

logit = tf.nn.softmax(logit)

x = tf.placeholder(tf.float32, shape=[208, 208, 3])

logs_train_dir = '/home/kevin/tensorflow/cats_vs_dogs/logs/train/'

saver = tf.train.Saver()

    with tf.Session() as sess:

        print("Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(logs_train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Loading success, global_step is %s' % global_step)
        else:
            print('No checkpoint file found')

        prediction = sess.run(logit, feed_dict={x: image_array})
        max_index = np.argmax(prediction)
        if max_index==0:
            print('This is a cat with possibility %.6f' %prediction[:, 0])
        else:
            print('This is a dog with possibility %.6f' %prediction[:, 1])