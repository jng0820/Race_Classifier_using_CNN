import tensorflow as tf
import numpy as np
import Layer_class

BATCH_SIZE = 1
N_CLASSES = 2
val_src = "images/validate/"
filename = "son.jpg"
image_array = []
image_array.append(val_src+filename)

image_array = tf.cast(image_array,tf.float32)
#image_array = tf.image.per_image_standardization(image_array)
image_array = tf.reshape(image_array,[BATCH_SIZE,208,208,3])

x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 208, 208, 3])

logs_train_dir = '/train/logs/'

sess = tf.Session()

saver = tf.train.Saver()
print("Reading checkpoints...")
ckpt = tf.train.get_checkpoint_state(logs_train_dir)
if ckpt and ckpt.model_checkpoint_path:
    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    saver.restore(sess, ckpt.model_checkpoint_path)
    print('Loading success, global_step is %s' % global_step)
else:
    print('No checkpoint file found')

logits = Layer_class._build_net(x, batch_size=BATCH_SIZE, n_classes=N_CLASSES, keep_prob=1)
prediction = sess.run(logits, feed_dict={x: image_array})
max_index = np.argmax(prediction)
print("prediction :", max_index)
