import tensorflow as tf
import numpy as np
import Layer_class
import matplotlib.pyplot as plt
from PIL import Image

with tf.Graph().as_default():
    BATCH_SIZE = 1
    N_CLASSES = 3
    val_src = 'images/validate/'
    filename = 'saul.jpg'
    filename2 = 'son.jpg'

    image_file = Image.open(val_src + filename2)
    image_file = image_file.resize([208, 208])
    plt.imshow(image_file)
    img_file1 = np.array(image_file)

    images = np.reshape(img_file1,newshape=[BATCH_SIZE,208,208,3])

    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 208, 208, 3])
    keep_prob = tf.placeholder(tf.float32)

    logits = Layer_class._build_net(x, batch_size=BATCH_SIZE, n_classes=N_CLASSES, keep_prob=1)
    logits = tf.nn.softmax(logits)
    logs_train_dir = 'train/logs/'

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
        prediction = sess.run(logits, feed_dict={x: images, keep_prob : 1})
        max_index = np.argmax(prediction)

        print("prediction :", max_index)
