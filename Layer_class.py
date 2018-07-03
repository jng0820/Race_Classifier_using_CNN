import tensorflow as tf

learning_rate = 0.001

def _build_net(images,batch_size,n_classes,keep_prob):
     with tf.variable_scope('conv') as conv:
        W1 = tf.Variable(tf.random_normal([3, 3, 3, 16], stddev=0.01))
        L1 = tf.nn.conv2d(images, W1, strides=[1, 1, 1, 1], padding='SAME')
        L1 = tf.nn.relu(L1)
        L1 = tf.nn.max_pool(L1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

        W2 = tf.Variable(tf.random_normal([3, 3, 16, 16], stddev=0.01))
        L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
        L2 = tf.nn.relu(L2)
        L2 = tf.nn.max_pool(L2, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
        L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
        L2 = tf.reshape(L2, shape=[batch_size, -1])

        dim = L2.get_shape()[1].value
        W3 = tf.get_variable("W4", shape=[dim, 128], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.random_normal([128]))
        L3 = tf.nn.relu(tf.matmul(L2, W3) + b)
        L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

        W5 = tf.get_variable("W5", shape=[128, n_classes], initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.Variable(tf.random_normal([n_classes]))
        hypothesis = tf.matmul(L3, W5) + b2

        return hypothesis


def cost(logits,label):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=label)
    cost = tf.reduce_mean(cross_entropy)
    return cost

def optimizer(cost,learning_rates):
    opt = tf.train.AdamOptimizer(learning_rate=learning_rates).minimize(cost)
    return opt

def accuaracy(logits,label):
    correct = tf.nn.in_top_k(logits,label,1)
    correct = tf.cast(correct,tf.float16)
    accuara = tf.reduce_mean(correct)
    return accuara

