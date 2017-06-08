import network
import tensorflow as tf

class VGGNet(network.Network):
    def __init__(self, data=None, trainable=True, reuse=False):
        self.inputs = []

        if data:
            self.data = data
        else:
            self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='data')

        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict({'data': self.data})
        self.trainable = trainable
        self.reuse = reuse
        self.setup()

    def setup(self):

        (self.feed('data')
             .conv(3, 3, 64, 1, 1, name='conv1_1', trainable=self.trainable, reuse=self.reuse)
             .conv(3, 3, 64, 1, 1, name='conv1_2', trainable=self.trainable, reuse=self.reuse)
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool1')
             .conv(3, 3, 128, 1, 1, name='conv2_1', trainable=self.trainable, reuse=self.reuse)
             .conv(3, 3, 128, 1, 1, name='conv2_2', trainable=self.trainable, reuse=self.reuse)
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool2')
             .conv(3, 3, 256, 1, 1, name='conv3_1', trainable=self.trainable, reuse=self.reuse)
             .conv(3, 3, 256, 1, 1, name='conv3_2', trainable=self.trainable, reuse=self.reuse)
             .conv(3, 3, 256, 1, 1, name='conv3_3', trainable=self.trainable, reuse=self.reuse)
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool3')
             .conv(3, 3, 512, 1, 1, name='conv4_1', trainable=self.trainable, reuse=self.reuse)
             .conv(3, 3, 512, 1, 1, name='conv4_2', trainable=self.trainable, reuse=self.reuse)
             .conv(3, 3, 512, 1, 1, name='conv4_3', trainable=self.trainable, reuse=self.reuse)
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool4')
             .conv(3, 3, 512, 1, 1, name='conv5_1', trainable=self.trainable, reuse=self.reuse)
             .conv(3, 3, 512, 1, 1, name='conv5_2', trainable=self.trainable, reuse=self.reuse)
             .conv(3, 3, 512, 1, 1, name='conv5_3', trainable=self.trainable, reuse=self.reuse))

