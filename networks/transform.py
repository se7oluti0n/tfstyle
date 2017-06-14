from . import network
from .network import layer
import tensorflow as tf

class TransformNet(network.Network):
    def __init__(self, data=None, trainable=True, reuse=False):
        self.inputs = []

        self.data = data
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict({'data': self.data})
        self.trainable = trainable
        self.reuse = reuse
        self.setup()

    def setup(self):
        (self.feed('data')
            .conv(9, 9, 32, 1, 1, name='conv1_1', trainable=self.trainable, reuse=self.reuse, biased=False)
            .conv(3, 3, 64, 2, 2, name='conv1_2', trainable=self.trainable, reuse=self.reuse, biased=False)
            .conv(3, 3, 128, 2, 2, name='conv1_3', trainable=self.trainable, reuse=self.reuse, biased=False)
            .residual_block(3, 3, 128, 1, 1, name='resid1', biased=False, relu=False)
            .residual_block(3, 3, 128, 1, 1, name='resid2', biased=False, relu=False)
            .residual_block(3, 3, 128, 1, 1, name='resid3', biased=False, relu=False)
            .residual_block(3, 3, 128, 1, 1, name='resid4', biased=False, relu=False)
            .upconv(None, 64, 3, 2, name='upconv1', biased=False, relu=True)
            .upconv(None, 32, 3, 2, name='upconv2', biased=False, relu=True)
            .conv(9, 9, 3, 1, 1, biased=False, relu=False, name='conv4_1')
        )


    

