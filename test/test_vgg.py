import unittest
import sys
sys.path.append('..')
from networks import vgg 
import tensorflow as tf

class TestVGG(unittest.TestCase):
    def test_load_npy(sefl):
        model = vgg.VGGNet()

        sess = tf.InteractiveSession()
        model.load('data/VGG_imagenet.npy', sess, ignore_missing=True )

if __name__ == '__main__':
    unittest.main()
