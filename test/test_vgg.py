import unittest
import sys
sys.path.append('..')
from networks import vgg 
import tensorflow as tf

class testvgg(unittest.testcase):
    def test_load_npy(sefl):
        model = vgg.vggnet()

        sess = tf.interactivesession()
        model.load('data/vgg_imagenet.npy', sess, ignore_missing=true )

if __name__ == '__main__':
    unittest.main()
