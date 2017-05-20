import unittest
import sys
sys.path.append('..')
from vgg_transfer import VGG_Style
import tensorflow as tf

class test_transfer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = VGG_Style()
        cls.sess = tf.InteractiveSession()

    def test_content_loss(self):
        self.model.build_content_loss()

    def test_style_loss(self):
        self.model.build_style_loss()
        

if __name__ == '__main__':
    unittest.main()
