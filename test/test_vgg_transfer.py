import unittest
import sys
sys.path.append('..')
from vgg_transfer import VGG_Style
import tensorflow as tf
import numpy as np

class test_transfer(unittest.TestCase):
    def setUp(self):
        self.model = VGG_Style(0.025, 5.0, 1.0, 512, 512)
        config = tf.ConfigProto(
            device_count = {'GPU': 0}
        )
        self.sess = tf.InteractiveSession(config=config)

#    def test_content_loss(self):
#        self.model.build_content_loss()

    # def test_style_loss(self):
    #     self.model.build_style_loss()
    # def test_tv_loss(self):
    #     self.model.build_total_variation_loss()

    # def test_build_loss(self):
        self.model.build_loss_grads()
    
    def test_main_loop(self):
        content = np.random.uniform(0, 255, (512, 512, 3))
        style = np.random.uniform(0, 255, (512, 512, 3))

        self.model.create_styled_image(content, style, 1, './data/VGG_imagenet.npy')

if __name__ == '__main__':
    unittest.main()
