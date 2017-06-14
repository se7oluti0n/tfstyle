import unittest
import sys
sys.path.append('..')
from networks import transform 
import tensorflow as tf

class test_transform(unittest.TestCase):
    def test_load_npy(self):
        data  = tf.Variable(tf.zeros([1, 512, 512, 3]), dtype=tf.float32, name='data')
        model = transform.TransformNet(data=data)


if __name__ == '__main__':
    unittest.main()
