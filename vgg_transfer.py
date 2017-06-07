from networks import vgg
import tensorflow as tf
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time
import cv2
import argparse

def total_variation_loss(img, h, w):
    a = img[:-1, :-1, :] - img[:-1, 1:, :]
    b = img[:-1, :-1, :] - img[1:, :-1, :]
    return tf.reduce_sum(tf.square(a) + tf.square(b))

def gram_matrix(x):
    shape = tf.shape(x)
    maps = tf.transpose(x, (2, 0, 1))
    flatten_maps = tf.reshape(maps, [shape[2], -1])
    
    gram = tf.matmul(flatten_maps, tf.transpose(flatten_maps, (1, 0)))
    
    gram = gram / tf.cast(tf.reduce_prod(shape), tf.float32)
    return gram

def style_loss(style, combination, h, w):
    S = gram_matrix(style)
    C = gram_matrix(combination)

    return tf.reduce_sum(tf.square(S-C)) 

def zero_centered_vgg(img):
    img = img.astype(np.float32)
#    img = img[None, :, :, :]
    img[ :, :, 0] -= 103.939
    img[ :, :, 1] -= 116.779
    img[ :, :, 2] -= 123.68

    return img

def sample_image(x, filename):
    
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = np.clip(x, 0, 255).astype('uint8')
    cv2.imwrite(filename, x)


class VGG_Style(vgg.VGGNet):
    def __init__(self, content_weight, style_weight, tv_weight, h = 244, w = 244):
        init_weights = tf.truncated_normal_initializer(0.0, stddev=0.001)
        self.content_var = tf.Variable(tf.zeros([ h, w, 3]), dtype=tf.float32, trainable=False)
        self.style_var = tf.Variable(tf.zeros([h, w, 3]), dtype=tf.float32, trainable=False)
        self.comb_var  = tf.Variable(tf.zeros([h, w, 3]), dtype=tf.float32)

        data = tf.stack([self.content_var, self.style_var, self.comb_var], name='data')
        super(VGG_Style, self).__init__(data, trainable=True)
        self.height = h
        self.width = w
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight
        self.loss_result = None
        self.grad_result = None

        self.combination_image = self.data[2, :, :, :]

    def build_content_loss(self):
        layer_features = self.get_output('conv2_2') 
        content_image_features = layer_features[0, :, :, :]
        combination_image_features = layer_features[2, :, :, :]

        content_loss = tf.reduce_sum(tf.square(combination_image_features - content_image_features ))
        return self.content_weight * content_loss

    def build_style_loss(self):
        feature_layers = ['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3', 'conv5_3']
        style_losses = []        
        for i, layer_name in enumerate(feature_layers):
            layer_features = self.get_output(layer_name)
            style_features = layer_features[1, :, :, :]
            combination_features = layer_features[2, :, :, :]
            sl = style_loss(style_features, combination_features, self.height, self.width)
            style_losses += [self.style_weight[i] * sl]
      
        return style_losses

    def build_total_variation_loss(self):
        tv_loss = total_variation_loss(self.combination_image[:, :, :], self.height, self.width)
        return self.tv_weight * tv_loss

    def build_loss_grads(self):
        self.content_loss = self.build_content_loss()
        self.style_losses = self.build_style_loss()
        self.loss = 0
        for l in self.style_losses:
            self.loss += l

        self.tv_loss = self.build_total_variation_loss()
        self.loss = self.loss + self.content_loss + self.tv_loss
#        self.combination_grads = tf.gradients(self.loss, [self.combination_image])[0]

    def create_styled_image(self, content, style, n_iterations, model_path):
        with tf.Session() as sess:

            content = zero_centered_vgg(content)
            style = zero_centered_vgg(style)

            self.build_loss_grads()
            
            optimizer = tf.train.AdamOptimizer(3)
            grad_vars = optimizer.compute_gradients(self.loss, [self.comb_var])
            train_op = optimizer.apply_gradients(grad_vars)

            sess.run(tf.global_variables_initializer())
            self.load(model_path, sess, ignore_missing=True)

            

            x = np.random.uniform(0, 255, (self.height, self.width, 3)) - 128.


            sess.run(tf.assign(self.content_var, content))
            sess.run(tf.assign(self.style_var, style))
            sess.run(tf.assign(self.comb_var, x))


            for i in range(n_iterations):
                print('Start of iteration', i)
                start_time = time.time()

                sess.run([train_op])
                end_time = time.time()
                print('Iteration %d completed in %ds' % (i, end_time - start_time))
                if i % 50 == 49:
                    loss_, x = sess.run([self.loss, self.comb_var])

                    print('Current loss value:', loss_)
                    sample_image(np.copy(x), 'sample_'+ str(i) + '.jpg')

            x = x.reshape(self.height, self.width, 3)
            x[:, :, 0] += 103.939
            x[:, :, 1] += 116.779
            x[:, :, 2] += 123.68
            x = np.clip(x, 0, 255).astype('uint8')

            return x

def main(args):

    content_img = cv2.imread(args.content_file)
    style_img = cv2.imread(args.style_file)

    content_img = cv2.resize(content_img, (args.height, args.width))
    style_img = cv2.resize(style_img, (args.height, args.width))

    model = VGG_Style(10, [3e5, 1e3, 15, 1, 1] , 2e-2, args.height, args.width)
    output_img = model.create_styled_image(content_img, style_img, args.iters, './data/VGG_imagenet.npy')

    cv2.imwrite(args.out_file, output_img)

def parse_augument():

    parser = argparse.ArgumentParser()

    parser.add_argument('content_file', type=str, help='Path to content file')
    parser.add_argument('style_file', type=str, help='Path to style file')
    parser.add_argument('out_file', type=str, help='Path to output file')

    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--iters', type=int, default=10)
    

    return parser.parse_args()

if __name__ == '__main__':
    main(parse_augument())
