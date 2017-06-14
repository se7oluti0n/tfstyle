from networks import vgg
import tensorflow as tf
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time
import cv2
import argparse
import matplotlib.pyplot as plt
import math

def total_variation_loss(img):
    a = img[:, :-1, :-1, :] - img[:, :-1, 1:, :]
    b = img[:, :-1, :-1, :] - img[:, 1:, :-1, :]
    return tf.reduce_sum(tf.square(a) + tf.square(b))

def gram_matrix(x):
    shape = tf.shape(x)
    maps = tf.transpose(x, (2, 0, 1))
    flatten_maps = tf.reshape(maps, [shape[2], -1])
    
    gram = tf.matmul(flatten_maps, tf.transpose(flatten_maps, (1, 0)))
    gram = gram / tf.cast(tf.reduce_prod(shape), tf.float32)
    return gram

def content_loss(content, combination):
    return tf.reduce_sum(tf.square(content - combination))

def style_loss(style, combination):
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

def visualize_layers(layers):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    filters = layers.shape[3]
    plt.figure(1, figsize=(20,20), dpi=200)
    n_columns = 6
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(16):
        plt.subplot(4, 4, i+1)
        img = layers[0,:,:,i]
        res = clahe.apply(img.astype(np.uint16))
        plt.imshow(res, interpolation="nearest", cmap="gray")
    plt.show()

class VGG_Style(vgg.VGGNet):
    def __init__(self, content_weight, style_weight, tv_weight, h = 244, w = 244):

        data  = tf.Variable(tf.zeros([1, h, w, 3]), dtype=tf.float32, name='data')

        super(VGG_Style, self).__init__(data, trainable=False)
        self.height = h
        self.width = w
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight


    def build_content_loss(self, content_image_features, feature_layer):
        layer_features = self.get_output(feature_layer) 
        combination_image_features = layer_features[0, :, :, :]

        content_loss = content_loss(combination_image_features , content_image_features )
        return self.content_weight * content_loss

    def build_style_loss(self, features, feature_layers):

        style_losses = []


        for i, layer_name in enumerate(feature_layers):
            style_features = features[i]
            layer_features = self.get_output(layer_name)
            combination_features = layer_features

            sl = style_loss(style_features[0], combination_features[0])
            style_losses += [self.style_weight[i] * sl]
      
        return style_losses

    def build_total_variation_loss(self):
        tv_loss = total_variation_loss(self.data)
        return self.tv_weight * tv_loss

    def build_loss_grads(self, content, content_layers, style, style_layers):
        self.content_loss = self.build_content_loss(content, content_layers)
        self.style_losses = self.build_style_loss(style, style_layers)
        self.loss = 0

        for l in self.style_losses:
            self.loss += l

        self.tv_loss = self.build_total_variation_loss()
        self.loss = self.loss + self.content_loss + self.tv_loss

    def extract_content_features(self, sess, content_image, feature_layer):
        layer_features = self.get_output(feature_layer)
        return sess.run(layer_features, feed_dict={self.data: content_image[None, ...]})

    def extract_style_features(self, sess, style_image, feature_layers):

        layer_features = []
        for layer_name in feature_layers:
            layer_features.append(self.get_output(layer_name))
        
        return sess.run(layer_features, feed_dict={self.data: style_image[None, ...]}) 

    def create_styled_image(self, content, style, n_iterations, model_path):

        with tf.Session() as sess:

            #Normalize input to VGG16's format
            content = zero_centered_vgg(content)
            style = zero_centered_vgg(style)

            # Define content and style layers
            content_layer = 'conv2_2'
            style_layers = ['conv1_2', 'conv2_2',
                  'conv3_3', 'conv4_3',
                  'conv5_3']
      # Initialize graph and load pretrained model

            sess.run(tf.global_variables_initializer())
            self.load(model_path, sess, ignore_missing=True)

            content_features = self.extract_content_features(sess, content, content_layer)
            visualize_layers(content_features)
            style_features = self.extract_style_features(sess, style, style_layers)
            for features in style_features:
                visualize_layers(features)
            print ('Style features shpae: ', style_features[0].shape)

            # Build loss with extraced features
            self.build_loss_grads(content_features, \
                                  content_layer,
                                  style_features, \
                                  style_layers)

            # Define gradient
            with tf.variable_scope("optimizer") as opt_scope:
                optimizer = tf.train.AdamOptimizer(3)
                grad_vars = optimizer.compute_gradients(self.loss, [self.data])
                train_op = optimizer.apply_gradients(grad_vars)

            opt_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=opt_scope.name)
            sess.run(tf.variables_initializer(opt_vars))

            # Create random image to generate
            x = np.random.uniform(0, 255, (1, self.height, self.width, 3)) - 128.
            sess.run(tf.assign(self.data, x))

            # Run optimizer to generate data
            for i in range(n_iterations):
                print('Start of iteration', i)
                start_time = time.time()

                sess.run([train_op])
                end_time = time.time()
                print('Iteration %d completed in %lf s' % (i, end_time - start_time))
                if i % 50 == 49:
                    loss_, x = sess.run([self.loss, self.data])

                    print('Current loss value:', loss_)
                    sample_image(np.copy(x[0]), 'tmp/sample_'+ str(i) + '.jpg')

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
