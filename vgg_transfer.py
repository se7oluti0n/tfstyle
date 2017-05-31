from networks import vgg
import tensorflow as tf
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time
import cv2
import argparse

def total_variation_loss(x, h, w):
    a = tf.square(x[:h-1, :w-1, :] - x[1:, :w-1, :])
    b = tf.square(x[:h-1, :w-1, :] - x[:h-1, 1:, :])
    return tf.reduce_sum(tf.pow(a+b, 1.25))

def gram_matrix(x):
    x = tf.transpose(x, (2, 0, 1))
    x_flatten = tf.reshape(x, [tf.shape(x)[0], -1])

    return tf.matmul(x_flatten, x_flatten, transpose_b=True)

def style_loss(style, combination, h, w):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    size = h * w
    channels = 3

    return tf.reduce_sum(tf.square(S-C)) / (4. * (channels ** 2) * (size ** 2))

def zero_centered_vgg(img):
    img = img.astype(np.float32)
    img = img[None, :, :, :]
    img[:, :, :, 0] -= 103.939
    img[:, :, :, 1] -= 116.779
    img[:, :, :, 2] -= 123.68

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
        data = tf.Variable(tf.zeros([3, h, w, 3]), name='data', dtype=tf.float32)
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
        for layer_name in feature_layers:
            layer_features = self.get_output(layer_name)
            style_features = layer_features[1, :, :, :]
            combination_features = layer_features[2, :, :, :]
            sl = style_loss(style_features, combination_features, self.height, self.width)
            style_losses += [(self.style_weight / len(feature_layers)) * sl]
      
        return style_losses

    def build_total_variation_loss(self):
        tv_loss = total_variation_loss(self.combination_image[:, :, :], self.height, self.width)
        return self.tv_weight * tv_loss

    def build_loss_grads(self):
        self.loss = self.build_content_loss()
        style_losses = self.build_style_loss()
        for l in style_losses:
            self.loss += l
        self.loss += self.build_total_variation_loss()
#        self.combination_grads = tf.gradients(self.loss, [self.combination_image])[0]

    def create_styled_image(self, content, style, n_iterations, model_path):
        with tf.Session() as sess:

            content = zero_centered_vgg(content)
            style = zero_centered_vgg(style)

            self.build_loss_grads()
            
            optimizer = tf.train.AdamOptimizer(10)
            grad_vars = optimizer.compute_gradients(self.loss, [self.data])
            train_op = optimizer.apply_gradients(grad_vars)

            input_data_var = self.data[:2, :, :, :]

            self.load(model_path, sess, ignore_missing=True)

            sess.run(tf.global_variables_initializer())

            x = np.random.uniform(0, 255, (1, self.height, self.width, 3)) - 128.

            def evaluate_loss_grads(x):
                assert self.loss_result is None
                x = x.reshape((1, self.height, self.width, 3))

                combined_image = np.concatenate([content, style, x])
                loss_, grad_ = sess.run([self.loss, self.combination_grads], 
                    feed_dict={self.data: combined_image})

                self.loss_result = loss_
                self.grad_result = grad_.flatten().astype(np.float64)
                return self.loss_result

            def evaluate_grads(x):
                assert self.loss_result is not None
                grad_values = np.copy(self.grad_result)
                self.loss_result = None
                self.grad_result = None

                return grad_values


            combined_image = np.concatenate([content, style, x])
            sess.run(tf.assign(self.data, combined_image))
            

            for i in range(n_iterations):
                print('Start of iteration', i)
                start_time = time.time()

#                x, min_val, info = fmin_l_bfgs_b(evaluate_loss_grads, x.flatten(),
#                                                evaluate_grads, maxfun=20)
                input_image = np.concatenate([content, style])

                sess.run([train_op], feed_dict={input_data_var: input_image})

                loss_, comb = sess.run([self.loss, self.data])
                x = comb[2, :, :, :]

                print('Current loss value:', loss_)
                end_time = time.time()
                print('Iteration %d completed in %ds' % (i, end_time - start_time))
                if i % 10 == 0:
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

    model = VGG_Style(0.025, 5.0, 1.0, args.height, args.width)
    output_img = model.create_styled_image(content_img, style_img, args.iters, './data/VGG_imagenet.npy')

    cv2.imwrite(args.out_file, output_img + 128)
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
