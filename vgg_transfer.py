from networks import vgg
import tensorflow as tf

def gram_matrix(x):
    
    x_flatten = tf.reshape(x, (x.get_shape()[0], -1))
    return tf.matmul(x_flatten, x_flatten, transpose_b=True)

def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)

    channels = 3
    
    return tf.reduce_sum(tf.square(S-C)) / (4. * (channels ** 2) )
         

class VGG_Style(vgg.VGGNet):
    def __init__(self):
        super(VGG_Style, self).__init__()
        self.loss = tf.Variable(0.) 

    def build_content_loss(self):
        layer_features = self.get_output('conv2_2') 
        content_image_features = layer_features[0, :, :, :]
        combination_image_features = layer_features[2, :, :, :]

        content_loss = tf.reduce_sum(tf.square(content_image_features - combination_image_features))
        self.loss += content_loss
        

    def build_style_loss(self):
        feature_layers = ['conv1_2, conv2_2', 'conv3_3', 'conv4_3', 'conv5_3']

        for layer_name in feature_layers:
            layer_features = self.get_output(layer_name)
            style_features = layer_features[1, :, :, :]
            combination_features = layer_features[2, :, :, :]
            sl = style_loss(style_features, combination_features)
            self.loss += sl

        
        
        

        



