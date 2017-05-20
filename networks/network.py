import numpy as np
import tensorflow as tf

DEFAULT_PADDING = 'SAME'
def layer(op):
    def layer_decorated(self, *args, **kwargs):
        #Automatic set a name if not provided
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))

        # Figure out the layer input
        if len(self.inputs) == 0:
            raise RuntimeError('No input variable found for layer %s.' % name)
        elif len(self.inputs) == 1:
            layer_input = self.inputs[0]
        else:
            layer_input = list(self.input)

        # perform operation and get the output
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT
        self.layers[name] = layer_output
        # This output is now the input for the next layer
        self.feed(layer_output)

        return self
    return layer_decorated

class Network(object):
    def __init__(self, inputs, trainable=True):
        self.inputs = []
        self.layers = dict(inputs)
        self.trainable = trainable
        self.setup()

    def setup(self):
        raise NotImplementedError('Must be subclassed.')

    def load(self, data_path, session, ignore_missing=False):
        data_dict = np.load(data_path).item()

        for key in data_dict:
            with tf.variable_scope(key, reuse=True):
                for subkey in data_dict[key]:
                    try:
                        var = tf.get_variable(subkey)
                        session.run(var.assign(data_dict[key][subkey]))
                        print "assign pretrained model "+subkey+" to "+key
                    except ValueError:
                        print "ignore " + key
                        if not ignore_missing:
                            raise
    
    def feed(self, *args):
        assert len(args) != 0
        self.inputs = []

        for layer in args:
            if isinstance(layer, basestring):
                try:
                    layer = self.layers[layer]
                    print layer
                except KeyError:
                    print self.layers.keys()
                    raise KeyError('Unknown layer name fed: %s' % layer)
            self.inputs.append(layer)
        return self

    def get_output(self, layer):
        try:
            layer = self.layers[layer]
        except KeyError:
            print self.layers.keys()
            raise KeyError('Unknown layer name: %s' % layer)
        return layer

    def get_unique_name(self, prefix):
        id = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return "%s_%d" % (prefix, id)


    def make_var(self, name, shape, initializer=None, trainable=True, regularizer=None):
        return tf.get_variable(name, shape, initializer=initializer, trainable=trainable, regularizer=regularizer)

    def validate_padding(self, padding):
        assert padding in ('SAME', 'VALID')

    @layer
    def conv(self, input, k_h, k_w, c_o, s_h, s_w, name, biased=True, relu=True, padding=DEFAULT_PADDING, trainable=True):
        self.validate_padding(padding)
        c_i = input.get_shape()[-1]
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)

        with tf.variable_scope(name) as scope:

            init_weights = tf.contrib.layers.variance_scaling_initializer(factor=0.01, mode='FAN_AVG', uniform=False)
            init_biases = tf.constant_initializer(0.0)

            kernel = self.make_var('weights', [k_h, k_w, c_i, c_o], init_weights, trainable, \
                                        regularizer=self.l2_regularizer(2e-4))

            out = convolve(input, kernel)
            if biased:
                biases = self.make_var('biases', [c_o], init_biases, trainable)
                out =  tf.nn.bias_add(out, biases)
            
            if relu:
                out = tf.nn.relu(out)
            return out

    @layer
    def fc(self, input, num_out, name, relu=True, trainable=True):

        # Define scope
        with tf.variable_scope(name) as scope:
            if isinstance(input, tuple):
                input = input[0]
            # Reshape
            input_shape = input.get_shape()
            if input.ndims == 4:
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d

                feed_in = tf.reshape(tf.transpose(input, [0,3,1,2]), [-1, dim])
            else:
                feed_in, dim = (input, int(input_shape[-1]))

            # initializer
            if name == 'bbox_pred':
                init_weights = tf.truncated_normal_initializer(0.0, stddev=0.001)
                init_biases = tf.constant_initializer(0.0)
            else:
                init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
                init_biases = tf.constant_initializer(0.0)

            # declare variable

            weights = self.make_var('weights', [dim, num_out], init_weights, trainable, \
                                    regularizer=self.l2_regularizer(2e-4))
            biases = self.make_var('biases', [num_out], init_biases, trainable=True)

            # define op
            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)

            return fc

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)
    
    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.avg_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def dropout(self, input, keep_prob, name):
        return tf.nn.dropout(input, keep_prob=keep_prob, name=name)

    def l2_regularizer(self, weight_decay = 0.0005, scope=None):
        def regularizer(tensor):
            with tf.name_scope(scope, default_name='l2_regularizer', values=[tensor]):
                l2_weight = tf.convert_to_tensor(weight_decay, \
                                                dtype=tensor.dtype.base_dtype,
                                                name='weight_decay')
                return tf.multiply(l2_weight, tf.nn.l2_loss(tensor), name='value')
            return regularizer
