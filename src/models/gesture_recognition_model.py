from src.base.base_model import BaseModel
import tensorflow as tf


class GestureRecognitionModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)

        self.is_training = None
        self.x = None
        self.y = None
        self.train_step = None
        self.saver = None
        self.hold_prob = None

        self.build_model()
        self.init_saver()

    def __init_weights(self, shape):
        initializer = tf.contrib.layers.xavier_initializer()
        init_xavier = initializer(shape)
        return tf.Variable(init_xavier)

    def __init_bias(self, shape):
        init_bias_vals = tf.constant(0.1, shape=shape)
        return tf.Variable(init_bias_vals)

    def __max_pool_2d(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def __convolution_layer(self, input_x, shape, strides=None):
        if strides is None:
            strides = [1, 1, 1, 1]
        W = self.__init_weights(shape)
        b = self.__init_bias([shape[-1]])
        return tf.nn.conv2d(input_x, W, strides=strides, padding='SAME') + b

    def __normal_full_layer(self, input_layer, size):
        input_size = int(input_layer.get_shape()[1])  # becase 0 is the number of training examples.
        W = self.__init_weights([input_size, size])
        b = self.__init_bias([size])
        return tf.matmul(input_layer, W) + b

    def __batch_norm(self, input_layer):
        return tf.contrib.layers.batch_norm(input_layer, activation_fn=tf.nn.leaky_relu, is_training=self.is_training)

    def __conv_bn_layer(self, input_layer, shape, strides=None, use_bn=True):
        if strides is None:
            strides = [1, 1, 1, 1]
        _conv = self.__convolution_layer(input_layer, shape=shape, strides=strides)
        if use_bn:
            return self.__batch_norm(_conv)
        else:
            return tf.nn.leaky_relu(_conv)

    def __conv_with_lr(self, input_x, shape, strides=None):
        if strides is None:
            strides = [1, 1, 1, 1]
        W = self.__init_weights(shape)
        b = self.__init_bias([shape[-1]])
        return tf.nn.leaky_relu(tf.nn.conv2d(input_x, W, strides=strides, padding='SAME') + b)

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)

        self.x = tf.placeholder(tf.float32, shape=[None] + self.config.state_size)
        self.y = tf.placeholder(tf.float32, shape=[None, 10])
        self.hold_prob_conv = tf.placeholder(tf.float32)
        self.hold_prob_fc = tf.placeholder(tf.float32)

        convo_1 = self.__conv_bn_layer(self.x, shape=[3, 3, 1, 8])
        convo_2 = self.__conv_bn_layer(convo_1, shape=[3, 3, 8, 16])
        convo_1_pooling = self.__max_pool_2d(convo_2)
        dropout_conv_1 = tf.nn.dropout(convo_1_pooling, self.hold_prob_conv)

        convo_3 = self.__conv_bn_layer(dropout_conv_1, shape=[3, 3, 16, 32])
        convo_4 = self.__conv_bn_layer(convo_3, shape=[3, 3, 32, 64])
        convo_2_pooling = self.__max_pool_2d(convo_4)
        dropout_conv_2 = tf.nn.dropout(convo_2_pooling, self.hold_prob_conv)


        convo_5 = self.__conv_bn_layer(dropout_conv_2, shape=[3, 3, 64, 128])
        convo_6 = self.__conv_bn_layer(convo_5, shape=[3, 3, 128, 256])
        convo_3_pooling = self.__max_pool_2d(convo_6)
        dropout_conv_3 = tf.nn.dropout(convo_3_pooling, self.hold_prob_conv)


        flattened = tf.reshape(dropout_conv_3,
                               [-1, 8 * 8 * 256])

        full_layer_1 = self.__normal_full_layer(flattened, 128)
        fully_bn_1 = self.__batch_norm(full_layer_1)
        full_dropout_1 = tf.nn.dropout(fully_bn_1, self.hold_prob_fc)

        full_layer_2 = self.__normal_full_layer(full_dropout_1, 128)
        fully_bn_1 = self.__batch_norm(full_layer_2)

        full_dropout_2 = tf.nn.dropout(fully_bn_1, self.hold_prob_fc)

        self.y_pred = self.__normal_full_layer(full_dropout_2, 10)

        with tf.name_scope("loss"):
            self.cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.y_pred))

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.cross_entropy,
                                                                                             global_step=self.global_step_tensor)
            correct_predictions = tf.equal(tf.argmax(tf.nn.softmax(self.y_pred), 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
            self.predictions = tf.argmax(tf.nn.softmax(self.y_pred), 1)

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep, save_relative_paths=True)
