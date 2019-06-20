from src.base.base_model import BaseModel
import tensorflow as tf


class SimpleModel(BaseModel):
    def __init__(self, config):
        super(SimpleModel, self).__init__(config)

        self.is_training = None
        self.x = None
        self.y = None
        self.train_step = None
        self.saver = None
        self.hold_prob = None


        self.build_model()
        self.init_saver()

    def __init_weights(self, shape):
        init_random_dist = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(init_random_dist)

    def __init_bias(self, shape):
        init_bias_vals = tf.constant(0.1, shape=shape)
        return tf.Variable(init_bias_vals)

    def __conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def __max_pool_2d(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def __convolution_layer(self, input_x, shape):
        W = self.__init_weights(shape)
        b = self.__init_bias([shape[3]])
        return tf.nn.relu(self.__conv2d(input_x, W) + b)

    def __normal_full_layer(self, input_layer, size):
        input_size = int(input_layer.get_shape()[1])  # becase 0 is the number of training examples.
        W = self.__init_weights([input_size, size])
        b = self.__init_bias([size])
        return tf.matmul(input_layer, W) + b

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)

        self.x = tf.placeholder(tf.float32, shape=[None] + self.config.state_size)
        self.y = tf.placeholder(tf.float32, shape=[None, 100])
        self.hold_prob = tf.placeholder(tf.float32)

        convo_1 = self.__convolution_layer(self.x, shape=[3, 3, 3, 32])
        convo_2 = self.__convolution_layer(convo_1, shape=[3, 3, 32, 64])
        convo_2_pooling = self.__max_pool_2d(convo_2)

        convo_3 = self.__convolution_layer(convo_2_pooling, shape=[3, 3, 64, 128])
        convo_4 = self.__convolution_layer(convo_3, shape=[3, 3, 128, 256])
        convo_4_pooling = self.__max_pool_2d(convo_4)

        convo_2_flat = tf.reshape(convo_4_pooling,
                                  [-1, 8 * 8 * 256])  # 8*8 because of 2 max-pool operations each drop size to half

        full_layer_one = tf.nn.relu(self.__normal_full_layer(convo_2_flat, 1024))
        full_one_dropout = tf.nn.dropout(full_layer_one, self.hold_prob)
        y_pred = self.__normal_full_layer(full_one_dropout, 100)

        with tf.name_scope("loss"):
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=y_pred))
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.cross_entropy,
                                                                                             global_step=self.global_step_tensor)
            correct_predictions = tf.equal(tf.argmax(y_pred, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)