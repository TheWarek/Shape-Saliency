import tensorflow as tf
from tensorflow.contrib.layers import flatten




class SALSH: #Saliency Shape

    # Create model
    def __init__(self, psize=32):
        self.p_size = psize
        self.x = tf.placeholder(tf.float32, [None, self.p_size, self.p_size, 1], 'x_input')
        self.y = tf.placeholder(tf.float32, [None, self.p_size, self.p_size, 1], 'y_output')

        self.keep_prob = tf.placeholder_with_default(0.5, shape=(), name='keep_prob')
        self.training = tf.placeholder_with_default(True, shape=(), name='training')

        with tf.variable_scope("shape_saliency", reuse=tf.AUTO_REUSE) as scope:

            self.loss = self.loss()


    def variable_summaries(self, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))

    def layer_summary(self, layer_name, input_tensor, output):
        with tf.name_scope(layer_name):
            # This Variable will hold the state of the weights for the layer
            with tf.name_scope('weights'):
                weights = tf.get_variable(layer_name+'/kernel')
                self.variable_summaries(var=weights)
            if self.use_bias:
                with tf.name_scope('biases'):
                    biases = tf.get_variable(layer_name+'/bias')
                    self.variable_summaries(var=biases)

    def loss(self):
        return 0
