import tensorflow as tf


class BaseModel:
    """Standard base_model-class for easy multiple-inheritance.

    Attributes:
        config: Config object to store data related to training, testing and validation.
    """

    def __init__(self, config):
        self.config = config
        self.init_global_step()
        self.init_cur_epoch()
        self.loaded_saved_model = False

    def save(self, sess):
        """
        Saves model to config.checkpoint_dir with global_step.

        Args:
            sess: tensorflow session to work with.
        Returns:
            """
        print("Saving model...")
        self.saver.save(sess, self.config.checkpoint_dir, self.global_step_tensor)
        print("Model saved")

    def load(self, sess):
        """Loads model in directory config.checkpoint_dir.

        Args:
            sess: tensorflow session to work with.
        Returns:
            """
        latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint{} ...".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            self.loaded_saved_model = True
            print("Model loaded")

    def init_cur_epoch(self):
        """Initialize current epoch with 0.
        Args:
        Returns:
            """
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

    def init_global_step(self):
        """Initialize current epoch with 0.
        Args:
        Returns:
            """
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

    def init_saver(self):
        """Initialize tensorflow saver.
        Args:
        Returns:
            """
        raise NotImplemented

    def build_model(self):
        """Build the architecture of the model.
        Args:
        Returns:
            """
        raise NotImplemented
