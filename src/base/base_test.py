import tensorflow as tf


class BaseTest:
    """Standard base_test-class for easy multiple-inheritance.
    It defines the functions needed in model testing.

    Attributes:
        sess: Tensorflow session to use.
        model: Model to be trained.
        data: Data_loader object to interact with dataset.
        config: Config object to store data related to training, testing and validation.
        logger: Logger object to use tensorboard.
    """

    def __init__(self, sess, model, data, config, logger):
        self.model = model
        self.config = config
        self.sess = sess
        self.data = data
        self.logger = logger

    def test(self):
        """Test model that is in config.checkpoint_dir.
        Calls test_step per batch in test_set.
        Args:
        Returns:
            """
        raise NotImplemented

    def test_step(self):
        raise NotImplemented
