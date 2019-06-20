import tensorflow as tf


class BaseTrain:
    """Standard base_train-class for easy multiple-inheritance.
        It is responsible for defining the functions to be implemented with any child.

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
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    def train(self):
        """Train the model for the number of epochs in config.num_epochs.
        Calls validate_epoch if config.use_val is set to true and per config.val_per_epoch.
        Returns:

        """

        # If model is loaded -> don't initialize global step or cur epoch.
        if not self.model.loaded_saved_model:
            self.sess.run(self.init)

        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs + 1, 1):
            self.data.prepare_new_epoch_data()
            self.train_epoch()
            if self.config.use_val and (
                                cur_epoch % self.config.val_per_epoch == 0 or cur_epoch == self.config.num_epochs):
                self.validate_epoch()

            self.sess.run(self.model.increment_cur_epoch_tensor)

    def train_epoch(self):
        """Implements the logic of training_epoch:
        -Loop over the batches of the training data and call the train step for each.
        -Add any summaries you want using the summary
        """
        raise NotImplemented

    def train_step(self):
        """Implements the logic of the train step:
        -Run the tensorflow session
        -Returns:
         Any of the metrics needs to be summarized.
        """

        raise NotImplementedError

    def validate_epoch(self):
        """Implements the logic of validation_epoch:
        -Loop over the batches of the validation data and call the validate step for each.
        -Add any summaries you want using the summary
        """
        raise NotImplemented

    def validate_step(self):
        """Implements the logic of the validate step:
        -Run the tensorflow session
        -Returns:
         Any of the metrics needs to be summarized.
        """
        raise NotImplemented
