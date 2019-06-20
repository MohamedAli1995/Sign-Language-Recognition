import numpy as np
from src.utils.utils import unpickle
from src.data_loader.preprocessing import paths_to_images, one_hot_encoding
from random import shuffle
from glob import glob


class DataGenerator:
    """DataGenerator class responsible for dealing with cifar-100 dataset.

    Attributes:
        config: Config object to store data related to training, testing and validation.
        all_train_data: Contains the whole dataset(since the dataset fits in memory).
        x_all_train: Contains  the whole input training-data.
        x_all_train: Contains  the whole target_output labels for training-data.
        x_train: Contains training set inputs.
        y_train: Contains training set target output.
        x_val: Contains validation set inputs.
        y_val: Contains validation set target output.
        meta: Contains meta-data about Cifar-100 dataset(including label names).
    """

    def __init__(self, config, training=True, data_split_seed=64):
        self.config = config
        self.training = training
        self.data_split_seed = data_split_seed
        self.x_all_data = None
        self.y_all_data = None
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None
        self.all_test = None

        self.num_batches_train = None
        self.num_batches_val = None
        self.num_batches_test = None

        self.indx_batch_train = 0
        self.indx_batch_val = 0
        self.indx_batch_test = 0

        if self.training:
            np.random.seed(self.data_split_seed)
            if config.split_val_test:
                self.__load_train_val_test(train_ratio=0.8, val_ratio=0.1)
            else:
                self.__load_train_val_test(train_ratio=0.8, val_ratio=0.2)

    def __load_train_val_test(self, train_ratio=0.8, val_ratio=0.1):
        """Private function.
        Returns:
        """
        if train_ratio + val_ratio > 1.0:
            print("Wrong data splitting ratio.")
            return

        self.x_train = np.empty(shape=[0, 1])
        self.y_train = np.empty(shape=[0, 10], dtype=int)

        self.x_val = np.empty(shape=[0, 1])
        self.y_val = np.empty(shape=[0, 10], dtype=int)

        self.x_test = np.empty(shape=[0, 1])
        self.y_test = np.empty(shape=[0, 10], dtype=int)

        for i in range(10):
            paths = np.asanyarray(glob(self.config.train_data_path + str(i) + "/*"))
            perm = np.random.permutation(paths.shape[0])  # To randomly data per class to split.
            split_point_1 = int(train_ratio * len(paths))
            split_point_2 = int((train_ratio + val_ratio) * len(paths))

            self.x_train = np.append(self.x_train,
                                     paths[perm[0:split_point_1]])
            self.y_train = np.concatenate((self.y_train,
                                           np.asanyarray(
                                               [one_hot_encoding(i, 10) for x in range(split_point_1 - 0)]
                                           )))

            self.x_val = np.append(self.x_val,
                                   paths[perm[split_point_1:split_point_2]])
            self.y_val = np.concatenate((self.y_val,
                                         np.asanyarray(
                                             [one_hot_encoding(i, 10) for x in range(split_point_2 - split_point_1)]
                                         )))

            self.x_test = np.append(self.x_test,
                                    paths[perm[split_point_2:perm.shape[0]]])
            self.y_test = np.concatenate((self.y_test,
                                          np.asanyarray(
                                              [one_hot_encoding(i, 10) for x in range(perm.shape[0] - split_point_2)]
                                          )))

        if (self.x_test.shape[0] != self.y_test.shape[0]):
            print(" Shape mismatch between test data and labels.")
            return
        if (self.x_val.shape[0] != self.y_val.shape[0]):
            print(" Shape mismatch between test data and labels.")
            return
        if (self.x_train.shape[0] != self.y_train.shape[0]):
            print(" Shape mismatch between test data and labels.")
            return

        self.num_batches_train = int(np.ceil(self.x_train.shape[0] / self.config.batch_size))
        self.num_batches_val = int(np.ceil(self.x_val.shape[0] / self.config.batch_size))
        self.num_batches_test = int(np.ceil(self.x_test.shape[0] / self.config.batch_size))
        self.__shuffle_all_data()


    def __shuffle_all_data(self):
        """Private function.
        Shuffles the whole training set to avoid patterns recognition by the model(I liked that course:D).
        shuffle function is used instead of sklearn shuffle function in order reduce usage of
        external dependencies.

        Returns:
        """
        indices_list = [i for i in range(self.x_train.shape[0])]
        shuffle(indices_list)
        # Next two lines may cause memory error if no sufficient ram.
        self.x_train = self.x_train[indices_list]
        self.y_train = self.y_train[indices_list]

        indices_list = [i for i in range(self.x_val.shape[0])]
        shuffle(indices_list)
        # Next two lines may cause memory error if no sufficient ram.
        self.x_val = self.x_val[indices_list]
        self.y_val = self.y_val[indices_list]


    def load_test_set(self, path):
        self.x_test = np.asanyarray(glob(path + "/*"))
        self.all_test = self.x_test
        self.num_batches_test = int(np.ceil(self.x_test.shape[0] / self.config.batch_size))


    def load_single_image(self, path):
        self.x_test = np.asanyarray([path])
        self.all_test = self.x_test
        self.num_batches_test = 1




    def prepare_new_epoch_data(self):
        """Prepares the dataset for a new epoch by setting the indx of the batches to 0 and shuffling
        the training data.

        Returns:
        """
        self.indx_batch_train = 0
        self.indx_batch_val = 0
        self.indx_batch_test = 0
        self.__shuffle_all_data()

    def next_batch(self, batch_type="train"):
        """Moves the indx_batch_... pointer to the next segment of the data.

        Args:
            batch_type: the type of the batch to be returned(train, test, validation, unlabeled_test).

        Returns:
            The next batch of the data with type of batch_type.
        """
        if batch_type == "unlabeled_test":
            x = self.x_test[self.indx_batch_test:self.indx_batch_test + self.config.batch_size]
            self.indx_batch_test = (self.indx_batch_test + self.config.batch_size) % self.x_test.shape[0]
            x = paths_to_images(x, self.config.state_size)
            return x

        if batch_type == "train":
            x = self.x_train[self.indx_batch_train:self.indx_batch_train + self.config.batch_size]
            y = self.y_train[self.indx_batch_train:self.indx_batch_train + self.config.batch_size]
            self.indx_batch_train = (self.indx_batch_train + self.config.batch_size) % self.x_train.shape[0]

        elif batch_type == "val":
            x = self.x_val[self.indx_batch_val:self.indx_batch_val + self.config.batch_size]
            y = self.y_val[self.indx_batch_val:self.indx_batch_val + self.config.batch_size]
            self.indx_batch_val = (self.indx_batch_val + self.config.batch_size) % self.x_val.shape[0]

        elif batch_type == "test":
            x = self.x_test[self.indx_batch_test:self.indx_batch_test + self.config.batch_size]
            y = self.y_test[self.indx_batch_test:self.indx_batch_test + self.config.batch_size]
            self.indx_batch_test = (self.indx_batch_test + self.config.batch_size) % self.x_test.shape[0]

        x = paths_to_images(x, self.config.state_size)
        return x, y
