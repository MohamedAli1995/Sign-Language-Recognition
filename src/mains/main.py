import tensorflow as tf
import numpy as np
import cv2
from src.data_loader.data_generator import DataGenerator
from src.models.gesture_recognition_model import GestureRecognitionModel
from src.trainers.gesture_recognition_trainer import GestureRecognitionTrainer
from src.testers.gesture_recognition_tester import GestureRecognitionTester
from src.utils.config import processing_config
from src.utils.logger import Logger
from src.utils.utils import get_args
from src.utils.utils import print_predictions


def main():
    args = None
    config = None
    try:
        args = get_args()
        print(args.config)
        # config = processing_config(
        #     "/media/syrix/programms/projects/Sign-Language-Recognition/configs/config_model.json")
        # config = processing_config(args.config)
    except:
        print("Missing or invalid arguments")
        exit(0)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = config.per_process_gpu_memory_fraction
    sess = tf.Session()
    logger = Logger(sess, config)
    model = GestureRecognitionModel(config)
    model.load(sess)

    if args.test_path is not None:
        data = DataGenerator(config, training=False)
        data.load_test_set(args.test_path)
        tester = GestureRecognitionTester(sess, model, data, config, logger)
        predictions = tester.predict()
        print_predictions(data.all_test, predictions)
        return

    if args.img_path is not None:
        data = DataGenerator(config, training=False)
        data.load_single_image(args.img_path)

        tester = GestureRecognitionTester(sess, model, data, config, logger)
        predictions = tester.predict()
        print_predictions(data.all_test, predictions)
        return

    data = DataGenerator(config, training=True)

    print("Dataset:\n  train_samples:%d \n  val_samples:%d \n  test_samples:%d\n"
          % (data.x_train.shape[0], data.x_val.shape[0], data.x_test.shape[0]))

    trainer = GestureRecognitionTrainer(sess, model, data, config, logger)
    trainer.train()
    tester = GestureRecognitionTester(sess, model, data, config, logger)
    tester.test()


if __name__ == '__main__':
    main()
