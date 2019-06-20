import tensorflow as tf
from src.base.base_test import BaseTest
from tqdm import tqdm
import numpy as np
import cv2
class SimpleTester(BaseTest):
    def __init__(self, sess, model, data, config, logger):
        super().__init__(sess, model, data, config, logger)


    def test(self):
        loop = tqdm(range(self.data.num_batches_test))
        losses = []
        accs = []
        for _ in loop:
            loss, acc = self.test_step()
            losses.append(loss)
            accs.append(acc)
        loss = np.mean(losses)
        acc = np.mean(accs)
        print("test_accuracy: ",
              acc * 100, "% train_loss: ", loss)

    def test_step(self):
        batch_x, batch_y = self.data.next_batch(batch_type="test")
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.is_training: False,
                     self.model.hold_prob: 1.0}

        loss, acc = self.sess.run([self.model.cross_entropy, self.model.accuracy],
                                     feed_dict=feed_dict)
        return loss, acc




