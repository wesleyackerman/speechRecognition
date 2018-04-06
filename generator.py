import tensorflow as tf
import os
import tqdm, trange


class GeneratorTrainer:
    def __init__(self):
        self.model_directory = './model'
        self.sess = tf.Session()
        self.saver = tf.train.Saver()

    def train_generator(self, load_existing_model=True):
        if load_existing_model and os.path.exists(self.model_directory):
            self.load_trained_model()
        else:
            self.sess.run(tf.global_variables_initializer())

    def save_trained_model(self):
        self.saver.save(self.sess, os.path.join(self.model_directory, 'model'))

    def load_trained_model(self):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(self.model_directory))


if __name__ == '__main__':
    GeneratorTrainer().train_generator()
