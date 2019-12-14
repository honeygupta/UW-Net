"""Code for training and testing UW-Net.
Author: Honey Gupta (hn.gpt1@gmail.com)
"""

import os
import csv
import json
import click
import random
import numpy as np

import tensorflow as tf
import skimage.io as io
from random import shuffle
from datetime import datetime
from skimage.io import imsave

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import uwnet_datasets
import data_loader, losses
from skimage.transform import resize
from model import discriminator_A, discriminator_B, denseNet

slim = tf.contrib.slim

BATCH_SIZE = 1
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3

def load_data(dataset_name):
    list1 = []
    csvfile = open(uwnet_datasets.PATH_TO_CSV[dataset_name], 'r')
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        list1.append(row[0])
    return list1

class Data(object):
    def __init__(self, list1, bs=1, shuffle=False):
        self.list1 = list1
        self.bs = bs
        self.index = 0
        self.number = len(self.list1)
        self.index_total = range(self.number)
        self.shuffle = shuffle
        if self.shuffle: 
            self.index_total = np.random.permutation(self.number)

    def next_batch(self):
        start = self.index
        self.index += self.bs
        if self.index > self.number:
            if self.shuffle: 
                self.index_total = np.random.permutation(self.number)
            self.index = 0
            start = self.index
            self.index += self.bs
        end = self.index
        img1_batch = []
        name = []
        for i in range(start, end):
            im1 = io.imread(self.list1[self.index_total[i]]).astype(np.float32) / 127.5
            im1 = im1 - 1.0
            im1 = resize(im1, [256,256], preserve_range = True)
            txt = self.list1[self.index_total[i]]
            name.append(txt[-32:])
            img1_batch.append(im1)
        return np.array(img1_batch), name

class UWNet:
    """The UWNet module."""

    def __init__(self, output_root_dir, max_step, network_version,
                 dataset_name, checkpoint_dir):
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

        self._output_dir = os.path.join(output_root_dir, current_time)
        self._images_dir = os.path.join(self._output_dir, 'imgs')
        self._num_imgs_to_save = 50
        self._network_version = network_version
        self._dataset_name = dataset_name
        self._checkpoint_dir = checkpoint_dir

    def model_setup(self):
        self.input_b = tf.placeholder(
            tf.float32, [
                1,
                IMG_WIDTH,
                IMG_HEIGHT,
                IMG_CHANNELS
            ], name="input_B")

        inputs = {
            'images_b': self.input_b
        }

        outputs = self.get_outputs(
            inputs, network=self._network_version)

        self.prob_real_b_is_real = outputs['prob_real_b_is_real']
        self.fake_images_a = outputs['fake_images_a']
        self.prob_fake_a_is_real = outputs['prob_fake_a_is_real']
        self.cycle_images_b = outputs['cycle_images_b']

    def save_images(self, sess, epoch):
        """
        Saves input and output images.

        :param sess: The session.
        :param epoch: Current epoch.
        """
        if not os.path.exists(self._images_dir):
            os.makedirs(self._images_dir)

        if not os.path.exists(os.path.join(self._images_dir, 'imgs')):
            os.makedirs(os.path.join(self._images_dir, 'imgs'))
        
        names = ['inputB_', 'fakeB_depth_' , 'cycB_']

        with open(os.path.join(
                self._output_dir, 'epoch_' + str(epoch) + '.html'), 'w') as v_html:
            for i in range(0, self._num_imgs_to_save):
                print("Saving image {}/{}".format(i, self._num_imgs_to_save))
                x1_t, name1 = self.dataset.next_batch()
                count = 0
                fake_A_temp, cyc_B_temp = sess.run([
                    self.fake_images_a,
                    self.cycle_images_b], 
                    feed_dict={self.input_b: x1_t})
                
                fakedepth = fake_A_temp[:,:,:,-1]
                tensors = [x1_t, fakedepth, cyc_B_temp]

                for name, tensor in zip(names, tensors):
                    #print(name)
                    # if name == 'inputB_' or name == 'fakeB_depth_':
                        # image_name = name1[count] + '_' + name + str(epoch) + "_" + str(i) + ".jpg"
                        # imsave(os.path.join(self._images_dir, 'imgs', image_name), ((tensor[0] + 1) * 127.5).astype(np.uint8))
                    # else:
                    image_name = name + str(epoch) + "_" + str(i) + ".jpg"
                    imsave(os.path.join(self._images_dir, image_name), ((tensor[0] + 1) * 127.5).astype(np.uint8))
                    v_html.write(
                        "<img src=\"" +
                        os.path.join('imgs', image_name) + "\">"
                    )
                v_html.write("<br>")
                count += 1

    def get_outputs(self, inputs, network="tensorflow"):
        images_b = inputs['images_b']

        with tf.variable_scope("Model") as scope:
            prob_real_b_is_real = discriminator_B(images_b, name="d_B")
            fake_images_a = denseNet(images_b, gen_type = 'B', name="g_B")
            
            # scope.reuse_variables()
            
            prob_fake_a_is_real = discriminator_A(fake_images_a, name="d_A")

            cycle_images_b = denseNet(fake_images_a, gen_type = 'A', name="g_A")

        return {
            'prob_real_b_is_real': prob_real_b_is_real,
            'prob_fake_a_is_real': prob_fake_a_is_real,
            'cycle_images_b': cycle_images_b,
            'fake_images_a': fake_images_a,
        }

    def test(self):
        """Test Function."""
        print("Testing the results")

        list1 = load_data(self._dataset_name)
        self.dataset = Data(list1, shuffle=False    )

        self.model_setup()
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True

        with tf.Session(config=config) as sess:
            sess.run(init)

            chkpt_fname = tf.train.latest_checkpoint(self._checkpoint_dir)
            saver.restore(sess, chkpt_fname)
            print("Restored the checkpoint: ", chkpt_fname)
            self._num_imgs_to_save = uwnet_datasets.DATASET_TO_SIZES[
                self._dataset_name]
            self.save_images(sess, 0)


@click.command()
@click.option('--log_dir',
              type=click.STRING,
              default = 'out/test',
              help='Where the data is logged to.')
@click.option('--config_filename',
              type=click.STRING,
              default='configs/exp_01_test.json',
              help='The name of the configuration file.')
@click.option('--checkpoint_dir',
              type=click.STRING,
              default='checkpoints/finetune',
              help='The name of the train/test split.')

def main(log_dir, config_filename, checkpoint_dir):
    """

    PARAMETERS:

    to_train: Specify whether it is training or testing. 1: training; 2:
     resuming from latest checkpoint; 0: testing.

    log_dir: The root dir to save checkpoints and imgs. The actual dir
            is the root dir appended by the folder with the name timestamp.
    
    config_filename: The configuration file.
    checkpoint_dir: The directory that saves the latest checkpoint. It
    only takes effect when to_train == 2.
    """

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    with open(config_filename) as config_file:
        config = json.load(config_file)

    max_step = int(config['max_step']) if 'max_step' in config else 5
    network_version = str(config['network_version'])
    dataset_name = str(config['dataset_name'])

    uwnet_model = UWNet(log_dir, max_step, network_version,
                              dataset_name, checkpoint_dir)

    uwnet_model.test()


if __name__ == '__main__':
    main()
