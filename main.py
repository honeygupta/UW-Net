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
import data_loader, losses, model
from skimage.transform import resize

slim = tf.contrib.slim

def load_data(dataset_name):
    list1 = []
    list2 = []
    csvfile = open(uwnet_datasets.PATH_TO_CSV[dataset_name], 'r')
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        list1.append(row[0])
        list2.append(row[1])
    return list1, list2

def random_crop(img1, img2, crop_size=(model.IMG_HEIGHT, model.IMG_WIDTH)):
    dx = crop_size[0]
    dy = crop_size[1]
    h = min(img1.shape[0] , img2.shape[0])
    w = min(img1.shape[1] , img2.shape[1])
    y = random.randint(0, w-dx-1)
    x = random.randint(0, h-dy-1)
    cropped_x = img1[x: x + dx, y : y + dy, :]
    cropped_y = img2[x: x + dx, y : y + dy, :]
    return cropped_x, cropped_y

class Data(object):
    def __init__(self, list1, list2, bs=1, shuffle=False):
        self.list1 = list1
        self.list2 = list2
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
        img2_batch = []
        name = []
        for i in range(start, end):
            im = np.load(self.list1[self.index_total[i]]).astype(np.float32)
            txt = self.list2[self.index_total[i]]
            name.append(txt[-32:])
            im1 = (im * 2.0) - 1.0            
            im2 = io.imread(self.list2[self.index_total[i]]).astype(np.float32) / 127.5
            im2 = im2 - 1.0
            im2 = resize(im2, [256,256], preserve_range = True)

            # For processing random crops of images; useful while training the network
            # img1, img2 = random_crop(im1, im2)
            img1 = im1
            img2 = im2

            img1_batch.append(img1)
            img2_batch.append(img2)

        return np.array(img1_batch), np.array(img2_batch),name

class UWNet:
    """The UWNet module."""

    def __init__(self, pool_size, lambda_a,
                 lambda_b, output_root_dir, to_restore,
                 base_lr, max_step, network_version,
                 dataset_name, checkpoint_dir, do_flipping):
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

        self._pool_size = pool_size
        self._lambda_a = lambda_a
        self._lambda_b = lambda_b
        self._output_dir = os.path.join(output_root_dir, current_time)
        self._images_dir = os.path.join(self._output_dir, 'imgs')
        self._num_imgs_to_save = 50
        self._to_restore = to_restore
        self._base_lr = base_lr
        self._max_step = max_step
        self._network_version = network_version
        self._dataset_name = dataset_name
        self._checkpoint_dir = checkpoint_dir
        self._do_flipping = do_flipping

        self.fake_images_A = np.zeros((self._pool_size, 1, model.IMG_HEIGHT, model.IMG_WIDTH,model.IMG_CHANNELS + 1))
        self.fake_images_B = np.zeros((self._pool_size, 1, model.IMG_HEIGHT, model.IMG_WIDTH,model.IMG_CHANNELS))

    def model_setup(self):
        self.input_a = tf.placeholder(
            tf.float32, [
                1,
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                model.IMG_CHANNELS + 1
            ], name="input_A")

        self.input_b = tf.placeholder(
            tf.float32, [
                1,
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                model.IMG_CHANNELS
            ], name="input_B")

        self.fake_pool_A = tf.placeholder(
            tf.float32, [
                None,
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                model.IMG_CHANNELS + 1
            ], name="fake_pool_A")

        self.fake_pool_B = tf.placeholder(
            tf.float32, [
                None,
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                model.IMG_CHANNELS
            ], name="fake_pool_B")

        self.global_step = slim.get_or_create_global_step()
        self.num_fake_inputs = 0
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name="lr")

        inputs = {
            'images_a': self.input_a,
            'images_b': self.input_b,
            'fake_pool_a': self.fake_pool_A,
            'fake_pool_b': self.fake_pool_B,
        }

        outputs = model.get_outputs(
            inputs, network=self._network_version)

        self.prob_real_a_is_real = outputs['prob_real_a_is_real']
        self.prob_real_b_is_real = outputs['prob_real_b_is_real']
        self.fake_images_a = outputs['fake_images_a']
        self.fake_images_b = outputs['fake_images_b']
        self.prob_fake_a_is_real = outputs['prob_fake_a_is_real']
        self.prob_fake_b_is_real = outputs['prob_fake_b_is_real']

        self.cycle_images_a = outputs['cycle_images_a']
        self.cycle_images_b = outputs['cycle_images_b']

        self.prob_fake_pool_a_is_real = outputs['prob_fake_pool_a_is_real']
        self.prob_fake_pool_b_is_real = outputs['prob_fake_pool_b_is_real']

    def compute_losses(self):
        """
        In this function we are defining the variables for loss calculations
        and training model.

        d_loss_A/d_loss_B -> loss for discriminator A/B
        g_loss_A/g_loss_B -> loss for generator A/B
        *_trainer -> Various trainer for above loss functions
        *_summ -> Summary variables for above loss functions
        """

        cycle_consistency_loss_a = self._lambda_a * losses.cycle_consistency_loss(real_images=self.input_a, generated_images=self.cycle_images_a)
        cycle_consistency_loss_b = self._lambda_b * losses.cycle_consistency_loss(real_images=self.input_b, generated_images=self.cycle_images_b)

        ssim_loss_A = 0.25 * (2 - (tf.reduce_mean(tf.image.ssim_multiscale(self.input_a[:,:,:,:-1], self.fake_images_b,2,power_factors=[0.0448, 0.2856, 0.3001]) + \
            tf.image.ssim_multiscale(self.fake_images_b, self.cycle_images_a[:,:,:,:-1],2,power_factors=[0.0448, 0.2856, 0.3001]))))

        ssim_loss_B = 0.25 * (2 - (tf.reduce_mean(tf.image.ssim_multiscale(self.input_b, self.fake_images_a[:,:,:,:-1],2,power_factors=[0.0448, 0.2856, 0.3001]) + \
            tf.image.ssim_multiscale(self.fake_images_a[:,:,:,:-1], self.cycle_images_b,2,power_factors=[0.0448, 0.2856, 0.3001]))))

        lsgan_loss_a = losses.lsgan_loss_generator(self.prob_fake_a_is_real)
        lsgan_loss_b = losses.lsgan_loss_generator(self.prob_fake_b_is_real)

        grad_loss_B = tf.reduce_mean(tf.image.image_gradients(tf.expand_dims(self.fake_images_a[:,:,:,-1], axis = 3))) 

        g_loss_A = cycle_consistency_loss_a + cycle_consistency_loss_b + lsgan_loss_b + ssim_loss_A + ssim_loss_B
        g_loss_B = cycle_consistency_loss_b + cycle_consistency_loss_a + lsgan_loss_a + ssim_loss_B + ssim_loss_A + grad_loss_B

        #g_loss_A = cycle_consistency_loss_a + cycle_consistency_loss_b + lsgan_loss_b 
        #g_loss_B = cycle_consistency_loss_b + cycle_consistency_loss_a + lsgan_loss_a

        d_loss_A = losses.lsgan_loss_discriminator(
            prob_real_is_real=self.prob_real_a_is_real,
            prob_fake_is_real=self.prob_fake_pool_a_is_real,
        )
        d_loss_B = losses.lsgan_loss_discriminator(
            prob_real_is_real=self.prob_real_b_is_real,
            prob_fake_is_real=self.prob_fake_pool_b_is_real,
        )

        optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9)

        self.model_vars = tf.trainable_variables()

        d_A_vars = [var for var in self.model_vars if 'd_A' in var.name]
        g_A_vars = [var for var in self.model_vars if 'g_A' in var.name]
        d_B_vars = [var for var in self.model_vars if 'd_B' in var.name]
        g_B_vars = [var for var in self.model_vars if 'g_B' in var.name]

        self.d_A_trainer = optimizer.minimize(d_loss_A, var_list=d_A_vars)
        self.d_B_trainer = optimizer.minimize(d_loss_B, var_list=d_B_vars)
        self.g_A_trainer = optimizer.minimize(g_loss_A, var_list=g_A_vars)
        self.g_B_trainer = optimizer.minimize(g_loss_B, var_list=g_B_vars)

        for var in self.model_vars:
            print(var.name)

        # Summary variables for tensorboard
        self.g_A_loss_summ = tf.summary.scalar("g_A_loss", g_loss_A)
        self.g_B_loss_summ = tf.summary.scalar("g_B_loss", g_loss_B)
        self.d_A_loss_summ = tf.summary.scalar("d_A_loss", d_loss_A)
        self.d_B_loss_summ = tf.summary.scalar("d_B_loss", d_loss_B)
        self.ssim_A_loss_summ = tf.summary.scalar("ssim_A_loss", ssim_loss_A)
        self.ssim_B_loss_summ = tf.summary.scalar("ssim_B_loss", ssim_loss_B)
        self.grad_B_loss_summ = tf.summary.scalar("gradient_loss", grad_loss_B)

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
        
        
        names = ['inputA_', 'inputA_depth_', 'inputB_', 'fakeB_depth_', 'cycA_', 'cycA_depth_' , 'cycB_']

        with open(os.path.join(
                self._output_dir, 'epoch_' + str(epoch) + '.html'), 'w') as v_html:
            for i in range(0, self._num_imgs_to_save):
                print("Saving image {}/{}".format(i, self._num_imgs_to_save))
                x1_t, x2_t, name1 = self.dataset.next_batch()
                count = 0
                fake_A_temp, fake_B_temp, cyc_A_temp, cyc_B_temp = sess.run([
                    self.fake_images_a,
                    self.fake_images_b,
                    self.cycle_images_a,
                    self.cycle_images_b], 
                    feed_dict={self.input_a: x1_t, self.input_b: x2_t})
                
                in1 = np.array(x1_t)
                inputa = in1[:,:,:,:-1]
                deptha = in1[:,:,:,-1]
                fakedepth = fake_A_temp[:,:,:,-1]
                cyca = cyc_A_temp
                cycrgb = cyca[:,:,:,:-1]
                cycdepth = cyca[:,:,:,-1]
                tensors = [inputa, deptha, x2_t, fakedepth, cycrgb, cycdepth, cyc_B_temp]

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

    def fake_image_pool(self, num_fakes, fake, fake_pool):
        """
        This function saves the generated image to corresponding
        pool of images.

        It keeps on feeling the pool till it is full and then randomly
        selects an already stored image and replace it with new one.
        """
        if num_fakes < self._pool_size:
            fake_pool[num_fakes] = fake
            return fake
        else:
            p = random.random()
            if p > 0.5:
                random_id = random.randint(0, self._pool_size - 1)
                temp = fake_pool[random_id]
                fake_pool[random_id] = fake
                return temp
            else:
                return fake

    def train(self):
        """Training Function."""
        # Load Dataset from the dataset folder
        
        list1, list2 = load_data(self._dataset_name)
        self.dataset = Data(list1, list2, shuffle=False)
        
        # Build the network
        self.model_setup()

        # Loss function calculations
        self.compute_losses()

        # Initializing the global variables
        init = (tf.global_variables_initializer(), tf.local_variables_initializer())
        saver = tf.train.Saver(max_to_keep = 10)

        max_images = uwnet_datasets.DATASET_TO_SIZES[self._dataset_name]

        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True

        with tf.Session(config=config) as sess:
            sess.run(init)

            # Restore the model to run the model from last checkpoint
            if self._to_restore:
                chkpt_fname = tf.train.latest_checkpoint(self._checkpoint_dir)
                saver.restore(sess, chkpt_fname)
                print("Restored the checkpoint: ", chkpt_fname)
            
            writer = tf.summary.FileWriter(self._output_dir)

            if not os.path.exists(self._output_dir):
                os.makedirs(self._output_dir)
            # self.global_step = tf.constant(0)

            # Training Loop
            for epoch in range(sess.run(self.global_step), self._max_step):
                print("In the epoch ", epoch)
                if epoch % 10 == 0:
                    saver.save(sess, os.path.join(
                        self._output_dir, "uwnet"), global_step=epoch)

                # Dealing with the learning rate as per the epoch number
                if epoch < 50:
                    curr_lr = self._base_lr
                elif epoch >= 50 and epoch < 100:
                    curr_lr = self._base_lr * 0.99
                elif epoch >=100 and epoch < 200:
                    curr_lr = self._base_lr - self._base_lr * (epoch - 100) / 100.0
                elif epoch >= 200 and epoch < 300:
                    curr_lr = self._base_lr * 0.01
                else:
                    curr_lr = self._base_lr * 0.001

                

                self.save_images(sess, epoch)

                for i in range(0, max_images):
                    print("Processing batch {}/{}".format(i, max_images))

                    x1_t, x2_t,_ = self.dataset.next_batch()

                    # Optimizing the G_A network
                    _, fake_B_temp, summary_str1, summary_str2 = sess.run(
                        [self.g_A_trainer,
                         self.fake_images_b,
                         self.g_A_loss_summ,
                         self.ssim_A_loss_summ],
                        feed_dict={self.input_a: x1_t, self.input_b:x2_t, self.learning_rate: curr_lr})
                    writer.add_summary(summary_str1, epoch * max_images + i)
                    writer.add_summary(summary_str2, epoch * max_images + i)

                    fake_B_temp1 = self.fake_image_pool(
                        self.num_fake_inputs, fake_B_temp, self.fake_images_B)

                    # Optimizing the D_B network
                    _, summary_str = sess.run(
                        [self.d_B_trainer, self.d_B_loss_summ],
                        feed_dict={
                            self.input_a: x1_t,
                            self.input_b: x2_t,
                            self.learning_rate: curr_lr,
                            self.fake_pool_B: fake_B_temp1
                        }
                    )
                    writer.add_summary(summary_str, epoch * max_images + i)

                    # Optimizing the G_B network
                    _, fake_A_temp, summary_str1, summary_str2, summary_str3 = sess.run(
                        [self.g_B_trainer,
                         self.fake_images_a,
                         self.g_B_loss_summ,
                         self.ssim_B_loss_summ,
                         self.grad_B_loss_summ],
                        feed_dict={
                            self.input_a: x1_t,
                            self.input_b: x2_t,
                            self.learning_rate: curr_lr
                        }
                    )
                    writer.add_summary(summary_str1, epoch * max_images + i)
                    writer.add_summary(summary_str2, epoch * max_images + i)
                    writer.add_summary(summary_str3, epoch * max_images + i)

                    fake_A_temp1 = self.fake_image_pool(
                        self.num_fake_inputs, fake_A_temp, self.fake_images_A)

                    # Optimizing the D_A network
                    _, summary_str = sess.run(
                        [self.d_A_trainer, self.d_A_loss_summ],
                        feed_dict={
                            self.input_a: x1_t,
                            self.input_b: x2_t,
                            self.learning_rate: curr_lr,
                            self.fake_pool_A: fake_A_temp1
                        }
                    )
                    writer.add_summary(summary_str, epoch * max_images + i)

                    writer.flush()
                    self.num_fake_inputs += 1

                self.global_step = epoch + 1

            writer.add_graph(sess.graph)

    def test(self):
        """Test Function."""
        print("Testing the results")

        list1, list2 = load_data(self._dataset_name)
        self.dataset = Data(list1, list2, shuffle=True)

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
@click.option('--to_train',
              type=click.INT,
              default=0,
              help='Whether it is train or false.')
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

def main(to_train, log_dir, config_filename, checkpoint_dir):
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

    lambda_a = float(config['_LAMBDA_A']) if '_LAMBDA_A' in config else 10.0
    lambda_b = float(config['_LAMBDA_B']) if '_LAMBDA_B' in config else 10.0
    pool_size = int(config['pool_size']) if 'pool_size' in config else 50

    to_restore = (to_train == 2)
    base_lr = float(config['base_lr']) if 'base_lr' in config else 0.0002
    max_step = int(config['max_step']) if 'max_step' in config else 5
    network_version = str(config['network_version'])
    dataset_name = str(config['dataset_name'])
    do_flipping = bool(config['do_flipping'])

    uwnet_model = UWNet(pool_size, lambda_a, lambda_b, log_dir,
                              to_restore, base_lr, max_step, network_version,
                              dataset_name, checkpoint_dir, do_flipping)

    if to_train > 0:
        uwnet_model.train()
    else:
        uwnet_model.test()


if __name__ == '__main__':
    main()
