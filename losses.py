"""Code containing functions to calculate the loss
Author: Honey Gupta (hn.gpt1@gmail.com)
"""

import tensorflow as tf
import numpy as np

def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)


def tf_ssim(img1, img2, cs_map=False, mean_metric=True, size=13, sigma=2):
    window = _tf_fspecial_gauss(size, sigma) # window shape [size, size]
    K1 = 0.02
    K2 = 0.03
    L = 2  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1],padding='VALID')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='VALID') - mu1_mu2
    if cs_map:
        value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)),
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value


def tf_ms_ssim(img1, img2, mean_metric=True, level=5):
    weight = tf.constant([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=tf.float32)
    mssim = []
    mcs = []
    tf.squeeze(img1)
    tf.squeeze(img2)
    for l in range(level):
      img10, img11, img12 = tf.split(img1, num_or_size_splits=3, axis=3)
      img20, img21, img22 = tf.split(img2, num_or_size_splits=3, axis=3)
      ssim_map1, cs_map1 = tf_ssim(img10, img20, cs_map=True, mean_metric=False)
      ssim_map2, cs_map2 = tf_ssim(img11, img21, cs_map=True, mean_metric=False)
      ssim_map3, cs_map3 = tf_ssim(img12, img22, cs_map=True, mean_metric=False)
      ssim_map = (ssim_map1+ssim_map2+ssim_map3)/3
      cs_map = (cs_map1+cs_map2+cs_map3)/3
      mssim.append(tf.reduce_mean(ssim_map))
      mcs.append(tf.reduce_mean(cs_map))
      filtered_im10 = tf.nn.avg_pool(img10, [1,2,2,1], [1,2,2,1], padding='SAME')
      filtered_im20 = tf.nn.avg_pool(img20, [1,2,2,1], [1,2,2,1], padding='SAME')
      filtered_im11 = tf.nn.avg_pool(img11, [1,2,2,1], [1,2,2,1], padding='SAME')
      filtered_im21 = tf.nn.avg_pool(img21, [1,2,2,1], [1,2,2,1], padding='SAME')
      filtered_im12 = tf.nn.avg_pool(img12, [1,2,2,1], [1,2,2,1], padding='SAME')
      filtered_im22 = tf.nn.avg_pool(img22, [1,2,2,1], [1,2,2,1], padding='SAME')
      img10 = filtered_im10
      img20 = filtered_im20
      img11 = filtered_im11
      img21 = filtered_im21
      img12 = filtered_im12
      img22 = filtered_im22

    # list to tensor of dim D+1
    mssim = tf.stack(mssim, axis=0)
    mcs = tf.stack(mcs, axis=0)

    value = (tf.reduce_prod(mcs[0:level-1]**weight[0:level-1])*
                            (mssim[level-1]**weight[level-1]))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value

def cycle_consistency_loss(real_images, generated_images):
    return tf.reduce_mean(tf.abs(real_images - generated_images))


def lsgan_loss_generator(prob_fake_is_real):
    return tf.reduce_mean(tf.squared_difference(prob_fake_is_real, 1))


def lsgan_loss_discriminator(prob_real_is_real, prob_fake_is_real):
    return (tf.reduce_mean(tf.squared_difference(prob_real_is_real, 1)) +
            tf.reduce_mean(tf.squared_difference(prob_fake_is_real, 0))) * 0.5
