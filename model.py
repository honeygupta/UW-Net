"""Code for constructing the model and get the outputs from the model.
Author: Honey Gupta (hn.gpt1@gmail.com)
"""

import utils
import layers
import tensorflow as tf
import tensorflow.contrib.slim as slim

# The number of samples per batch.
BATCH_SIZE = 1

# The height of each image.
IMG_HEIGHT = 256

# The width of each image.
IMG_WIDTH = 256

# The number of color channels per image.
IMG_CHANNELS = 3

POOL_SIZE = 50
ngf = 32
ndf = 64


def get_outputs(inputs, network="tensorflow"):
    images_a = inputs['images_a']
    images_b = inputs['images_b']

    fake_pool_a = inputs['fake_pool_a']
    fake_pool_b = inputs['fake_pool_b']

    with tf.variable_scope("Model") as scope:

        prob_real_a_is_real = discriminator_A(images_a, name="d_A")
        prob_real_b_is_real = discriminator_B(images_b, name="d_B")

        fake_images_b = denseNet(images_a, gen_type = 'A', name="g_A")
        fake_images_a = denseNet(images_b, gen_type = 'B', name="g_B")

        scope.reuse_variables()

        prob_fake_a_is_real = discriminator_A(fake_images_a, name="d_A")
        prob_fake_b_is_real = discriminator_B(fake_images_b, name="d_B")

        cycle_images_a = denseNet(fake_images_b, gen_type = 'B', name="g_B")
        cycle_images_b = denseNet(fake_images_a, gen_type = 'A', name="g_A")

        scope.reuse_variables()

        prob_fake_pool_a_is_real = discriminator_A(fake_pool_a, name="d_A")
        prob_fake_pool_b_is_real = discriminator_B(fake_pool_b, name="d_B")

    return {
        'prob_real_a_is_real': prob_real_a_is_real,
        'prob_real_b_is_real': prob_real_b_is_real,
        'prob_fake_a_is_real': prob_fake_a_is_real,
        'prob_fake_b_is_real': prob_fake_b_is_real,
        'prob_fake_pool_a_is_real': prob_fake_pool_a_is_real,
        'prob_fake_pool_b_is_real': prob_fake_pool_b_is_real,
        'cycle_images_a': cycle_images_a,
        'cycle_images_b': cycle_images_b,
        'fake_images_a': fake_images_a,
        'fake_images_b': fake_images_b,
    }

def discriminator_A(inputdisc, name="discriminator"):
    with tf.variable_scope(name):
        f = 4

        patch_input = tf.random_crop(inputdisc, [1, 70, 70, 4])
        o_c1 = layers.general_conv2d(patch_input, ndf, f, f, 2, 2,
                                     0.02, "SAME", "c1", do_norm="False",
                                     relufactor=0.2)
        o_c2 = layers.general_conv2d(o_c1, ndf * 2, f, f, 2, 2,
                                     0.02, "SAME", "c2", relufactor=0.2)
        o_c3 = layers.general_conv2d(o_c2, ndf * 4, f, f, 2, 2,
                                     0.02, "SAME", "c3", relufactor=0.2)
        o_c4 = layers.general_conv2d(o_c3, ndf * 8, f, f, 2, 2,
                                     0.02, "SAME", "c4", relufactor=0.2)
        o_c5 = layers.general_conv2d(
            o_c4, 1, f, f, 1, 1, 0.02, "SAME", "c5", do_norm=False,
            do_relu=False)

        return o_c5

def discriminator_B(inputdisc, name="discriminator"):
    with tf.variable_scope(name):
        f = 4

        patch_input = tf.random_crop(inputdisc, [1, 70, 70, 3])
        o_c1 = layers.general_conv2d(patch_input, ndf, f, f, 2, 2,
                                     0.02, "SAME", "c1", do_norm="False",
                                     relufactor=0.2)
        o_c2 = layers.general_conv2d(o_c1, ndf * 2, f, f, 2, 2,
                                     0.02, "SAME", "c2", relufactor=0.2)
        o_c3 = layers.general_conv2d(o_c2, ndf * 4, f, f, 2, 2,
                                     0.02, "SAME", "c3", relufactor=0.2)
        o_c4 = layers.general_conv2d(o_c3, ndf * 8, f, f, 2, 2,
                                     0.02, "SAME", "c4", relufactor=0.2)
        o_c5 = layers.general_conv2d(
            o_c4, 1, f, f, 1, 1, 0.02, "SAME", "c5", do_norm=False,
            do_relu=False)

        return o_c5
               
def TransitionDown(inputs, n_filters, scope=None):
  with tf.name_scope(scope) as sc:
    l = preact_conv(inputs, n_filters, kernel_size=[1, 1])
    l = slim.pool(l, [2, 2], stride=[2, 2], pooling_type='MAX')
    return l

def preact_conv(inputs, n_filters, kernel_size=[3, 3]):
    preact = tf.nn.relu(inputs)
    conv = slim.conv2d(preact, n_filters, kernel_size, activation_fn=None, normalizer_fn=None,weights_initializer=tf.truncated_normal_initializer(
                stddev=0.02), biases_initializer=tf.constant_initializer(0.0))
    return conv
    
def TransitionUp(block_to_upsample, skip_connection, n_filters_keep, scope=None):
  with tf.name_scope(scope) as sc:
    l = slim.conv2d_transpose(block_to_upsample, n_filters_keep, kernel_size=[3, 3], stride=[2, 2], activation_fn=None,weights_initializer=tf.truncated_normal_initializer(stddev=0.02),biases_initializer=tf.constant_initializer(0.0))
    l = tf.concat([l, skip_connection], axis=-1)
    return l
    

def DenseBlock(stack, n_layers, growth_rate, scope=None):
  with tf.name_scope(scope) as sc:
    new_features = []
    for j in range(n_layers):
      layer = preact_conv(stack, growth_rate)
      new_features.append(layer)
      stack = tf.concat([stack, layer], axis=-1)
    new_features = tf.concat(new_features, axis=-1)
    return stack, new_features
    
         
def denseNet(inputgen, gen_type, name="generator", preset_model = 'FC-DenseNet56', n_filters_first_conv=32, n_pool=1, growth_rate=2, n_layers_per_block=3):
    if preset_model == 'FC-DenseNet56':
          n_pool=5
          growth_rate=12
          n_layers_per_block=4
    elif preset_model == 'FC-DenseNet67':
          n_pool=5
          growth_rate=16
          n_layers_per_block=5
    elif preset_model == 'FC-DenseNet103':
        n_pool=5
        growth_rate=16
        n_layers_per_block=[4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4]
    else:
        raise ValueError("Unsupported FC-DenseNet model '%s'. This function only supports FC-DenseNet56, FC-DenseNet67, and FC-DenseNet103" % (preset_model)) 
     
     
    with tf.variable_scope(name):
        if type(n_layers_per_block) == list:
            assert (len(n_layers_per_block) == 2 * n_pool + 1)
        elif type(n_layers_per_block) == int:
            n_layers_per_block = [n_layers_per_block] * (2 * n_pool + 1)
        else:
            raise ValueError

        #stack = slim.conv2d(inputgen, n_filters_first_conv, [7, 7], scope='first_conv', activation_fn=None)
        pad_input = tf.pad(inputgen, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        stack = layers.general_conv2d(inputgen, ngf, 7, 7, 1, 1, 0.02, "SAME", "c1")
        #stack = layers.general_conv2d(stack, ngf * 2, 3, 3, 1, 1, 0.02, "SAME", "c2")
        n_filters = ngf
        skip_connection_list = []

        for i in range(n_pool):
            stack, _ = DenseBlock(stack, n_layers_per_block[i], growth_rate, scope='denseblock%d' % (i+1))
            n_filters += growth_rate * n_layers_per_block[i]
            skip_connection_list.append(stack)
            stack = TransitionDown(stack, n_filters, scope='transitiondown%d'%(i+1))

        skip_connection_list = skip_connection_list[::-1]

        stack, block_to_upsample = DenseBlock(stack, n_layers_per_block[n_pool], growth_rate, scope='denseblock%d' % (n_pool + 1))

        for i in range(n_pool):
            n_filters_keep = growth_rate * n_layers_per_block[n_pool + i]
            stack = TransitionUp(block_to_upsample, skip_connection_list[i], n_filters_keep, scope='transitionup%d' % (n_pool + i + 1))
            stack, block_to_upsample = DenseBlock(stack, n_layers_per_block[n_pool + i + 1], growth_rate, scope='denseblock%d' % (n_pool + i + 2))
        
        o_c3 = layers.general_conv2d(stack, ngf * 4, 1, 1, 1, 1, 0.02, "SAME", "c3")
        o_r1 = tf.nn.relu(build_resnet_block(o_c3, ngf * 4, "r1", padding = "REFLECT"))
        o_r2 = build_resnet_block(o_r1, ngf * 4, "r2", padding = "REFLECT")
        o_r3 = build_resnet_block(o_r2, ngf * 4, "r3", padding = "REFLECT")
        
        if gen_type == 'A':
            out_layers = IMG_CHANNELS
        elif gen_type == 'B':
            out_layers = IMG_CHANNELS + 1
        
        net = tf.nn.tanh(layers.general_conv2d(o_r3, out_layers, 3, 3, 1, 1, 0.02, "SAME", "c6", do_norm=False, do_relu=False))
    return net
        
def build_resnet_block(inputres, dim, name="resnet", padding="REFLECT"):
    """build a single block of resnet.

    :param inputres: inputres
    :param dim: dim
    :param name: name
    :param padding: for tensorflow version use REFLECT; for pytorch version use
     CONSTANT
    :return: a single block of resnet.
    """
    with tf.variable_scope(name):
        out_res = tf.pad(inputres, [[0, 0], [1, 1], [
            1, 1], [0, 0]], padding)
        out_res = layers.general_conv2d(
            out_res, dim, 3, 3, 1, 1, 0.02, "VALID", "c1")
        out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)
        out_res = layers.general_conv2d(
            out_res, dim, 3, 3, 1, 1, 0.02, "VALID", "c2", do_relu=False)

        return tf.nn.relu(out_res + inputres)
        

