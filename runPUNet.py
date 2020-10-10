#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright:    Zhipeng Wu
Filename:     runPuNet.py
Description:

@author:      wuzhipeng
@email:       763008300@qq.com
@website:     https://wuzhipeng.cn/
@create on:   10/1/2020 7:32 PM
@software:    PyCharm
"""

import tensorflow as tf
import os
from glob import glob
import time
import numpy as np
from tensorflow.contrib import slim
import argparse
import tifffile
from matplotlib import image, pyplot

def imread(filename):
    if filename.endswith('.wzp'):
        return np.fromfile(filename, dtype=np.float32)
    elif filename.endswith('.tif'):
        return tifffile.imread(filename)
    else:
        print('Unsupported file type: %s' % filename)
        print('Only supports binary files (*.wzp, float32) and Tiff files (*.tif, float32)')

def imwrite(img,filename):
    if filename.endswith('.wzp'):
        img.tofile(filename)
    elif filename.endswith('.tif'):
        tifffile.imwrite(filename,img)
    elif filename.endswith('.png') or filename.endswith('.jpg'):
        image.imsave(filename, img, pyplot.jet())
    else:
        print('Unsupported file type: %s' % filename)
        print('Only supports binary files (*.wzp, float32), Tiff files (*.tif, float32) and JPEG files (*.png, *.jpg)')

def dilated_conv2D_layer(inputs,num_outputs,kernel_size,rate,padding,scope,use_bias,weights_regularizer):
    with  tf.variable_scope(name_or_scope=scope):
        in_channels = inputs.get_shape().as_list()[3]
        kernel=[kernel_size,kernel_size,in_channels,num_outputs]
        filter_weight = slim.variable(name='weights',
                                      shape=kernel,
                                      initializer=tf.truncated_normal_initializer(stddev=0.1),
                                      regularizer=weights_regularizer)

        inputs = tf.nn.atrous_conv2d(inputs, filter_weight, rate=rate, padding=padding)  # + bias
        if use_bias:
            bias = tf.Variable(tf.constant(0.01, shape=[num_outputs]))
            inputs = inputs+bias
        return inputs

def PUNet(input, is_training=True, output_channels=3):
    with tf.variable_scope('block1'):
        oSum = tf.layers.conv2d(input, 64, 3, padding='same', activation=tf.nn.relu)

    for layers in range(1, 8 + 1):
        with tf.variable_scope('block1%d' % layers):
            out1 = tf.layers.conv2d(oSum, 64, 3, padding='same', name='conv1%d' % layers, use_bias=False)
            out1 = tf.nn.relu(tf.layers.batch_normalization(out1, training=is_training))

        with tf.variable_scope('block2%d' % layers):
            out2 = dilated_conv2D_layer(oSum, 64, 3, 2, padding='SAME', scope='conv2%d' % layers,use_bias=False,
                                          weights_regularizer=slim.l2_regularizer(scale=0.01))
            out2 = tf.nn.relu(tf.layers.batch_normalization(out2, training=is_training))

        with tf.variable_scope('block3%d' % layers):
            out3 = dilated_conv2D_layer(oSum, 64, 3, 3, padding='SAME', scope='conv3%d' % layers,use_bias=False,
                                          weights_regularizer=slim.l2_regularizer(scale=0.01))
            out3 = tf.nn.relu(tf.layers.batch_normalization(out3, training=is_training))

        with tf.variable_scope('blockS%d' % layers):
            oSumTmp = tf.concat([out1,out2,out3],3)
            oSumTmp = tf.layers.conv2d(oSumTmp, 64, 1, padding='same', name='convS%d' % layers, use_bias=False)
            oSumTmp = oSum + oSumTmp
            oSum = tf.nn.relu(tf.layers.batch_normalization(oSumTmp, training=is_training))

    for layers in range(41, 50 + 1):
        with tf.variable_scope('block%d' % layers):
            oSum = oSum+tf.layers.conv2d(oSum, 64, 3, padding='same', name='conv%d' % layers, use_bias=False)
            oSum = tf.nn.relu(tf.layers.batch_normalization(oSum, training=is_training))

    with tf.variable_scope('block51'):
        output = tf.layers.conv2d(oSum, output_channels, 3, padding='same', use_bias=False)
    return output

def load(sess, checkpoint_dir):
    print("[*] Reading checkpoint...")
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        full_path = tf.train.latest_checkpoint(checkpoint_dir)
        saver.restore(sess, full_path)
        return True
    else:
        return False

def unwrap(sess, interfFolder, unwFolder, ckpt_dir,outputPng, rows, cols):
    assert os.path.isdir(interfFolder), 'Path does not exist: '+interfFolder
    interfFile = glob(interfFolder+'/*[.wzp,.tif]')
    interfFile = sorted(interfFile)

    if not os.path.exists(unwFolder):
        os.mkdir(unwFolder)

    X = tf.placeholder(tf.float32, [None, None, None, 1])
    is_training = tf.placeholder(tf.bool, name='is_training')
    Y = PUNet(X, is_training=is_training, output_channels=1)

    # init variables
    tf.global_variables_initializer().run(session=sess)
    assert len(interfFile) != 0, 'No interferogram found!'
    load_model_status = load(sess, ckpt_dir)
    assert load_model_status == True, '[!] Load weights FAILED...'
    print(" [*] Load weights SUCCESS...")
    
    start = time.time()
    for i in range(len(interfFile)):
        interf = imread(interfFile[i])
        interf = interf.reshape(1,rows, cols,1)

        estimatedConstant = np.mean(np.abs(interf)) * np.sign(np.mean(interf))
        interf = np.angle(np.exp(1j * (interf - estimatedConstant)))
        unwrappedPhase = sess.run([Y], feed_dict={X: interf, is_training: False})                      
        unwrappedPhase = unwrappedPhase + estimatedConstant

        (filepath, tempfilename) = os.path.split(interfFile[i])

        unwrappedPhase = np.asarray(unwrappedPhase)[0,0,:,:,0]
        imwrite(unwrappedPhase, os.path.join(unwFolder,tempfilename))

        if outputPng:
            imwrite(unwrappedPhase, os.path.join(unwFolder, tempfilename+'.png'))

        print('Over '+tempfilename)

    end = time.time()
    print("Elapsed time: %.2f seconds" % (end - start))
    print("Average time: %.2f seconds" % ((end - start)/len(interfFile)))

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Run PUNet for phase unwrapping.")
    parser.add_argument('--input', type=str, default='data/interf',
                        help='The path where the interferograms is located (default: data/interf)')
    parser.add_argument('--output', type=str, default='data/unwrapped',
                        help='Output folder for unwrapped phase (*.wzp)')
    parser.add_argument('--outputPng', type=int, default=1,
                        help='Output the corresponding pseudo-color image (default: 1)')
    parser.add_argument('--rows', type=int, default=180,
                        help='rows of data (default: 180)')
    parser.add_argument('--cols', type=int, default=180,
                        help='cols of data (default: 180)')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoint',
                        help='checkpoint folder (default: ./checkpoint)')
    args = parser.parse_args()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    unwrap(sess, args.input, args.output, args.ckpt_dir, args.outputPng, args.rows, args.cols)