# -*- coding: utf-8 -*-
# /usr/bin/python2

from __future__ import print_function

from hparams import logdir_path
import hparams as hp
import numpy as np
import tensorflow as tf
from models import Model
import convert, eval2
from data_load import get_batch
import argparse


def train(logdir1='logdir/default/train1', logdir2='logdir/default/train2', queue=False):
    model = Model(mode="train2", batch_size=hp.Train2.batch_size, queue=queue)
    saver = tf.train.Saver()
    # Loss
    loss_op = model.total_loss()

    # Training Scheme
    global_step = tf.Variable(0, name='global_step', trainable=False)

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_op_d = model.D_train_step
        train_op_g = model.G_train_step
    g_adv_loss = model.loss_adv_g()
    d_adv_loss = model.loss_adv_d()
    total_adv_loss = model.loss_adv()

    # Summary
    tf.summary.scalar('net2/train/g_adv_loss', model.G_adv_loss)
    tf.summary.scalar('net2/train/total_g', g_adv_loss)
    tf.summary.scalar('net2/train/d_adv_loss', d_adv_loss)
    tf.summary.scalar('net2/train/reconstruction', model.net_G_loss)
    summ_op = tf.summary.merge_all()

    session_conf = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            allow_growth=True,
        ),
    )

    # Training
    with tf.Session(config=session_conf) as sess:
        # Load trained model
        sess.run(tf.global_variables_initializer())
        model.load(sess, mode='train2', logdir=logdir1, logdir2=logdir2)

        writer = tf.summary.FileWriter(logdir2, sess.graph)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        gs = 0
        K = 3
        for epoch in range(1, hp.Train2.num_epochs + 1):
            for step in range(model.num_batch):
                mfcc, spec, mel = get_batch(model.mode, model.batch_size)
                z = np.random.normal(size=(model.batch_size, np.shape(mfcc)[1], hp.Train2.noise_depth))
		        # train Discriminator
                sess.run(train_op_d,
                         feed_dict={model.x_mfcc: mfcc, model.y_spec: spec, model.y_mel: mel, model.z: z})

		        # train Generator
                if step % K != 0:
                    sess.run(train_op_g,
                             feed_dict={model.x_mfcc: mfcc, model.y_spec: spec, model.y_mel: mel, model.z: z})

                # Write checkpoint files
                if (gs % 10) == 0:
                    summ = sess.run(summ_op, feed_dict={model.x_mfcc: mfcc, model.y_spec: spec, model.y_mel: mel, model.z: z})
                    writer.add_summary(summ, global_step=gs)

                print('epoch : ' + str(epoch) + ' step : ' + str(step))

                gs += 1

            if epoch % hp.Train2.save_per_epoch == 0:
                saver.save(sess,
                           '{}/epoch_{}_step_{}'.format(logdir2, epoch, gs))

                # Convert at every n epochs
                with tf.Graph().as_default():
                    convert.convert(logdir2, queue=False)


        writer.close()
        coord.request_stop()
        coord.join(threads)


def summaries(loss):
    tf.summary.scalar('net2/train/loss', loss)
    return tf.summary.merge_all()


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('case1', type=str, help='experiment case name of train1')
    parser.add_argument('case2', type=str, help='experiment case name of train2')
    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':
    args = get_arguments()
    case1, case2 = args.case1, args.case2
    logdir1 = '{}/{}/train1'.format(logdir_path, case1)
    logdir2 = '{}/{}/train2'.format(logdir_path, case2)
    train(logdir1=logdir1, logdir2=logdir2)
    print("Done")
