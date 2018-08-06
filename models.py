# -*- coding: utf-8 -*-
# !/usr/bin/env python

import os

import tensorflow as tf

from data_load import get_batch_queue, load_vocab, load_data
from hparams import Default as hp_default
from modules import prenet, cbhg, gru, normalize
import hparams as hp


class Model:
    def __init__(self, mode=None, batch_size=hp_default.batch_size, queue=True):
        self.mode = mode
        self.batch_size = batch_size
        self.queue = queue
        self.is_training = self.get_is_training(mode)

        # Input Generator
        self.x_mfcc = tf.placeholder(tf.float32, shape=(batch_size, None, hp_default.n_mfcc))
        self.y_ppgs = tf.placeholder(tf.int32, shape=(batch_size, None,))
        self.y_spec = tf.placeholder(tf.float32, shape=(batch_size, None, 1 + hp_default.n_fft // 2))
        self.y_mel = tf.placeholder(tf.float32, shape=(batch_size, None, hp_default.n_mels))

        wav_files = load_data(mode=mode)
        self.num_batch = len(wav_files) // batch_size

        self.z = tf.placeholder(tf.float32, shape=(batch_size, None, hp.Train2.noise_depth))

        # Network G 각 네트워크별 스코프 지정
        self.net_G = tf.make_template('net', self._netG)
        # G output
        self.ppgs, self.pred_ppg, self.logits_ppg, self.pred_spec, self.pred_mel = self.net_G(self.x_mfcc, self.z)

        # Network D 각 네트워크별 스코프 지정
        self.net_D = tf.make_template('net_D', self._netD)
        # real_D
        real_input = tf.concat([self.y_spec, self.ppgs], 2)
        self.real_d_logit = self.net_D(real_input)

        # fake_D
        fake_input = tf.concat([self.pred_spec, self.ppgs], 2)
        self.fake_d_logit = self.net_D(fake_input, True)

        # Train Variables 추
        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'net_D/netD' in var.name]
        self.g_vars = [var for var in t_vars if 'net/netG' in var.name]

        # todo 여기서 바로 loss 를 만들고 밑에서는 단순히 리턴만하면 어떨까? => 잘되는 거 같음
        #  net_G l2 loss - 일단 spectrogram loss 만
        loss_spec = tf.reduce_mean(tf.squared_difference(self.pred_spec, self.y_spec))
        # loss_mel = tf.reduce_mean(tf.squared_difference(self.pred_mel, self.y_mel))
        self.net_G_loss = loss_spec

        # adv loss
        # self.D_adv_loss = -tf.reduce_mean(tf.log(self.real_d_logit) + tf.log(1 - self.fake_d_logit))
        # self.G_adv_loss = -tf.reduce_mean(tf.log(self.fake_d_logit))
        # cross entropy로 변경. cross entropy를 쓰는 이유는 기존의 loss function은 D와 G에 대한 loss를 줄이는 방향으로 갈 뿐이지
        # 우리가 원하는 D의 출력이 G나 Real Data에 대한 결과인 0 혹은 1이 나오도록 의도하고 있지 않기 때문에, 명확하게 방향을 정해주는 역할.
        self.D_adv_loss = -tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.real_d_logit, labels=tf.ones_like(self.real_d_logit))
                                          + tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_d_logit, labels=tf.zeros_like(self.fake_d_logit)))
        self.G_adv_loss = -tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_d_logit,labels=tf.ones_like(self.fake_d_logit)))
        # todo : 기본 loss로 구현 했는데 cross entropy로 loss 구현 하기도 하는 거 같음 but 아직 cross entropy에 대하여 이해가 필요
        # todo 간단하게 adv loss는 만들었으니 각각 loss리턴 함수 만들고 train2 에서 optimizer 만들어서 구조 만들면 될 듯
        # one-sided label smoothing을 고려 - cross entropy로 바꿀 때 고려 해볼 것 G grad 폭발을 조금 막아준다고 함

        #Optimizer 추가 변수리스트 적용 완료
        self.G_train_step = tf.train.AdamOptimizer().minimize((self.net_G_loss + self.G_adv_loss), var_list=self.g_vars)
        self.D_train_step = tf.train.AdamOptimizer().minimize(self.D_adv_loss, var_list=self.d_vars)

    def __call__(self):
        return self.pred_spec


    def get_is_training(self, mode):
        if mode in ('train1', 'train2'):
            is_training = True
        else:
            is_training = False
        return is_training

    def _net1(self, x_mfcc):
        with tf.variable_scope('net1'):
            # Load vocabulary
            phn2idx, idx2phn = load_vocab()

            # Pre-net
            prenet_out = prenet(x_mfcc,
                                num_units=[hp.Train1.hidden_units, hp.Train1.hidden_units // 2],
                                dropout_rate=hp.Train1.dropout_rate,
                                is_training=self.is_training)  # (N, T, E/2)

            # CBHG
            out = cbhg(prenet_out, hp.Train1.num_banks, hp.Train1.hidden_units // 2, hp.Train1.num_highway_blocks,
                       hp.Train1.norm_type, self.is_training)

            # Final linear projection
            logits = tf.layers.dense(out, len(phn2idx))  # (N, T, V)
            ppgs = tf.nn.softmax(logits / hp.Train1.t)  # (N, T, V)
            preds = tf.to_int32(tf.arg_max(logits, dimension=-1))  # (N, T)

        return ppgs, preds, logits

    def loss_net1(self):
        istarget = tf.sign(tf.abs(tf.reduce_sum(self.x_mfcc, -1)))  # indicator: (N, T)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_ppg / hp.Train1.t, labels=self.y_ppgs)
        loss *= istarget
        loss = tf.reduce_mean(loss)
        return loss

    def acc_net1(self):
        istarget = tf.sign(tf.abs(tf.reduce_sum(self.x_mfcc, -1)))  # indicator: (N, T)
        num_hits = tf.reduce_sum(tf.to_float(tf.equal(self.pred_ppg, self.y_ppgs)) * istarget)
        num_targets = tf.reduce_sum(istarget)
        acc = num_hits / num_targets
        return acc

    def _netG(self, x_mfcc, z):
        # PPGs from net1
        ppgs, preds_ppg, logits_ppg = self._net1(x_mfcc)

        with tf.variable_scope('netG'):
            noise_ppgs = tf.concat([ppgs, z], 2)

            # Pre-net
            prenet_out = prenet(noise_ppgs,
                                num_units=[hp.Train2.hidden_units, hp.Train2.hidden_units // 2],
                                dropout_rate=hp.Train2.dropout_rate,
                                is_training=self.is_training)  # (N, T, E/2)

            # CBHG1: mel-scale
            pred_mel = cbhg(prenet_out, hp.Train2.num_banks, hp.Train2.hidden_units // 2, hp.Train2.num_highway_blocks,
                            hp.Train2.norm_type, self.is_training, scope="cbhg1")
            pred_mel = tf.layers.dense(pred_mel, self.y_mel.shape[-1])  # log magnitude: (N, T, n_mels)

            # CBHG2: linear-scale
            pred_spec = tf.layers.dense(pred_mel, hp.Train2.hidden_units // 2)  # log magnitude: (N, T, n_mels)
            pred_spec = cbhg(pred_spec, hp.Train2.num_banks, hp.Train2.hidden_units // 2, hp.Train2.num_highway_blocks,
                             hp.Train2.norm_type, self.is_training, scope="cbhg2")
            pred_spec = tf.layers.dense(pred_spec, self.y_spec.shape[-1])  # log magnitude: (N, T, 1+hp.n_fft//2)

        return ppgs, preds_ppg, logits_ppg, pred_spec, pred_mel

    def total_loss(self):
        return self.loss_adv_g() + self.loss_adv_d()

    def loss_adv_d(self):
        return self.D_adv_loss

    def loss_adv_g(self):
        return self.G_adv_loss + self.net_G_loss

    def _netD(self, d_input, reuse=False):
        with tf.variable_scope('netD', reuse=reuse):
            # Pre-net
            prenet_out = prenet(d_input,
                                num_units=[hp.Train1.hidden_units, hp.Train1.hidden_units // 2],
                                dropout_rate=hp.Train1.dropout_rate,
                                is_training=self.is_training)  # (N, T, E/2)

            # CBHG
            out = cbhg(prenet_out, hp.Train1.num_banks, hp.Train1.hidden_units // 2, hp.Train1.num_highway_blocks,
                       hp.Train1.norm_type, self.is_training, scope="cbhg1")

            # CBHG2
            out = tf.layers.dense(out, hp.Train1.hidden_units // 2)
            out = cbhg(out, hp.Train1.num_banks, hp.Train1.hidden_units // 2, hp.Train1.num_highway_blocks,
                       hp.Train1.norm_type, self.is_training, scope="cbhg2")

            # recurrent
            out = gru(out, hp.Train1.hidden_units // 2, True)

            ## Disciriminator output
            out = out[:, -1, :]
            # memory = tf.normalize(memory, reuse=reuse)
            W_dis = tf.get_variable("weights", shape=[hp.Train1.hidden_units, 1])
            b_dis = tf.get_variable("bias", shape=[1])
            out = tf.sigmoid(tf.matmul(out, W_dis) + b_dis)

            # # Final linear projection
            # logits = tf.layers.dense(out, len(phn2idx))  # (N, T, V)
            # ppgs = tf.nn.softmax(logits / hp.Train1.t)  # (N, T, V)
            # preds = tf.to_int32(tf.arg_max(logits, dimension=-1))  # (N, T)

            return out

    def loss_adv(self):
        # G = self._net2()
        return self.G_adv_loss + self.D_adv_loss
        # pass

    @staticmethod
    def load(sess, mode, logdir, logdir2=None):

        def print_model_loaded(mode, logdir):
            model_name = Model.get_model_name(logdir)
            print('Model loaded. mode: {}, model_name: {}'.format(mode, model_name))

        if mode in ['train1', 'test1']:
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'net/net1')
            if Model._load_variables(sess, logdir, var_list=var_list):
                print_model_loaded(mode, logdir)

        elif mode == 'train2':
            var_list1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'net/net1')
            if Model._load_variables(sess, logdir, var_list=var_list1):
                print_model_loaded(mode, logdir)

            var_list2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'net_D/netD')
            if Model._load_variables(sess, logdir2, var_list=var_list2):
                print_model_loaded(mode, logdir2)

            var_list3 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'net/netG')
            if Model._load_variables(sess, logdir2, var_list=var_list3):
                print_model_loaded(mode, logdir2)

        elif mode in ['test2', 'convert']:
            if Model._load_variables(sess, logdir, var_list=None):  # Load all variables
                print_model_loaded(mode, logdir)

    @staticmethod
    def _load_variables(sess, logdir, var_list):
        ckpt = tf.train.latest_checkpoint(logdir)
        if ckpt:
            tf.train.Saver(var_list=var_list).restore(sess, ckpt)
            return True
        else:
            return False

    @staticmethod
    def get_model_name(logdir):
        path = '{}/checkpoint'.format(logdir)
        if os.path.exists(path):
            ckpt_path = open(path, 'r').read().split('"')[1]
            _, model_name = os.path.split(ckpt_path)
        else:
            model_name = None
        return model_name

    @staticmethod
    def get_global_step(logdir):
        model_name = Model.get_model_name(logdir)
        if model_name:
            gs = int(model_name.split('_')[3])
        else:
            gs = 0
        return gs

    @staticmethod
    def all_model_names(logdir):
        import glob, os
        path = '{}/*.meta'.format(logdir)
        model_names = map(lambda f: os.path.basename(f).replace('.meta', ''), glob.glob(path))
        return model_names
