# -*- coding: utf-8 -*-
import os

import tensorflow as tf

from data_load import get_batch_queue, load_vocab, load_data
from hparams import Default as hp_default
from modules import prenet, cbhg, gru, normalize, conv1d, conv1d_banks, highwaynet, attention_decoder
import hparams as hp

class Model:
    def __init__(self, mode=None, batch_size=hp_default.batch_size, queue=True):
        self.mode = mode
        self.batch_size = batch_size
        self.queue = queue
        self.is_training = self.get_is_training(mode)

        # Input Generator
        self.x_mfcc = tf.placeholder(tf.float32, shape=(batch_size, None, hp_default.n_mfcc), name='x_mfcc')
        self.y_ppgs = tf.placeholder(tf.int32, shape=(batch_size, None,), name='y_ppgs')
        self.y_spec = tf.placeholder(tf.float32, shape=(batch_size, None, 1 + hp_default.n_fft // 2), name='y_spec')
        self.y_mel = tf.placeholder(tf.float32, shape=(batch_size, None, hp_default.n_mels), name='y_mel')

        wav_files = load_data(mode=mode)
        self.num_batch = len(wav_files) // batch_size
        if mode in ['train1', 'test1']:
            self.net_1 = tf.make_template('net', self._net1) 
            self.ppgs, self.pred_ppg, self.logits_ppg = self.net_1(self.x_mfcc)
        elif mode in ['train2', 'test2'] :
            self.z = tf.placeholder(tf.float32, shape=(batch_size, None, hp.Train2.noise_depth))
            
            # Network G Scope
            self.net_G = tf.make_template('net', self._netG)
            # G output
            self.ppgs, self.pred_ppg, self.logits_ppg, self.pred_spec, self.pred_mel = self.net_G(self.x_mfcc, self.z)

            # Network D Scope
            self.net_D = tf.make_template('net_D', self._netD)
            # real_D
            real_input = tf.concat([self.y_spec, self.ppgs], 2)
            self.real_d_logit = self.net_D(real_input)

            # fake_D
            fake_input = tf.concat([self.pred_spec, self.ppgs], 2)
            self.fake_d_logit = self.net_D(fake_input, True)

            # Train Variables
            t_vars = tf.trainable_variables()

            self.d_vars = [var for var in t_vars if 'net_D/netD' in var.name]
            self.g_vars = [var for var in t_vars if 'net/netG' in var.name]

            loss_spec = tf.reduce_mean(tf.abs(self.y_spec - self.pred_spec))
            loss_mel = tf.reduce_mean(tf.abs(self.y_mel - self.pred_mel))
            self.net_G_loss = loss_spec + loss_mel

            # one-sided label smoothing
            smooth = tf.Variable(0.9, trainable=False)
            real_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.real_d_logit,
		                                                     labels=tf.multiply(tf.ones_like(self.real_d_logit), smooth)))
            tf.summary.scalar('net2/train/real_d', real_d)
            fake_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_d_logit,
		                                                     labels=tf.zeros_like(self.fake_d_logit)))
            tf.summary.scalar('net2/train/fake_d', fake_d)
            self.D_adv_loss = real_d + fake_d
            self.G_adv_loss = tf.reduce_mean(
		        tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_d_logit, labels=tf.ones_like(self.fake_d_logit)))

            # grad clip optimizer
            grad_d, var_d = zip(*tf.train.AdamOptimizer().compute_gradients(self.D_adv_loss, var_list=self.d_vars))
            grad_d_clipped, _ = tf.clip_by_global_norm(grad_d, 5.)
            grad_g, var_g = zip(
		        *tf.train.AdamOptimizer().compute_gradients(self.net_G_loss + self.G_adv_loss, var_list=self.g_vars))
            grad_g_clipped, _ = tf.clip_by_global_norm(grad_g, 5.)
            self.D_train_step = tf.train.AdamOptimizer().apply_gradients(zip(grad_d_clipped, var_d))
            self.G_train_step = tf.train.AdamOptimizer().apply_gradients(zip(grad_g_clipped, var_g))


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

    def decode1(self, decoder_inputs, memory, is_training=True, scope="decoder1", reuse=None):
        '''
        Args:
          decoder_inputs: A 3d tensor with shape of [N, T', C'], where C'=hp.n_mels*hp.r,
            dtype of float32. Shifted melspectrogram of sound files.
          memory: A 3d tensor with shape of [N, T, C], where C=hp.embed_size.
          is_training: Whether or not the layer is in training mode.
          scope: Optional scope for `variable_scope`
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.

        Returns
          Predicted melspectrogram tensor with shape of [N, T', C'].
        '''
        with tf.variable_scope(scope, reuse=reuse):
            # Decoder pre-net
            dec = prenet(decoder_inputs,
                         num_units=[hp.Train2.hidden_units, hp.Train2.hidden_units // 2],
                         dropout_rate=hp.Train2.dropout_rate,
                         is_training=self.is_training)
            # Attention RNN
            dec = attention_decoder(dec, memory, num_units=hp.Train2.hidden_units)  # (N, T', E)

            # Decoder RNNs
            # dec += gru(dec, hp.Train2.hidden_units, False, scope="decoder_gru1")  # (N, T', E)
            dec += gru(dec, hp.Train2.hidden_units, False, scope="decoder_gru1")  # (N, T', E)
            dec += gru(dec, hp.Train2.hidden_units, False, scope="decoder_gru2")  # (N, T', E)

            # Outputs => (N, T', hp.n_mels)
            out_dim = decoder_inputs.get_shape().as_list()[-1]
            outputs = tf.layers.dense(dec, out_dim)

        return outputs

    def decode2(self, inputs, is_training=True, scope="decoder2", reuse=None):
        '''
        Args:
          inputs: A 3d tensor with shape of [N, T', C'], where C'=hp.n_mels*hp.r,
            dtype of float32. Log magnitude spectrogram of sound files.
          is_training: Whether or not the layer is in training mode.
          scope: Optional scope for `variable_scope`
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.

        Returns
          Predicted magnitude spectrogram tensor with shape of [N, T', C''],
            where C'' = (1+hp.n_fft//2)*hp.r.
        '''
        with tf.variable_scope(scope, reuse=reuse):
            # Decoder pre-net
            prenet_out = prenet(inputs,
                                num_units=[hp.Train2.hidden_units, hp.Train2.hidden_units // 2],
                                dropout_rate=hp.Train2.dropout_rate,
                                is_training=self.is_training)

            # Decoder Post-processing net = CBHG
            dec = cbhg(prenet_out, hp.Train2.num_banks, hp.Train2.hidden_units // 2, hp.Train2.num_highway_blocks,
                                  hp.Train2.norm_type, self.is_training)

            # Outputs => (N, T', (1+hp.n_fft//2))
            out_dim = (1 + hp.Default.n_fft // 2)
            outputs = tf.layers.dense(dec, out_dim)

        return outputs

    def total_loss(self):
        return self.loss_adv_g() + self.loss_adv_d()

    def loss_adv(self):
        return self.G_adv_loss + self.D_adv_loss

    def loss_adv_d(self):
        return self.D_adv_loss

    def loss_adv_g(self):
        return self.G_adv_loss + self.net_G_loss

    def _netG(self, x_mfcc, z):
        # PPGs from net1
        ppgs, preds_ppg, logits_ppg = self._net1(x_mfcc)

        with tf.variable_scope('netG'):
            noise_ppgs = tf.concat([ppgs, z], 2)

            # Pre-net
            prenet_out = prenet(noise_ppgs,
                                num_units=[hp.Train2.hidden_units, hp_default.n_mels],
                                dropout_rate=hp.Train2.dropout_rate,
                                is_training=self.is_training)  # (N, T, E/2)

            reuse = None
            pred_mel = self.decode1(prenet_out,
                                    ppgs,
                                     is_training=self.is_training, reuse=reuse)

            pred_spec = self.decode2(pred_mel, is_training=self.is_training)

        return ppgs, preds_ppg, logits_ppg, pred_spec, pred_mel

    def _netD(self, d_input, reuse=False):
        def prenet_dis(inputs, num_units=None, dropout_rate=0., is_training=True, scope="prenet", reuse=None):
            '''Prenet for Encoder and Decoder.
            Args:
              inputs: A 3D tensor of shape [N, T, hp.embed_size].
              is_training: A boolean.
              scope: Optional scope for `variable_scope`.
              reuse: Boolean, whether to reuse the weights of a previous layer
                by the same name.

            Returns:
              A 3D tensor of shape [N, T, num_units/2].
            '''
            with tf.variable_scope(scope, reuse=reuse):
                outputs = tf.layers.dense(inputs, units=num_units[0], activation=tf.nn.relu, name="dense1")
                outputs = tf.layers.dense(outputs, units=num_units[1], activation=tf.nn.relu, name="dense2")

            return outputs  # (N, T, num_units/2)

        with tf.variable_scope('netD', reuse=reuse):
            # Pre-net
            out = prenet_dis(d_input,
                             num_units=[hp.Train1.hidden_units, hp.Train1.hidden_units // 2],
                             dropout_rate=hp.Train1.dropout_rate,
                             is_training=self.is_training,
                             scope="prenet1")  # (N, T, E/2)

            out = cbhg(out, hp.Train1.num_banks, hp.Train1.hidden_units // 2, hp.Train1.num_highway_blocks,
                       hp.Train1.norm_type, self.is_training)

            # recurrent
            out = gru(out, hp.Train1.hidden_units // 2, True,
                      scope="gru1")


            ## Disciriminator output
            out = out[:, -1, :]
            out = normalize(out, reuse=reuse)
            W_dis = tf.get_variable("weights", shape=[hp.Train1.hidden_units, 1])
            b_dis = tf.get_variable("bias", shape=[1])
            out = tf.sigmoid(tf.matmul(out, W_dis) + b_dis)

            return out

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
