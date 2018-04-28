#!/usr/bin/env python3
"""
Refer to the work of OpenAI and DeepMind.

Algorithm:
OpenAI's Proximal Policy Optimization (PPO). [https://arxiv.org/abs/1707.06347]
Emergence of Locomotion Behaviours in Rich Environments (Google Deepmind): [https://arxiv.org/abs/1707.02286]

Dependencies:
tensorflow
gym
gym_OptClang

Thanks to MorvanZhou's implementation: https://morvanzhou.github.io/tutorials
The basic structure is derived from him.
However, the internal structure is tuned for gym_OptClang.
"""
import tensorflow as tf
import numpy as np
import matplotlib
# do not use x-server
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gym, gym_OptClang
import random, threading, queue, operator, os, sys, re
from operator import itemgetter
from random import shuffle
import random
from colorama import Fore, Style
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import time
import io
from time import gmtime, strftime
import argparse
import pytz
import Helpers as hp

class PPO(object):
    def __init__(self, env, ckptLocBase, ckptName, isTraining, EP_MAX, GAMMA, A_LR, C_LR, ClippingEpsilon, UpdateDepth, L1Neurons, L2Neurons, LR_DECAY=1, SharedStorage=None):
        tf.reset_default_graph()
        # if SharedStorage is None, it must be in inference mode without "update()"
        self.SharedStorage = SharedStorage
        self.EP_MAX = EP_MAX
        self.GAMMA = GAMMA
        self.A_LR = A_LR
        self.C_LR = C_LR
        self.LR_DECAY = LR_DECAY
        self.ClippingEpsilon = ClippingEpsilon
        self.UpdateDepth = UpdateDepth
        self.L1Neurons = L1Neurons
        self.L2Neurons = L2Neurons
        self.S_DIM = len(env.observation_space.low)
        self.A_DIM = env.action_space.n
        self.A_SPACE = 1
        self.sess = tf.Session(graph=tf.get_default_graph())
        self.tfs = tf.placeholder(tf.float32, [None, self.S_DIM], 'state')
        self.ckptLocBase = ckptLocBase
        self.UpdateStepFile = self.ckptLocBase + '/UpdateStep'
        self.ActorLrFile = self.ckptLocBase + '/ActorLrFile'
        self.CriticLrFile = self.ckptLocBase + '/CrticLrFile'
        hp.ColorPrint(Fore.LIGHTCYAN_EX, "Log dir={}".format(self.ckptLocBase))
        self.ckptLoc = ckptLocBase + '/' + ckptName
        self.UpdateStep = 0
        if os.path.exists(self.UpdateStepFile):
            with open(self.UpdateStepFile, 'r') as f:
                self.UpdateStep = int(f.read())
            hp.ColorPrint(Fore.GREEN, "Restored episode step={}".format(self.UpdateStep))
        if os.path.exists(self.ActorLrFile):
            with open(self.ActorLrFile, 'r') as f:
                self.A_LR = int(f.read())
            hp.ColorPrint(Fore.GREEN, "Restored A_LR={}".format(self.A_LR))
        if os.path.exists(self.CriticLrFile):
            with open(self.CriticLrFile, 'r') as f:
                self.C_LR = int(f.read())
            hp.ColorPrint(Fore.GREEN, "Restored C_LR={}".format(self.C_LR))
        if isTraining == 'N':
            self.isTraining = False
            hp.ColorPrint(Fore.LIGHTCYAN_EX, "This is inference procedure")
        else:
            self.isTraining = True
            hp.ColorPrint(Fore.LIGHTCYAN_EX, "This is training procedure with UpdateStep={}".format(self.UpdateStep))

        # critic
        with tf.variable_scope('Critic'):
            with tf.variable_scope('Fully_Connected'):
                l1 = self.add_layer(self.tfs, self.L1Neurons, activation_function=tf.nn.relu, norm=True)
                if self.L2Neurons != 0:
                    l2 = self.add_layer(l1, self.L2Neurons, activation_function=tf.nn.relu, norm=True)
            with tf.variable_scope('Value'):
                if self.L2Neurons != 0:
                    self.v = tf.layers.dense(l2, 1)
                else:
                    self.v = tf.layers.dense(l1, 1)
            with tf.variable_scope('Loss'):
                self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
                self.advantage = self.tfdc_r - self.v
                self.closs = tf.reduce_mean(tf.square(self.advantage))
                self.CriticLossSummary = tf.summary.scalar('CriticLoss', self.closs)
            with tf.variable_scope('CriticTrain'):
                self.ctrain_op = tf.train.AdamOptimizer(self.C_LR).minimize(self.closs)

        # pi: act_probs
        pi, pi_params = self._build_anet('Actor', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldActor', trainable=False)
        # operation of choosing action
        with tf.variable_scope('ActionsExp.'):
            self.acts_expect = tf.squeeze(pi, axis=0)
        with tf.variable_scope('Update'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        with tf.variable_scope('Actor/PPO-Loss'):
            self.tfa = tf.placeholder(tf.int32, [None, 1], 'action')
            self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
            # probabilities of actions which agent took with policy
            # depth=pi.shape[0] <-- each column is viewed as a vector
            # depth=pi.shape[1] <-- each row is viewed as a vector <-- we use this
            act_probs = pi * tf.one_hot(indices=self.tfa, depth=pi.shape[1])
            act_probs = tf.reduce_sum(act_probs, axis=1)
            # probabilities of actions which old agent took with policy
            act_probs_old = oldpi * tf.one_hot(indices=self.tfa, depth=oldpi.shape[1])
            act_probs_old = tf.reduce_sum(act_probs_old, axis=1)
            # add a small number to avoid NaN
            ratio = tf.divide(act_probs + 1e-10, act_probs_old + 1e-10)
            #ratio = tf.exp(tf.log(act_probs) - tf.log(act_probs_old))
            surr = tf.multiply(ratio, self.tfadv)
            clip = tf.clip_by_value(ratio, 1.-self.ClippingEpsilon, 1.+self.ClippingEpsilon)*self.tfadv
            # clipped surrogate objective
            self.aloss = -tf.reduce_mean(tf.minimum(surr, clip))
            # visualizing
            self.ppoRatioSummary = tf.summary.tensor_summary('ppoRatio', ratio)
            self.ActorLossSummary = tf.summary.scalar('ActorLoss', self.aloss)

        with tf.variable_scope('ActorTrain'):
            self.atrain_op = tf.train.AdamOptimizer(self.A_LR).minimize(self.aloss)

        with tf.variable_scope('Summary'):
            self.OverallSpeedup = tf.placeholder(tf.float32, name='OverallSpeedup')
            self.EpisodeReward = tf.placeholder(tf.float32, name='EpisodeReward')
            self.one = tf.constant(1.0, dtype=tf.float32)
            self.RecordSpeedup_op = tf.multiply(self.OverallSpeedup, self.one)
            self.SpeedupSummary = tf.summary.scalar('OverallSpeedup', self.RecordSpeedup_op)
            self.RecordEpiReward_op = tf.multiply(self.EpisodeReward, self.one)
            self.EpiRewardSummary = tf.summary.scalar('EpisodeReward', self.RecordEpiReward_op)

        self.writer = tf.summary.FileWriter(self.ckptLocBase, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        '''
        If the ckpt exist, restore it.
        '''
        if tf.train.checkpoint_exists(self.ckptLoc):
            #self.saver.restore(self.sess, self.ckptLoc)
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.ckptLocBase))
            hp.ColorPrint(Fore.LIGHTGREEN_EX, 'Restore the previous model.')
        elif self.isTraining == False:
            hp.ColorPrint(Fore.LIGHTRED_EX, "Missing trained model to inference, exit.")
            sys.exit(1)

    def save(self):
        """
        Save model
        """
        self.saver.save(self.sess, self.ckptLoc)

    def update(self):
        while not self.SharedStorage['Coordinator'].should_stop():
            if self.SharedStorage['Counters']['ep'] < self.EP_MAX:
                # blocking wait until get batch of data
                self.SharedStorage['Events']['update'].wait()
                # save the model
                if self.UpdateStep % 50 == 0:
                    self.save()
                    hp.ColorPrint(Fore.LIGHTRED_EX, "Save for every 50 updates.")
                else:
                    hp.ColorPrint(Fore.LIGHTBLUE_EX,
                            "This update does not need to be saved: {}".format(self.UpdateStep))
                # learning rate decay
                if self.UpdateStep % 2000 == 1999:
                    # decay
                    self.A_LR = self.A_LR * self.LR_DECAY
                    self.C_LR = self.C_LR * self.LR_DECAY
                    # save
                    with open(self.ActorLrFile, 'w') as f:
                        f.write(str(self.A_LR))
                    with open(self.CriticLrFile, 'w') as f:
                        f.write(str(self.C_LR))
                    hp.ColorPrint(Fore.LIGHTRED_EX,
                            "Decay LR: A_LR={}, C_LR={}".format(self.A_LR, self.C_LR))
                # copy pi to old pi
                self.sess.run(self.update_oldpi_op)
                # collect data from all workers
                data = [self.SharedStorage['DataQueue'].get() for _ in range(self.SharedStorage['DataQueue'].qsize())]
                data = np.vstack(data)
                s, a, r = data[:, :self.S_DIM], data[:, self.S_DIM: self.S_DIM + self.A_SPACE], data[:, -1:]
                adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
                # update actor and critic in a update loop
                for _ in range(self.UpdateDepth):
                    self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv})
                    self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r})
                '''
                write summary
                '''
                # actor and critic loss
                result = self.sess.run(
                            tf.summary.merge([self.ActorLossSummary, self.CriticLossSummary, 
                                self.ppoRatioSummary]),
                            feed_dict={self.tfs: s, self.tfa: a, self.tfadv: adv, self.tfdc_r: r})
                self.writer.add_summary(result, self.UpdateStep)
                self.UpdateStep += 1
                # re-train will not overlap the summaries
                with open(self.UpdateStepFile, 'w') as f:
                    f.write(str(self.UpdateStep))

                # updating finished
                self.SharedStorage['Events']['update'].clear()
                self.SharedStorage['Locks']['counter'].acquire()
                # reset counter
                self.SharedStorage['Counters']['update_counter'] = 0
                self.SharedStorage['Locks']['counter'].release()
                # set collecting available
                self.SharedStorage['Events']['collect'].set()
        hp.ColorPrint(Fore.YELLOW, 'Updator stopped')

    def add_layer(self, inputs, out_size, trainable=True,activation_function=None, norm=False):
        in_size = inputs.get_shape().as_list()[1]
        Weights = tf.Variable(tf.random_normal([in_size, out_size], mean=1.0, stddev=1.0), trainable=trainable)
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, trainable=trainable)

        # fully connected product
        Wx_plus_b = tf.matmul(inputs, Weights) + biases

        # normalize fully connected product
        if norm:
            # Batch Normalize
            Wx_plus_b = tf.contrib.layers.batch_norm(
                    Wx_plus_b, updates_collections=None, is_training=self.isTraining)

        # activation
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            with tf.variable_scope('ActivationFunction'):
                outputs = activation_function(Wx_plus_b)

        return outputs

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            with tf.variable_scope('Fully_Connected'):
                l1 = self.add_layer(self.tfs, self.L1Neurons, trainable,activation_function=tf.nn.relu, norm=True)
                if self.L2Neurons != 0:
                    l2 = self.add_layer(l1, self.L2Neurons, trainable,activation_function=tf.nn.relu, norm=True)
            with tf.variable_scope('Action_Expectation'):
                # softmax may lead to NaN
                if self.L2Neurons != 0:
                    expectation = \
                            self.add_layer(l2, self.A_DIM, activation_function=tf.nn.softmax, norm=True)
                else:
                    expectation = \
                            self.add_layer(l1, self.A_DIM, activation_function=tf.nn.softmax, norm=True)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return expectation, params

    def choose_action(self, s, PassHistory):
        """
        return a int from 0 to 33
        Input "s" must be numpy array.
        In the world of reinforcement learning, the action space is from 0 to 33.
        However, in the world of modified-clang, the accepted passes are from 1 to 34.
        Therefore, "gym-OptClang" already done this effort for us.
        We don't have to be bothered by this.
        However, if you use the model withou gym-OptClang, you have to convert by yourself.
        e.g. Inference example in our examples.
        """
        s = s[np.newaxis, :]
        a_expect = self.sess.run(self.acts_expect, {self.tfs: s})
        print(a_expect)
        '''
        choose the one that was not applied yet
        '''
        # split the probabilities into list of [index ,probablities]
        aList = a_expect.tolist()
        probList = []
        idx = 0
        for prob in aList:
            probList.append([idx, prob])
            idx += 1
        # some probs may be the same.
        # Try to avoid that every time choose the same action
        shuffle(probList)
        # sort with probs in descending order
        probList.sort(key=itemgetter(1), reverse=True)
        # find the one that is not applied yet
        idx = 0
        while True:
            '''
            During training, we need some chance to get unexpected action to let
            the agent face different conditions as much as possible.
            '''
            # Use different strategies for different situations
            if self.isTraining == True:
                prob = random.uniform(0, 1)
                if prob < 0.8:
                    # the most possible action
                    PassIdx = probList[idx][0]
                    idx += 1
                else:
                    # random action
                    PassIdx = np.random.choice(np.arange(self.A_DIM))
            else:
                PassIdx = probList[idx][0]
                idx += 1
            #print('PassIdx={} with {} prob'.format(PassIdx, actionProb[1]))
            if PassIdx not in PassHistory:
                PassHistory[PassIdx] = 'Used'
                return PassIdx
        # the code should never come to here
        return 'Error'

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]

    def DrawToTf(self, speedup, overall_reward, step):
        """
        This is not thread-safe
        """
        try:
            result = self.sess.run(
                        tf.summary.merge([self.SpeedupSummary, self.EpiRewardSummary]),
                        feed_dict={self.OverallSpeedup: speedup,
                                   self.EpisodeReward: overall_reward})
            self.writer.add_summary(result, step)
            with open(self.ckptLocBase + '/EpiStepFile', 'w') as f:
                f.write(str(step))
            self.writer.flush()
        except Exception as e:
            ColorPrint(Fore.LIGHTRED_EX, "SpeedupSummary or EpiRewardSummary failed: {}".fomat(e))
