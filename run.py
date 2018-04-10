#!/usr/bin/env python3
"""
Dependencies:
tensorflow
gym
gym_OptClang
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
import DPPO
import Helpers as hp
from Helpers import EnvCalculator as calc

config = hp.LoadJsonConfig('./config.json')
#----------------Worker hyper-parameters()-----------------#
EP_MAX = config['WorkerParameters']['EP_MAX']
# parallel workers
N_WORKER = config['WorkerParameters']['N_WORKER']
# minimum batch size for updating PPO
MIN_BATCH_SIZE = config['WorkerParameters']['MIN_BATCH_SIZE']
# reward discount factor
GAMMA = config['WorkerParameters']['GAMMA']
#----------------PPO hyper-parameters--------------------#
# for clipping surrogate objective
ClippingEpsilon = config['RL_Parameters']['ClippingEpsilon']
# learning rate for actor
A_LR = config['RL_Parameters']['A_LR']
# learning rate for critic
C_LR = config['RL_Parameters']['C_LR']
# learn multiple times. Because of the PPO will constrain the update speed.
UpdateDepth = config['RL_Parameters']['UpdateDepth']
# number of L1 neurons
L1Neurons = config['RL_Parameters']['L1Neurons']
# number of L2 neurons
L2Neurons = config['RL_Parameters']['L2Neurons']

# Initialize the necessary vars
SharedStorage = hp.InitSharedStorage()

class Worker(object):
    def __init__(self, WorkerID, SharedStorage, LogDir):
        global GlobalPPO
        self.wid = WorkerID
        self.env = gym.make(Game).unwrapped
        self.ppo = GlobalPPO
        self.SharedStorage = SharedStorage
        self.LogDir = LogDir

    def work(self):
        while not self.SharedStorage['Coordinator'].should_stop():
            states, ResetInfo = self.env.reset()
            EpisodeReward = 0
            buffer_s, buffer_a, buffer_r = {}, {}, {}
            MeanSigmaDict = calc.getCpuMeanSigmaInfo()
            FirstEpi = True
            PassHistory = {}
            while True:
                # while global PPO is updating
                if not self.SharedStorage['Events']['collect'].is_set():
                    # wait until PPO is updated
                    self.SharedStorage['Events']['collect'].wait()
                    # clear history buffer, use new policy to collect data
                    buffer_s, buffer_a, buffer_r = {}, {}, {}
                '''
                Save the last profiled info to calculate real rewards
                '''
                try:
                    if FirstEpi:
                        oldCycles = ResetInfo["TotalCyclesStat"]
                        oldInfo = ResetInfo
                        FirstEpi = False
                        isUsageNotProcessed = True
                    else:
                        oldCycles = info["TotalCyclesStat"]
                        oldInfo = oldAllUsage
                        isUsageNotProcessed = False
                except Exception as e:
                    hp.ColorPrint(Fore.RED, "Exception happened and skipped: \n{}".format(e))
                    break
                '''
                Choose the features from the most inflential function
                '''
                state = calc.getMostInfluentialState(states, ResetInfo)
                action = self.ppo.choose_action(state, PassHistory)
                nextStates, reward, done, info = self.env.step(action)
                '''
                If build failed, skip it.
                '''
                if reward < 0:
                    # clear history of applied passes
                    PassHistory = {}
                    hp.ColorPrint(Fore.RED, 'WorkerID={} env.step() Failed. Use new target and forget these memories'.format(self.wid))
                    break
                '''
                Calculate actual rewards for all functions
                '''
                rewards, oldAllUsage = calc.calcEachReward(info,
                        MeanSigmaDict, nextStates, oldInfo,
                        oldCycles, isUsageNotProcessed)
                '''
                Speedup for tf.summary
                Skip this iteration, if the speedup/slowdown is not obvious
                '''
                speedup = calc.calcOverallSpeedup(ResetInfo, info)
                if abs(speedup) < 0.01:
                    hp.ColorPrint(Fore.RED,
                            "WorkerID={}, Speedup={} --> skip this iteration".format(self.wid, speedup))
                    if done:
                        # This may lose some data for training.
                        break
                    else:
                        states = nextStates
                        continue
                '''
                Match the states and rewards
                '''
                calc.appendStateRewards(buffer_s, buffer_a, buffer_r, states, rewards, action)

                '''
                Calculate overall reward for summary
                '''
                EpisodeReward = calc.calcEpisodeReward(rewards)

                # add the generated results
                self.SharedStorage['Locks']['counter'].acquire()
                self.SharedStorage['Counters']['update_counter'] = \
                    self.SharedStorage['Counters']['update_counter'] + len(nextStates.keys())
                self.SharedStorage['Locks']['counter'].release()
                if self.SharedStorage['Counters']['update_counter'] >= MIN_BATCH_SIZE or done:
                    '''
                    Calculate discounted rewards for all functions
                    '''
                    discounted_r = calc.calcDiscountedRewards(buffer_r, nextStates, self.ppo)
                    '''
                    Convert dict of list into row-array
                    '''
                    vstack_s, vstack_a, vstack_r = calc.DictToVstack(buffer_s, buffer_a, discounted_r)
                    '''
                    Remove data that are not important
                    '''
                    #vstack_s, vstack_a, vstack_r = calc.RemoveTrivialData(vstack_s, vstack_a, vstack_r)
                    '''
                    Split each of vector and assemble into a queue element.
                    '''
                    self.SharedStorage['Locks']['queue'].acquire()
                    # put data in the shared queue
                    for index, item in enumerate(vstack_s):
                        self.SharedStorage['DataQueue'].put(
                                np.hstack((vstack_s[index], vstack_a[index], vstack_r[index])))
                    self.SharedStorage['Locks']['queue'].release()
                    buffer_s, buffer_a, buffer_r = {}, {}, {}

                    if self.SharedStorage['Counters']['update_counter'] >= MIN_BATCH_SIZE:
                        # stop collecting data
                        self.SharedStorage['Events']['collect'].clear()
                        # globalPPO update
                        self.SharedStorage['Events']['update'].set()

                    if self.SharedStorage['Counters']['ep'] >= EP_MAX:
                        # stop training
                        self.SharedStorage['Coordinator'].request_stop()
                        hp.ColorPrint(Fore.RED, 'WorkerID={} calls to Stop'.format(self.wid))
                        break
                if not done:
                    states = nextStates
                    continue
                else:
                    # clear history of applied passes
                    PassHistory = {}
                    # record reward changes, plot later
                    self.SharedStorage['Locks']['plot_epi'].acquire()
                    # add episode count
                    self.SharedStorage['Counters']['ep'] += 1
                    '''
                    draw to tensorboard
                    '''
                    self.ppo.DrawToTf(speedup, EpisodeReward, self.SharedStorage['Counters']['ep'])
                    self.SharedStorage['Locks']['plot_epi'].release()
                    msg = '{0:}/{1:} ({2:.1f}%)'.format(self.SharedStorage['Counters']['ep'], EP_MAX,self.SharedStorage['Counters']['ep']/EP_MAX*100) + ' | WorkerID={}'.format(self.wid) + '\nEpisodeReward: {0:.4f}'.format(EpisodeReward) + ' | OverallSpeedup: {}'.format(speedup)
                    hp.ColorPrint(Fore.GREEN, msg)
                    break
        hp.ColorPrint(Fore.YELLOW, 'WorkerID={} stopped'.format(self.wid))



if __name__ == '__main__':
    Game='OptClang-v0'
    date = datetime.now(pytz.timezone('Asia/Taipei')).strftime("%m-%d_%H-%M")
    parser = argparse.ArgumentParser(
            description='\"Log dir\" and NN related settings\nex.\n$./DPPO.py -l ./logs -t Y\n')
    # path for save and restore the model
    parser.add_argument('-l', '--logdir', type=str, nargs='?',
                        default='./log_' + date,
                        help='Log directory to save and restore the NN model',
                        required=False)
    parser.add_argument('-t', '--training', type=str, nargs='?',
                        default='Y',
                        help='Is this run will be training procedure?\n\"Y\"=Training, \"N\"=Inference',
                        required=False)
    args = vars(parser.parse_args())
    # restore the episode count
    EpiStepFile = args['logdir']+'/EpiStepFile'
    if os.path.exists(EpiStepFile):
        with open(EpiStepFile, 'r') as f:
            SharedStorage['Counters']['ep'] = int(f.read())
            hp.ColorPrint(Fore.LIGHTCYAN_EX, "Restore Episode:{}".format(SharedStorage['Counters']['ep']))

    GlobalPPO = DPPO.PPO(gym.make(Game).unwrapped,
            args['logdir'], 'model.ckpt',
            isTraining=args['training'],
            SharedStorage=SharedStorage,
            EP_MAX=EP_MAX,
            GAMMA=GAMMA,
            A_LR=A_LR,
            C_LR=C_LR,
            ClippingEpsilon=ClippingEpsilon,
            L1Neurons=L1Neurons,
            L2Neurons=L2Neurons,
            UpdateDepth=UpdateDepth)
    # remove worker file list.
    WorkerListLoc = "/tmp/gym-OptClang-WorkerList"
    if os.path.exists(WorkerListLoc):
        os.remove(WorkerListLoc)

    workers = []
    for i in range(N_WORKER):
        workers.append(Worker(WorkerID=(i+1), SharedStorage=SharedStorage, LogDir=args['logdir']))

    threads = []
    for worker in workers:
        t = threading.Thread(target=worker.work, args=())
        t.start()
        threads.append(t)
    # add a PPO updating thread
    threads.append(threading.Thread(target=GlobalPPO.update,
        args=()))
    threads[-1].start()
    try:
        # Wait for all the threads to terminate, give them 1s grace period
        SharedStorage['Coordinator'].join(threads=threads, stop_grace_period_secs=1)
    except RuntimeError:
        hp.ColorPrint(Fore.RED, "Some of the workers cannot be stopped within 1 sec.\nYou can ignore the messages after this msg.")

