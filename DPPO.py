#!/usr/bin/env python3
"""
Refer to the work of OpenAI and DeepMind.

Algorithm:
A simple version of OpenAI's Proximal Policy Optimization (PPO). [https://arxiv.org/abs/1707.06347]
Distributing workers in parallel to collect data, then stop worker's roll-out and train PPO on collected data.
Restart workers once PPO is updated.
The global PPO updating rule is adopted from DeepMind's paper (DPPO):
Emergence of Locomotion Behaviours in Rich Environments (Google Deepmind): [https://arxiv.org/abs/1707.02286]

Dependencies:
tensorflow r1.5
gym 0.9.2
gym_OptClang

Thanks to MorvanZhou's implementation: https://morvanzhou.github.io/tutorials
I learned a lot from him =)
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym, gym_OptClang
import random, threading, queue, operator, os, sys, re
from operator import itemgetter
from random import shuffle
from colorama import Fore, Style
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import time

EP_MAX = 300
N_WORKER = 5                # parallel workers
GAMMA = 0.95                # reward discount factor
A_LR = 0.0001               # learning rate for actor
C_LR = 0.0002               # learning rate for critic
MIN_BATCH_SIZE = 64         # minimum batch size for updating PPO
EPSILON = 0.2               # for clipping surrogate objective

"""
Shared vars
"""
'''
All the methods in Python threading.events are atomic operations.
https://docs.python.org/3/library/threading.html
'''
SharedEvents = {}
SharedEvents['update'] = threading.Event()
SharedEvents['update'].clear()            # not update now
SharedEvents['collect'] = threading.Event()
SharedEvents['collect'].set()             # start to collect
# prevent race condition with 3 locks
Locks = {}
Locks['queue'] = threading.Lock()
Locks['counter'] = threading.Lock()
Locks['plot_epi'] = threading.Lock()
# counters for syncornization
GlobalCounters = {}
GlobalCounters['ep'] = 0
GlobalCounters['update_counter'] = 0
GlobalCounters['running_reward'] = []
# a global dict to access everything
GlobalStorage = {}
GlobalStorage['Events'] = SharedEvents
GlobalStorage['Locks'] = Locks
GlobalStorage['Counters'] = GlobalCounters
# coordinator for threads
GlobalStorage['Coordinator'] = tf.train.Coordinator()
# workers putting data in this queue
GlobalStorage['DataQueue'] = queue.Queue()


class PPO(object):
    def __init__(self, env, ckptLoc):
        tf.reset_default_graph()
        self.S_DIM = len(env.observation_space.low)
        self.A_DIM = env.action_space.n
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, self.S_DIM], 'state')
        self.ckptLoc = ckptLoc

        # critic
        with tf.variable_scope('Critic'):
            with tf.variable_scope('Fully_Connected'):
                l1 = self.add_layer(self.tfs, 100, activation_function=tf.nn.leaky_relu, norm=True)
            with tf.variable_scope('Value'):
                self.v = tf.layers.dense(l1, 1)
            with tf.variable_scope('Loss'):
                self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
                self.advantage = self.tfdc_r - self.v
                self.closs = tf.reduce_mean(tf.square(self.advantage))
            with tf.variable_scope('CriticTrain'):
                self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # actor
        pi, pi_params = self._build_anet('Actor', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldActor', trainable=False)
        # operation of choosing action
        with tf.variable_scope('ActionsProbs'):
            #self.sample_op = tf.squeeze(pi.sample(1), axis=0)
            self.sample_op = tf.squeeze(pi, axis=0)
        with tf.variable_scope('Update'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, self.A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        with tf.variable_scope('ppoLoss'):
            # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
            #ratio = pi.prob(self.tfa) / (oldpi.prob(self.tfa) + 1e-5)
            # add a small number to avoid NaN
            ratio = pi / (oldpi + 1e-6)
            # surrogate loss
            surr = ratio * self.tfadv
            # clipped surrogate objective
            self.aloss = -tf.reduce_mean(tf.minimum(
                surr,
                tf.clip_by_value(ratio, 1. - EPSILON, 1. + EPSILON) * self.tfadv))

        with tf.variable_scope('ActorTrain'):
            self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)
        self.writer = tf.summary.FileWriter('./logs', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        '''
        If the ckpt exist, restore it.
        '''
        if tf.train.checkpoint_exists(self.ckptLoc):
            self.saver.restore(self.sess, self.ckptLoc)
            print(Fore.LIGHTGREEN_EX + 'Restore the previous model.')
            print(Style.RESET_ALL)

    def update(self):
        global GlobalStorage
        while not GlobalStorage['Coordinator'].should_stop():
            if GlobalStorage['Counters']['ep'] < EP_MAX:
                # wait until get batch of data
                GlobalStorage['Events']['update'].wait()
                # copy pi to old pi
                self.sess.run(self.update_oldpi_op)
                # collect data from all workers
                data = [GlobalStorage['DataQueue'].get() for _ in range(GlobalStorage['DataQueue'].qsize())]
                data = np.vstack(data)
                s, a, r = data[:, :self.S_DIM], data[:, self.S_DIM: self.S_DIM + self.A_DIM], data[:, -1:]
                adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
                # update actor and critic in a update loop
                for _ in range(len(data)):
                    self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv})
                for _ in range(len(data)):
                    self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r})
                '''
                Save the model
                '''
                self.saver.save(self.sess, self.ckptLoc)

                # updating finished
                GlobalStorage['Events']['update'].clear()
                GlobalStorage['Locks']['counter'].acquire()
                # reset counter
                GlobalStorage['Counters']['update_counter'] = 0
                GlobalStorage['Locks']['counter'].release()
                # set collecting available
                GlobalStorage['Events']['collect'].set()
        print(Fore.YELLOW + 'Updator stopped')
        print(Style.RESET_ALL)

    def add_layer(self, inputs, out_size, trainable=True,activation_function=None, norm=False):
        in_size = inputs.get_shape().as_list()[1]
        Weights = tf.Variable(tf.random_normal([in_size, out_size], mean=0., stddev=1.), trainable=trainable)
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.01, trainable=trainable)

        # fully connected product
        Wx_plus_b = tf.matmul(inputs, Weights) + biases

        # normalize fully connected product
        if norm:
            with tf.variable_scope('BatchNormalization'):
                # Batch Normalize
                fc_mean, fc_var = tf.nn.moments(
                    Wx_plus_b,
                    axes=[0],   # the dimension you wanna normalize, here [0] for batch
                                # for image, you wanna do [0, 1, 2] for [batch, height, width] but not channel
                )
                scale = tf.Variable(tf.ones([out_size]))
                shift = tf.Variable(tf.zeros([out_size]))
                epsilon = 0.001

                # apply moving average for mean and var when train on batch
                ema = tf.train.ExponentialMovingAverage(decay=0.5)
                def mean_var_with_update():
                    ema_apply_op = ema.apply([fc_mean, fc_var])
                    with tf.control_dependencies([ema_apply_op]):
                        return tf.identity(fc_mean), tf.identity(fc_var)
                mean, var = mean_var_with_update()

                Wx_plus_b = tf.nn.batch_normalization(Wx_plus_b, mean, var, shift, scale, epsilon)
                # similar with this two steps:
                # Wx_plus_b = (Wx_plus_b - fc_mean) / tf.sqrt(fc_var + 0.001)
                # Wx_plus_b = Wx_plus_b * scale + shift

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
                #l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu, trainable=trainable)
                l1 = self.add_layer(self.tfs, 100, trainable,activation_function=tf.nn.leaky_relu, norm=True)
            '''
            with tf.variable_scope('mu'):
                #mu = 2 * tf.layers.dense(l1, self.A_DIM, tf.nn.tanh, trainable=trainable)
                mu = 2 * self.add_layer(l1, self.A_DIM, trainable,activation_function=tf.nn.tanh, norm=True)
            with tf.variable_scope('sigma'):
                #sigma = tf.layers.dense(l1, self.A_DIM, tf.nn.softplus, trainable=trainable)
                sigma = self.add_layer(l1, self.A_DIM, trainable,activation_function=tf.nn.softplus, norm=True)
            with tf.variable_scope('Normal_Distribution'):
                norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
            '''
            with tf.variable_scope('Action_Probs'):
                acts_prob = self.add_layer(l1, self.A_DIM, activation_function=tf.nn.softmax, norm=False)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        #return norm_dist, params
        return acts_prob, params

    def choose_action(self, s, PassHistory):
        """
        return a int from 0 to 33
        In the world of reinforcement learning, the action space is from 0 to 33.
        However, in the world of modified-clang, the accepted passes are from 1 to 34.
        Therefore, "gym-OptClang" already done this effort for us.
        We don't have to bother this by ourselves.
        """
        s = s[np.newaxis, :]
        #a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        a = self.sess.run(self.sample_op, {self.tfs: s})
        print(a)
        '''
        choose the one that was not applied yet
        '''
        # split the probabilities into list of [index ,probablities]
        aList = a.tolist()
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
        for actionProb in probList:
            PassIdx = actionProb[0]
            PassProb = actionProb[1]
            if PassIdx not in PassHistory:
                PassHistory[PassIdx] = 'Used'
                return PassIdx
        # the code should never come to here
        return 'Error'

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]


class Worker(object):
    def __init__(self, WorkerID):
        global GlobalPPO
        self.wid = WorkerID
        self.env = gym.make(Game).unwrapped
        self.ppo = GlobalPPO

    def getMostInfluentialState(self, states, ResetInfo):
        """
        return the most influential features from profiled data.
        If not profiled, random pick.
        If the function name does not match in the fetures, try others based on the usage
        in descending order.
        return an numpy array object.
        """
        retVec = None
        Stats = ResetInfo["FunctionUsageDict"]
        if not Stats.items():
            '''
            nothing profiled, random select
            '''
            key = random.choice(list(states.keys()))
        else:
            '''
            select the function with the maximum usage
            '''
            key = max(Stats.items(), key=operator.itemgetter(1))[0]
        #key = "hi_function"
        try:
            retVec = states[key]
        except KeyError:
            '''
            Random selection will never come to here.
            This is caused by perf profiled information which does not contain the function arguments.
            '''
            #print("Using re to search C++ style name\nKey error:\nkey={}\ndict.keys()={}\n".format(key, states.keys()))
            try:
                FunctionList = list(states.keys())
                done = False
                # build list of [key, usage]
                UsageList = []
                for name, usage in Stats.items():
                    UsageList.append([name, usage])
                # based on the usage, sort it
                sorted(UsageList, key=operator.itemgetter(1), reverse=True)
                # use RegExp to search C++ style name or ambiguity of arguments.
                NameList = []
                UsageTmpList = []
                done = False
                for item in UsageList:
                    NameList.append(item[0])
                    UsageTmpList.append(item[1])
                for cand in NameList:
                    # searching based on the usage order in descending.
                    realKey = self.RegExpSearch(cand, FunctionList)
                    if realKey is not None:
                        done = True
                        break
                if not done:
                    # if we cannot find the key, use the random one.
                    realKey = random.choice(FunctionList)
                retVec = states[realKey]
            except Exception as e:
                print("Unexpected exception\nkey={}\nrealKey={}\ndict.keys()={}\nreason={}\n".format(key, realKey ,states.keys()), e)
        return np.asarray(retVec)

    def RegExpSearch(self, TargetName, List):
        """
        Use regular exp. to search whether the List contains the TargetName.
        Inputs:
            TargetName: the name you would like to find.
            List: list of candidates for searching.
        Return:
            The matched name in List or None
        """
        retName = None
        done = False
        for candidate in List:
            matched = re.search(re.escape(TargetName), candidate)
            if matched is not None:
                retName = candidate
                done = True
                break
        if not done:
            ReEscapedInput = re.escape(TargetName)
            SearchTarget = ".*{name}.*".format(name=ReEscapedInput)
            r = re.compile(SearchTarget)
            reRetList = list(filter(r.search, List))
            if reRetList:
                retName = reRetList[0]
        return retName

    def calcEachReward(self, newInfo, MeanSigmaDict, Features, oldInfo, oldCycles, FirstEpi=False):
        """
        return dict={"function-name": reward(float)}

        if FirstEpi == True:
            oldInfo will be the ResetInfo
        if FirstEpi == False:
            oldInfo will be the usage dict from last epi.
        """
        Stats = newInfo["FunctionUsageDict"]
        TotalCycles = newInfo["TotalCyclesStat"]
        Target = newInfo["Target"]
        '''
        Generate dict for function name mapping between perf style and clang style
        (Info["FunctionUsageDict"] <--> Features)
        {"perf_style_name": "clang_style_name"}
        '''
        NameMapDict = {}
        AllFunctions = list(Features.keys())
        for perfName in list(Stats.keys()):
            NameMapDict[perfName] = self.RegExpSearch(perfName, AllFunctions)
        '''
        Create usage dict with clang_style_name as key.
        if not profiled, the value will be None
        '''
        newAllUsageDict = {k : None for k in AllFunctions}
        for perf_name, clang_name in NameMapDict.items():
            newAllUsageDict[clang_name] = Stats[perf_name]
        '''
        Prepare the old usage dict
        '''
        if FirstEpi == True:
            resetStats = oldInfo["FunctionUsageDict"]
            resetNameMapDict = {}
            resetAllFunctions = list(Features.keys())
            for perfName in list(resetStats.keys()):
                resetNameMapDict[perfName] = self.RegExpSearch(perfName, resetAllFunctions)
            oldAllUsageDict = {k : None for k in resetAllFunctions}
            for perf_name, clang_name in resetNameMapDict.items():
                oldAllUsageDict[clang_name] = resetStats[perf_name]
        else:
            oldAllUsageDict = oldInfo
        '''
        Calculate real reward based on the (new/old)AllUsageDict and MeanSigmaDict for all functions
        '''
        rewards = {f : None for f in AllFunctions}
        target = newInfo['Target']
        old_total_cycles = oldCycles
        new_total_cycles = TotalCycles
        delta_total_cycles = old_total_cycles - new_total_cycles
        abs_delta_total_cycles = abs(delta_total_cycles)
        sigma_total_cycles = MeanSigmaDict[target]['sigma']
        '''
        95% of results are in the twice sigma.
        Therefore, 2x is necessary.
        '''
        SigmaRatio = abs((abs_delta_total_cycles - sigma_total_cycles)/(2*sigma_total_cycles))
        UsageNumOverAll = 0
        for name, usage in newAllUsageDict.items():
            if usage is not None:
                UsageNumOverAll += 1
        UsageProfiledRatio = UsageNumOverAll/len(newAllUsageDict)
        for FunctionName in AllFunctions:
            old_usage = oldAllUsageDict[FunctionName]
            new_usage = newAllUsageDict[FunctionName]
            UseOverallPerf = False
            '''
            The Alpha and Beta need to be tuned.
            '''
            Alpha = 2
            Beta = 2
            isSpeedup = False
            isSlowDown = False
            if old_usage is None and new_usage is None:
                '''
                This function does not matters
                '''
                UseOverallPerf = True
            elif old_usage is None:
                '''
                may be slow down
                '''
                UseOverallPerf = True
                Alpha *= Beta
                isSlowDown = True
            elif new_usage is None:
                '''
                may be speedup
                '''
                UseOverallPerf = True
                Alpha *= Beta
                isSpeedup = True
            else:
                '''
                This may be more accurate
                How important: based on how many functions are profiled.
                '''
                UseOverallPerf = False
                Alpha = Alpha*(Beta*(1 / UsageProfiledRatio)) * 2 # more important
            if UseOverallPerf:
                if isSlowDown == True and delta_total_cycles > 0:
                    Alpha /= Beta
                    delta_total_cycles *= -1
                elif isSpeedup == True and delta_total_cycles < 0:
                    Alpha /= Beta
                    delta_total_cycles *= -1
                reward = Alpha*SigmaRatio*(delta_total_cycles/old_total_cycles)
            else:
                old_function_cycles = old_total_cycles * old_usage
                new_function_cycles = new_total_cycles * new_usage
                delta_function_cycles = old_function_cycles - new_function_cycles
                reward = Alpha*SigmaRatio*(delta_function_cycles/old_function_cycles)
            #print("FunctionName={}, reward={}".format(FunctionName, reward))
            #print("Alpha={}".format(Alpha))
            rewards[FunctionName] = reward
        #print("UsageProfiledRatio={}".format(UsageProfiledRatio))
        # return newAllUsageDict to be the "old" for next episode
        return rewards, newAllUsageDict

    def appendStateRewards(self, buffer_s, buffer_a, buffer_r, states, rewards, action):
        #FIXME: do we need to discard some results that the rewards are not that important?
        """
        No return value, they are append inplace in buffer_x
        buffer_s : dict of np.array as features
        buffer_a : dict of list of actions(int)
        buffer_r : dict of list of rewards(float)
        """
        tmpStates = states.copy()
        for name, featureList in tmpStates.items():
            # For some reason, the name may be '' (remove it!)
            if not name:
                target = ''
                states.pop(target, None)
                if rewards.get(target) is not None:
                    rewards.pop(target, None)
        for name, featureList in states.items():
            if buffer_s.get(name) is None:
                buffer_s[name] = []
                buffer_a[name] = []
                buffer_r[name] = []
            buffer_s[name].append(np.asarray(featureList, dtype=np.float32))
            actionFeature = [0]*34
            actionFeature[action] = 1
            buffer_a[name].append(actionFeature)
            buffer_r[name].append(rewards[name])



    def calcDiscountedRewards(self, buffer_r, nextObs):
        """
        return a dict of list of discounted rewards
        {"function-name":[discounted rewards]}
        """
        global GAMMA
        retDict = {}
        for name, FeatureList in nextObs.items():
            '''
            Get estimated rewards from critic
            '''
            nextOb = np.asarray(FeatureList, dtype=np.float32)
            StateValue = self.ppo.get_v(nextOb)
            discounted_r = []
            for r in buffer_r[name][::-1]:
                '''
                Calculate discounted rewards
                '''
                StateValue = r + GAMMA * StateValue
                discounted_r.append(StateValue)
            discounted_r.reverse()
            retDict[name] = discounted_r
        return retDict


    def calcEpisodeReward(self, rewards):
        """
        return the overall reward.
        """
        total = 0.0
        count = 0.0
        for name, reward in rewards.items():
            total += reward
            count += 1.0
        #return total / count
        return total


    def getCpuMeanSigmaInfo(self):
        """
        return a dict{"target name": {"mean": int, "sigma": int}}
        """
        path = os.getenv('LLVM_THESIS_RandomHome', 'Error')
        if path == 'Error':
            print("$LLVM_THESIS_RandomHome is not defined, exit!", file=sys.stderr)
            sys.exit(1)
        path = path + '/LLVMTestSuiteScript/GraphGen/output/newMeasurableStdBenchmarkMeanAndSigma'
        if not os.path.exists(path):
            print("{} does not exist.".format(path), file=sys.stderr)
            sus.exit(1)
        retDict = {}
        with open(path, 'r') as file:
            for line in file:
                '''
                ex.
                PAQ8p/paq8p; cpu-cycles-mean | 153224947840; cpu-cycles-sigma | 2111212874
                '''
                lineList = line.split(';')
                name = lineList[0].split('/')[-1].strip()
                mean = int(lineList[1].split('|')[-1].strip())
                sigma = int(lineList[2].split('|')[-1].strip())
                retDict[name] = {'mean':mean, 'sigma':sigma}
            file.close()
        return retDict

    def DictToVstack(self, buffer_s, buffer_a, buffer_r):
        """
        return vstack of state(normalized), action and rewards.
        """
        list_s = []
        list_a = []
        list_r = []
        for name, values in buffer_s.items():
            list_s.extend(buffer_s[name])
            list_a.extend(buffer_a[name])
            list_r.extend(buffer_r[name])
        return np.vstack(list_s), np.vstack(list_a), np.vstack(list_r)

    def work(self):
        global GlobalStorage
        while not GlobalStorage['Coordinator'].should_stop():
            states, ResetInfo = self.env.reset()
            EpisodeReward = 0
            buffer_s, buffer_a, buffer_r = {}, {}, {}
            MeanSigmaDict = self.getCpuMeanSigmaInfo()
            FirstEpi = True
            PassHistory = {}
            while True:
                # while global PPO is updating
                if not GlobalStorage['Events']['collect'].is_set():
                    # wait until PPO is updated
                    GlobalStorage['Events']['collect'].wait()
                    # clear history buffer, use new policy to collect data
                    buffer_s, buffer_a, buffer_r = {}, {}, {}
                '''
                Save the last profiled info to calculate real rewards
                '''
                if FirstEpi:
                    oldCycles = ResetInfo["TotalCyclesStat"]
                    oldInfo = ResetInfo
                    FirstEpi = False
                    isUsageNotProcessed = True
                else:
                    oldCycles = info["TotalCyclesStat"]
                    oldInfo = oldAllUsage
                    isUsageNotProcessed = False
                '''
                Choose the features from the most inflential function
                '''
                state = self.getMostInfluentialState(states, ResetInfo)
                action = self.ppo.choose_action(state, PassHistory)
                nextStates, reward, done, info = self.env.step(action)
                '''
                If build failed, skip it.
                '''
                if reward < 0:
                    break

                '''
                Calculate actual rewards for all functions
                '''
                rewards, oldAllUsage = self.calcEachReward(info,
                        MeanSigmaDict, nextStates, oldInfo,
                        oldCycles, isUsageNotProcessed)

                '''
                Match the states and rewards
                '''
                self.appendStateRewards(buffer_s, buffer_a, buffer_r, states, rewards, action)

                '''
                Calculate overall reward for plotting
                '''
                EpisodeReward = self.calcEpisodeReward(rewards)

                # add the generated results
                GlobalStorage['Locks']['counter'].acquire()
                GlobalStorage['Counters']['update_counter'] = \
                    GlobalStorage['Counters']['update_counter'] + len(nextStates.keys())
                GlobalStorage['Locks']['counter'].release()
                if GlobalStorage['Counters']['update_counter'] >= MIN_BATCH_SIZE or done:
                    '''
                    Calculate discounted rewards for all functions
                    '''
                    discounted_r = self.calcDiscountedRewards(buffer_r, nextStates)
                    '''
                    Convert dict of list into row-array
                    '''
                    vstack_s, vstack_a, vstack_r = self.DictToVstack(buffer_s, buffer_a, discounted_r)
                    '''
                    Split each of vector and assemble into a queue element.
                    '''
                    GlobalStorage['Locks']['queue'].acquire()
                    # put data in the shared queue
                    for index, item in enumerate(vstack_s):
                        GlobalStorage['DataQueue'].put(
                                np.hstack((vstack_s[index], vstack_a[index], vstack_r[index])))
                    GlobalStorage['Locks']['queue'].release()
                    buffer_s, buffer_a, buffer_r = {}, {}, {}

                    if GlobalStorage['Counters']['update_counter'] >= MIN_BATCH_SIZE:
                        # stop collecting data
                        GlobalStorage['Events']['collect'].clear()
                        # globalPPO update
                        GlobalStorage['Events']['update'].set()

                    if GlobalStorage['Counters']['ep'] >= EP_MAX:
                        # stop training
                        GlobalStorage['Coordinator'].request_stop()
                        print(Fore.RED + 'WorkerID={} calls to Stop'.format(self.wid))
                        print(Style.RESET_ALL)
                        break
                if not done:
                    states = nextStates
                    continue
                else:
                    # clear history of applied passes
                    PassHistory = {}
                    # record reward changes, plot later
                    GlobalStorage['Locks']['plot_epi'].acquire()
                    if len(GlobalStorage['Counters']['running_reward']) == 0:
                        GlobalStorage['Counters']['running_reward'].append(EpisodeReward)
                    else:
                        GlobalStorage['Counters']['running_reward'].append(GlobalStorage['Counters']['running_reward'][-1]*0.9+EpisodeReward*0.1)
                    GlobalStorage['Counters']['ep'] += 1
                    GlobalStorage['Locks']['plot_epi'].release()
                    print(Fore.GREEN + '{0:.1f}%'.format(GlobalStorage['Counters']['ep']/EP_MAX*100), '|W%i' % self.wid, '|EpisodeReward: %.4f' % EpisodeReward,)
                    print(Style.RESET_ALL)
                    break

        print(Fore.YELLOW + 'WorkerID={} stopped'.format(self.wid))
        print(Style.RESET_ALL)


if __name__ == '__main__':
    Game='OptClang-v0'
    # path for save and restore the model
    ckptLoc = './logs/model.ckpt'
    GlobalPPO = PPO(gym.make(Game).unwrapped, ckptLoc)
    # remove worker file list.
    WorkerListLoc = "/tmp/gym-OptClang-WorkerList"
    if os.path.exists(WorkerListLoc):
        os.remove(WorkerListLoc)

    workers = []
    for i in range(N_WORKER):
        workers.append(Worker(WorkerID=(i+1)))

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
        GlobalStorage['Coordinator'].join(threads=threads, stop_grace_period_secs=1)
    except RuntimeError:
        print(Fore.RED + "Some of the workers cannot be stopped within 1 sec.\nYou can ignore the messages after this msg.")
        print(Style.RESET_ALL)

    # plot changes of rewards
    plt.plot(np.arange(len(GlobalStorage['Counters']['running_reward'])), GlobalStorage['Counters']['running_reward'])
    plt.xlabel('Episode')
    plt.ylabel('Moving reward')
    plt.savefig('running_rewards.png')
    print(Fore.RED + 'running_rewards.png is saved.')
    print(Style.RESET_ALL)
