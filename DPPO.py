#!/usr/bin/env python3
"""
The algorithm is based on MorvanZhou's implementation: https://morvanzhou.github.io/tutorials
And he also refers to the work of OpenAI and DeepMind.

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
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym, gym_OptClang
import random, threading, queue, operator, os, sys

EP_MAX = 1000
N_WORKER = 5                # parallel workers
GAMMA = 0.9                 # reward discount factor
A_LR = 0.0001               # learning rate for actor
C_LR = 0.0002               # learning rate for critic
MIN_BATCH_SIZE = 24         # minimum batch size for updating PPO
UPDATE_STEP = 10            # loop update operation n-steps
EPSILON = 0.2               # for clipping surrogate objective


class PPO(object):
    def __init__(self, env):
        tf.reset_default_graph()
        self.S_DIM = len(env.observation_space.low)
        self.A_DIM = env.action_space.n
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, self.S_DIM], 'state')

        # critic
        l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
        self.v = tf.layers.dense(l1, 1)
        self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        self.advantage = self.tfdc_r - self.v
        self.closs = tf.reduce_mean(tf.square(self.advantage))
        self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # actor
        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        self.sample_op = tf.squeeze(pi.sample(1), axis=0)  # operation of choosing action
        self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, self.A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
        ratio = pi.prob(self.tfa) / (oldpi.prob(self.tfa) + 1e-5)
        surr = ratio * self.tfadv                       # surrogate loss

        self.aloss = -tf.reduce_mean(tf.minimum(        # clipped surrogate objective
            surr,
            tf.clip_by_value(ratio, 1. - EPSILON, 1. + EPSILON) * self.tfadv))

        self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)
        self.sess.run(tf.global_variables_initializer())

    def update(self):
        global GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():
            if GLOBAL_EP < EP_MAX:
                UPDATE_EVENT.wait()                     # wait until get batch of data
                self.sess.run(self.update_oldpi_op)     # copy pi to old pi
                data = [QUEUE.get() for _ in range(QUEUE.qsize())]      # collect data from all workers
                data = np.vstack(data)
                s, a, r = data[:, :self.S_DIM], data[:, self.S_DIM: self.S_DIM + self.A_DIM], data[:, -1:]
                adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
                # update actor and critic in a update loop
                [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(UPDATE_STEP)]
                [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(UPDATE_STEP)]
                UPDATE_EVENT.clear()        # updating finished
                GLOBAL_UPDATE_COUNTER = 0   # reset counter
                ROLLING_EVENT.set()         # set roll-out available

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 200, tf.nn.relu, trainable=trainable)
            mu = 2 * tf.layers.dense(l1, self.A_DIM, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l1, self.A_DIM, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(a, -2, 2)

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]


class Worker(object):
    def __init__(self, WorkerID, Locks, GAME):
        self.wid = WorkerID
        self.env = gym.make(GAME).unwrapped
        self.ppo = PPO(self.env)
        self.SharedLocks = Locks

    def getMostInfluentialState(self, states, ResetInfo):
        retVec = None
        stats = ResetInfo["FunctionUsageDict"]
        if not stats.items():
            '''
            nothing profiled, random select
            '''
            key = random.choice(list(states.keys()))
        else:
            '''
            select the function with the maximum usage
            '''
            key = max(stats.items(), key=operator.itemgetter(1))[0]

        try:
            retVec = states[key]
        except KeyError:
            '''
            Random selection will never come to here.
            This is caused by perf profiled information which does not contain function arguments.
            '''
            print("Using re to search C++ style name\nKey error:\nkey={}\ndict.keys()={}\n".format(key, states.keys()))
            try:
                FunctionList = list(states.keys())
                newFunctionList = []
                rglexKey = key.replace(' ', '')
                for func in FunctionList:
                    newFunctionList.append(func.replace(' ', ''))
                FunctionList = newFunctionList
                for func in FunctionList:
                    matched = re.search(re.escape(func), rglexKey)
                    if matched:
                        retVec = states[matched]
            except e:
                print("RegExp exception\nKey error:\nkey={}\nrglexKey={}\ndict.keys()={}\nmatched={}\nreason={}\n".format(key, rglexKey, states.keys()), matched, e)
        except e:
            print("Unexpected exception\nKey error:\nkey={}\ndict.keys()={}\nreason={}\n".format(key, states.keys()), e)
        if retVec is None:
            retVec = states[random.choice(list(states.keys()))]
        return retVec

    def calcEachReward(self, info):
        pass

    def appendStateRewards(self, buffer_s, buffer_a, buffer_r, states, rewards, action):
        pass

    def calcDiscountedRewards(self, buffer_r, GAMMA):
        pass

    def calcEpisodeReward(self, rewards):
        pass

    def work(self):
        global GLOBAL_EP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():
            QueueLock = self.SharedLocks[0]
            CounterLock = self.SharedLocks[1]
            PlotEpiLock = self.SharedLocks[2]
            states, ResetInfo = self.env.reset()
            EpisodeReward = 0
            buffer_s, buffer_a, buffer_r = [], [], []
            while True:
                if not ROLLING_EVENT.is_set():                  # while global PPO is updating
                    ROLLING_EVENT.wait()                        # wait until PPO is updated
                    buffer_s, buffer_a, buffer_r = [], [], []   # clear history buffer, use new policy to collect data
                '''
                Choose the features from the most inflential function
                '''
                state = self.getMostInfluentialState(states, ResetInfo)
                sys.exit(1)
                action = self.ppo.choose_action(state)
                nextStates, reward, done, info = self.env.step(action)
                '''
                If build failed, skip it.
                '''
                if reward < 0:
                    break

                '''
                Calculate actual rewards for all functions
                '''
                rewards = self.calcEachReward(info)

                '''
                Match the states and rewards
                '''
                self.appendStateRewards(buffer_s, buffer_a, buffer_r, states, rewards, action)

                '''
                Calculate overall reward for plotting
                '''
                EpisodeReward = self.calcEpisodeReward(rewards)

                # add the generated results
                CounterLock.acquire()
                GLOBAL_UPDATE_COUNTER += len(nextStates.keys())
                CounterLock.release()
                if GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE or done:
                    '''
                    Calculate discounted rewards for all functions
                    '''
                    discounted_r = self.calcDiscountedRewards(buffer_r, GAMMA)

                    bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                    buffer_s, buffer_a, buffer_r = [], [], []

                    QueueLock.acquire()
                    QUEUE.put(np.hstack((bs, ba, br)))          # put data in the shared queue
                    QueueLock.release()

                    if GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                        ROLLING_EVENT.clear()       # stop collecting data
                        UPDATE_EVENT.set()          # globalPPO update

                    if GLOBAL_EP >= EP_MAX:         # stop training
                        COORD.request_stop()
                        break
                if done:
                    break
                else:
                    states = nextStates

            # record reward changes, plot later
            PlotEpiLock.acquire()
            if len(GLOBAL_RUNNING_R) == 0:
                GLOBAL_RUNNING_R.append(EpisodeReward)
            else:
                GLOBAL_RUNNING_R.append(GLOBAL_RUNNING_R[-1]*0.9+EpisodeReward*0.1)
            GLOBAL_EP += 1
            PlotEpiLock.release()
            print('{0:.1f}%'.format(GLOBAL_EP/EP_MAX*100), '|W%i' % self.wid,  '|EpisodeReward: %.2f' % EpisodeReward,)


if __name__ == '__main__':
    Game='OptClang-v0'
    # remove worker file list.
    WorkerListLoc = "/tmp/gym-OptClang-WorkerList"
    if os.path.exists(WorkerListLoc):
        os.remove(WorkerListLoc)

    UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
    UPDATE_EVENT.clear()            # not update now
    ROLLING_EVENT.set()             # start to roll out
    # prevent race condition with 3 locks
    #TODO: release all lock when sigterm
    Locks = []
    for i in range(3):
        Locks.append(threading.Lock())
    workers = []
    for i in range(N_WORKER):
        workers.append(Worker(WorkerID=i, Locks=Locks, GAME=Game))

    GLOBAL_UPDATE_COUNTER, GLOBAL_EP = 0, 0
    GLOBAL_PPO = PPO(gym.make(Game).unwrapped)
    GLOBAL_RUNNING_R = []
    COORD = tf.train.Coordinator()
    QUEUE = queue.Queue()           # workers putting data in this queue
    threads = []
    for worker in workers:          # worker threads
        t = threading.Thread(target=worker.work, args=())
        t.start()                   # training
        threads.append(t)
    # add a PPO updating thread
    threads.append(threading.Thread(target=GLOBAL_PPO.update,))
    threads[-1].start()
    COORD.join(threads)

    # plot reward change
    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('Episode'); plt.ylabel('Moving reward'); plt.ion(); plt.show()
