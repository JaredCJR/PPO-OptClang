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
import json

def InitSharedStorage():
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
    # counters for synchrnization
    SharedCounters = {}
    SharedCounters['ep'] = 0
    SharedCounters['update_counter'] = 0
    SharedCounters['running_reward'] = []
    SharedCounters['overall_speedup'] = []
    # a global dict to access everything
    SharedStorage = {}
    SharedStorage['Events'] = SharedEvents
    SharedStorage['Locks'] = Locks
    SharedStorage['Counters'] = SharedCounters
    # coordinator for threads
    SharedStorage['Coordinator'] = tf.train.Coordinator()
    # workers putting data in this queue
    SharedStorage['DataQueue'] = queue.Queue()
    return SharedStorage

def LoadJsonConfig(path):
    with open(path, 'r') as f:
        data = json.load(f)
        f.close()
    return data

def ColorPrint(color, msg):
    """
    color should be one of the member of Fore.
    ex. Fore.RED
    """
    print(color + msg)
    print(Style.RESET_ALL)

def gen_plot(GraphTitle, InputList):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.plot(InputList)
    plt.title(GraphTitle)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

class EnvCalculator(object):
    def getMostInfluentialState(states, ResetInfo):
        """
        return the most influential features from profiled data.
        (However, it has some random chance to choose other function)
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
            #key = max(Stats.items(), key=operator.itemgetter(1))[0]
            '''
            select the function with probilities which from profiled usage
            '''
            FunctionList = list(states.keys())
            done = False
            # build list of [key, usage]
            UsageList = []
            for name, usage in Stats.items():
                UsageList.append([name, usage])
            # based on the usage, sort it
            sorted(UsageList, key=operator.itemgetter(1), reverse=True)
            NameList = []
            UsageTmpList = []
            for item in UsageList:
                NameList.append(item[0])
                UsageTmpList.append(item[1])
            # 90% based on the usage
            prob = random.uniform(0, 1)
            if prob < 0.9:
                # for python 3.6
                #key = random.choice(NameList, weights=UsageTmpList)[0]
                # for python 3.5
                choiceList = []
                UsageSum = sum(UsageTmpList)
                for index, elem in enumerate(UsageTmpList):
                    choiceList += NameList[index] * int((UsageTmpList[index]/UsageSum)*100)
                key = random.choice(choiceList)
            else:
                key = random.choice(FunctionList)
        try:
            retVec = states[key]
        except KeyError:
            '''
            Random selection will never come to here.
            This is caused by perf profiled information which does not contain the function arguments.
            '''
            try:
                # use RegExp to search C++ style name or ambiguity of arguments.
                done = False
                for cand in NameList:
                    # searching based on the usage order in descending.
                    realKey = EnvCalculator.RegExpSearch(cand, FunctionList)
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

    def RegExpSearch(TargetName, List):
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

    def calcEachReward(newInfo, MeanSigmaDict, Features, oldInfo, oldCycles, FirstEpi=False):
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
            NameMapDict[perfName] = EnvCalculator.RegExpSearch(perfName, AllFunctions)
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
                resetNameMapDict[perfName] = EnvCalculator.RegExpSearch(perfName, resetAllFunctions)
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

        #####    test:only use total cycles as speedup     #####
        reward = (delta_total_cycles / old_total_cycles) * 10
        for FunctionName in AllFunctions:
            rewards[FunctionName] = reward

        ##### Original #####
        """
        '''
        95% of results are in the twice sigma.
        Therefore, 2x is necessary.
        '''
        SigmaRatio = abs_delta_total_cycles/(2*sigma_total_cycles)
        if SigmaRatio < 0.5:
            SigmaRatio *= 0.25
        elif SigmaRatio < 1.0:
            SigmaRatio *= 0.5
        else:
            SigmaRatio *= 2.0
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
            Alpha = 20
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
                    #delta_total_cycles *= -1
                elif isSpeedup == True and delta_total_cycles < 0:
                    Alpha /= Beta
                    #delta_total_cycles *= -1
                reward = Alpha*SigmaRatio*(delta_total_cycles/old_total_cycles)
            else:
                old_function_cycles = old_total_cycles * old_usage
                new_function_cycles = new_total_cycles * new_usage
                delta_function_cycles = old_function_cycles - new_function_cycles
                reward = Alpha*SigmaRatio*(delta_function_cycles/old_function_cycles)
            rewards[FunctionName] = reward
        # return newAllUsageDict to be the "old" for next episode
        """
        return rewards, newAllUsageDict

    def appendStateRewards(buffer_s, buffer_a, buffer_r, states, rewards, action):
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
            '''
            Our function-name matching mechanism may fail sometimes.
            ex. key='btEmptyAlgorithm::~btEmptyAlgorithm()' will fail
            This does not matters a lot, so we skip it now by checking the key existence.
            '''
            if name in rewards:
                buffer_s[name].append(np.asarray(featureList, dtype=np.float32))
                #actionFeature = [0]*34
                #actionFeature[action] = 1
                buffer_a[name].append(action)
                buffer_r[name].append(rewards[name])



    def calcDiscountedRewards(buffer_r, nextObs, ppo):
        """
        return a dict of list of discounted rewards
        {"function-name":[discounted rewards]}
        """
        retDict = {}
        for name, FeatureList in nextObs.items():
            '''
            Get estimated rewards from critic
            '''
            nextOb = np.asarray(FeatureList, dtype=np.float32)
            StateValue = ppo.get_v(nextOb)
            discounted_r = []
            for r in buffer_r[name][::-1]:
                '''
                Calculate discounted rewards
                '''
                StateValue = r + ppo.GAMMA * StateValue
                discounted_r.append(StateValue)
            discounted_r.reverse()
            retDict[name] = discounted_r
        return retDict


    def calcEpisodeReward(rewards):
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


    def getCpuMeanSigmaInfo():
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

    def DictToVstack(buffer_s, buffer_a, buffer_r):
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

    def calcOverallSpeedup(ResetInfo, Info):
        """
        return the overall speedup(formatted float):
        >0: speedup
        <0: slow down
        ex. 0.032
        """
        old = ResetInfo["TotalCyclesStat"]
        new = Info["TotalCyclesStat"]
        speedup = (old-new)/old
        formatted = "%.3f" % speedup
        return float(formatted)

