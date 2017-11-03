# -*- coding: utf-8 -*-
# -----------------------------
# 核心程序
# Author: 樊亚磊
# Date: 2017.8.29
# -----------------------------

import tensorflow as tf
import numpy as np
import random
import sys
from sys import path
from collections import deque

# Hyper Parameters:
FRAME_PER_ACTION = 1
GAMMA = 0.99  # decay rate of past observations
OBSERVE = 100.  # timesteps to observe before training
EXPLORE = 200000.  # frames over which to anneal epsilon
FINAL_EPSILON = 0  # 0.001 # final value of epsilon
INITIAL_EPSILON = 0  # 0.01 # starting value of epsilon
REPLAY_MEMORY = 50000  # number of previous transitions to remember
BATCH_SIZE = 32  # size of minibatch
UPDATE_TIME = 100

# def getInstance(cls, *args, **kw):
#     instances = {}
#     def singleton(type, action):
#         if cls not in instances:
#             instances[cls] = cls(*args, **kw)
#         return instances[cls]
#     return singleton
#
# @getInstance
class Brain:

    def __init__(self):
        pass

    def getInstance(self, type = 'dqn', action = ''):
        type = type.lower()
        if type == 'dqn':
            from neuron.dqn import Dqn
            self.cls = Dqn(action)
            #cls = __import__(type.capitalize(), fromlist=['neuron.' + type.lower()])
        elif type == 'cnn':
            from neuron.cnn import Cnn
            self.cls = Cnn(action)
        elif type == 'lstm':
            from neuron.lstm import Lstm
            self.cls = Lstm()
        return self.cls