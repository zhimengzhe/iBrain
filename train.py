# -*- coding: utf-8 -*-
# -----------------------------
# 训练程序
# Author: 樊亚磊
# Date: 2017.8.30
# -----------------------------
import cv2
import sys
import numpy as np
from brain import Brain

# preprocess raw image to 80*80 gray image
def preprocess(observation):
    observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
    return np.reshape(observation,(80,80,1))

def trainDqn():
    sys.path.append("game/")  # test
    import wrapped_flappy_bird as game  # test
    # Step 1: init BrainDQN
    actions = 2
    brain = Brain()
    brain = brain.getInstance('dqn', actions)
    # Step 2: init Flappy Bird Game
    flappyBird = game.GameState()
    # Step 3: play game
    # Step 3.1: obtain init state
    action0 = np.array([1,0])  # do nothing
    observation0, reward0, terminal = flappyBird.frame_step(action0)
    observation0 = cv2.cvtColor(cv2.resize(observation0, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, observation0 = cv2.threshold(observation0,1,255,cv2.THRESH_BINARY)
    brain.setInitState(observation0)

    # Step 3.2: run the game
    while 1!= 0:
        action = brain.getAction()
        nextObservation,reward,terminal = flappyBird.frame_step(action)
        nextObservation = preprocess(nextObservation)
        brain.setPerception(nextObservation,action,reward,terminal)

def trainCnn():
    brain = Brain()
    brain = brain.getInstance('cnn', 'log/cat_vs_dog/train/')
    train_dir = 'data/cat_vs_dog/train/'
    from info import Info
    info = Info()
    train_batch, train_label_batch = info.getImages(train_dir, {'cat' : 0, 'dog' : 1})
    brain.trainCnnNetwork(train_batch, train_label_batch)

def OneCnn():
    brain = Brain()
    brain = brain.getInstance('cnn', 'log/cat_vs_dog/train/')
    from info import Info
    info = Info()
    image_info = info.getOneImage('data/cat_vs_dog/test/dog.9712.jpg')
    brain.evaluateOneImage(image_info, 2, {'cat' : 0, 'dog' : 1})

def trainLstm():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    brain = Brain()
    brain = brain.getInstance('lstm', '')
    brain.trainLstmNetwork(mnist)

def main():
    OneCnn()

if __name__ == '__main__':
	main()