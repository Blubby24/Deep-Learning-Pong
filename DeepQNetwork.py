import tensorflow as tf
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input
from tensorflow.keras.optimizers import Adam
import keras
import keras.backend as K
import cv2
import sys
from Pong import Pong
import random
import numpy as np
from collections import deque


class Environment:

    def __init__(self, agent):
        self.gameState = Pong()
        self.agent = agent
        self.newGame()

    def preProcessFrame(self, frame):
        # Convert the frame to grayscale
        frame = np.average(frame, axis=2)
        # Resize the frame
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_NEAREST)
        # Convert from an image to a numpy array
        frame = np.array(frame, dtype=np.uint8)
        return frame

    def takeStep(self):
        self.agent.totalTimeSteps += 1
        if self.agent.totalTimeSteps % 50000 == 0:
            agent.model.save("pong_weights.keras")
            agent.model.save("pong_bot.h5")
            print('\nWeights saved!')

        # Take an action based on what we thought was best after seeing the last state
        frame, reward, terminal = self.gameState.frameStep(self.agent.memory.actions[-1])
        frame = self.preProcessFrame(frame)
        # Make the next state
        nextState = [self.agent.memory.frames[-3], self.agent.memory.frames[-2], self.agent.memory.frames[-1], frame]
        nextState = np.moveaxis(nextState, 0,2) / 255  # We have to do this to get it into keras's goofy format of [batch_size,rows,columns,channels]
        nextState = np.expand_dims(nextState, 0)

        # Get the next action we think we should do
        nextAction = self.agent.getAction(nextState)

        # If the game ends end the episode
        if terminal:
            self.agent.memory.addToMemory(frame, nextAction, reward, terminal)
            return terminal

        # If the game does end add the new memory and attempt to train
        self.agent.memory.addToMemory(frame, nextAction, reward, terminal)

        # 9: If the threshold memory is satisfied, make the agent learn from memory
        if len(self.agent.memory.frames) > self.agent.memLen:
            self.agent.learn()

    def newGame(self):
        frame, _, _ = self.gameState.frameStep()
        frame = self.preProcessFrame(frame)
        for i in range(3):
            self.agent.memory.addToMemory(frame, 0, 0, False)

    def makeGame(self):
        self.__init__(self.agent)

    def playEpisode(self):
        self.makeGame()
        end = False
        while not end:
            end = self.takeStep()


class Memory:

    def __init__(self, maxMemories):
        # Create some queues to store our memories for later training
        self.maxMemories = maxMemories
        self.frames = deque(maxlen=maxMemories)
        self.rewards = deque(maxlen=maxMemories)
        self.actions = deque(maxlen=maxMemories)
        self.terminals = deque(maxlen=maxMemories)

    def addToMemory(self, frame, action, reward, terminal):
        self.frames.append(frame)
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminals.append(terminal)


class Agent:

    def __init__(self, possibleActions, maxMenLen, epsilon, learningRate, gamma, startingMemLen, inputSize=(84, 84, 4), load=True):
        self.memory = Memory(maxMenLen)
        self.possibleActions = possibleActions
        self.epsilon = epsilon
        self.learningRate = learningRate
        # Reward discount factor
        self.gamma = gamma
        self.epsilonDecayFactor = .9 / 100000
        self.lowestEpsilon = 0.05
        self.memLen = startingMemLen
        self.totalLearns = 0
        self.totalTimeSteps = 0
        self.inputSize = inputSize
        self.model = self.buildModel()
        self.model_target = clone_model(self.model)
        self.learns = 0
        if load:
            self.loadWeights()

    def loadWeights(self):
        self.model = keras.saving.load_model("pong_bot.h5")
        #self.model.load_weights("pong_weights.keras", skip_mismatch=True)
        self.model_target = keras.saving.load_model("pong_bot.h5")
        #self.model_target.load_weights("pong_weights.keras", skip_mismatch=True)
        print("Loaded Weights")

    def buildModel(self):
        model = Sequential()
        # Set the input nueron layer
        model.add(Input(self.inputSize))
        # Set the first of 3 convolutional layers. I need to learn more about these
        # For someone else reading this these help break down the image for the network to learn
        # patterns in the image like edges
        model.add(Conv2D(filters=32, kernel_size=(8, 8), strides=4, data_format="channels_last", activation='relu',
                         kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2)))

        model.add(Conv2D(filters=64, kernel_size=(4, 4), strides=2, data_format="channels_last", activation='relu',
                         kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2)))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=1, data_format="channels_last", activation='relu',
                         kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2)))
        # Flatten the data for the rest of the network
        model.add(Flatten())
        # Hidden Layyer
        model.add(Dense(512, activation='relu', kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2)))
        # Output layer. Each node is an output action
        model.add(Dense(len(self.possibleActions), activation='linear'))
        optimizer = Adam(self.learningRate)
        model.compile(optimizer, loss=tf.keras.losses.Huber())
        model.summary()
        return model

    def getAction(self, state):
        odds = np.random.rand()
        # Take a random exploratory action
        if odds < self.epsilon:
            return random.sample(self.possibleActions, 1)[0]

        # Take a greedy action
        #print(self.model.predict(state))
        #print(self.possibleActions[np.argmax(self.model.predict(state))])
        return self.possibleActions[np.argmax(self.model.predict(state))]

    def checkValid(self, index):
        return not (self.memory.terminals[index - 3] or self.memory.terminals[index - 2]
                    or self.memory.terminals[index - 1] or self.memory.terminals[index])

    def learn(self):
        # First we need to create a batch
        # We will be using batches of 32 each
        states = []
        nextStates = []
        actions = []
        next_rewards = []
        terminalFlags = []
        while len(states) < 32:
            index = np.random.randint(4, len(self.memory.frames) - 1)
            if self.checkValid(index):
                state = [self.memory.frames[index], self.memory.frames[index - 1], self.memory.frames[index - 2]
                    , self.memory.frames[index - 3]]
                state = np.moveaxis(state, 0, 2) / 255
                next_state = [self.memory.frames[index - 2], self.memory.frames[index - 1], self.memory.frames[index],
                              self.memory.frames[index + 1]]
                next_state = np.moveaxis(next_state, 0, 2) / 255

                states.append(state)
                nextStates.append(next_state)
                actions.append(self.memory.actions[index])
                next_rewards.append(self.memory.rewards[index + 1])
                terminalFlags.append(self.memory.terminals[index + 1])

        # What does the model think we should do in a state
        labels = self.model.predict(np.array(states))
        # What does the model think we should do in the next state
        next_state_values = self.model_target.predict(np.array(nextStates))

        # At each of the states assign a value to the action we took
        for i in range(32):
            action = self.possibleActions.index(actions[i])
            labels[i][action] = next_rewards[i] + (not terminalFlags[i]) * self.gamma * max(next_state_values[i])

        # Train the model based on this batch
        self.model.fit(np.array(states), labels, batch_size=32, epochs=1, verbose=0)

        # Decrease how greedy we are the longer we train
        if self.epsilon > self.lowestEpsilon:
            self.epsilon -= self.epsilonDecayFactor
        self.learns += 1

        if self.learns % 10000 == 0:
            self.model_target.set_weights(self.model.get_weights())
            print('\nTarget model updated')

agent = Agent(possibleActions=[0, 2, 3], startingMemLen=50000, maxMenLen=750000, epsilon=0.6,
              learningRate=.00025, gamma=0.95)
env = Environment(agent)

for i in range(1000000):
    env.playEpisode()
    print('\nEpisode: ' + str(i))
    print(len(agent.memory.frames))
    print(agent.epsilon)
