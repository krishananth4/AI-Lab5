# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
from backend import ReplayMemory

import nn
import model
import backend
import gridworld


import random,util,math
import numpy as np
import copy

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent
      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update
      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)
      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        
        # Dictionaries are used to store data values in key:value pairs.
        self.qvalues = {} # this is a dictionary that stores the q-states with associated q-values
        # the q-states are keys, and the q-values are associated values for the keys
        # self.qvalues = {(s1,a1): val1, (s2,a2): val2, ...} # pseudo code

        "*** YOUR CODE HERE ***"

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        # The (state, action) key is cross-checked in the qvalues dictionary, if the (state, action) key is present in the qvalues dictionary, we return the particular Q-value present
        if (state,action) in self.qvalues: # (state,action) is the key; (s,a) is the key (s,a): 0.5
            # return the associated q-value
            test = 0 # just for testing
            return self.qvalues[(state,action)]
        # Otherwise, if the (state, action) key is not present in the qvalues dictionary, we return a default Q-value of float 0.0
        else:
            test = 0 # just for testing
            # update the dictionary self.qvalues with this unseen (s,a) and initialize its value as 0.0
            self.qvalues[(state,action)] = 0.0
            # return this q-value
            return self.qvalues[(state,action)]     
            

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action) # which is V(state)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        # Get all of the possible legal actions in a particular state using the getLegalActions() method 
        legal_actions = self.getLegalActions(state)
        # If there are no legal actions to take at a particular state, return float 0.0 as the Q-value
        if not legal_actions:
            return 0.0
        # initialize the maxQvalue to -infinity because you want the smallest possible value so that you can work against it
        maxQValue = float('-inf')
        # Iterate over all the legal actions that we have at a particular state
        for action in legal_actions:
            # retrieve the Q-value
            q_value = self.getQValue(state,action)
            # if the current iterations Q-value is greater than the maximum Q-value that is in place...
            if q_value > maxQValue:
                # Set the Q-value as the maximum Q-value 
                maxQValue = q_value
        # return the maximum Q-value as a result of this method
        return maxQValue
        util.raiseNotDefined()

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        # Get all of the possible legal actions in a particular state using the getLegalActions() method 
        legal_actions = self.getLegalActions(state)
        # If there are no legal actions to take at a particular state, return float None as the action
        if not legal_actions:
            return None
        # Get the maximum Q-value among legal actions, which is produced by the computeValueFromQValues() method
        maxQValue = self.computeValueFromQValues(state)
        # Collect all actions that have the maximum Q-value
        bestActions = []
        # Iterate through all the legal actions present at a particular state
        for action in legal_actions:
          # if the Q-value at a particular state and action is the maximum Q-value
          if self.getQValue(state, action) == maxQValue:
            # append the action that we take to the bestActions list (so we store the best particular action an agent can take to maximize the reward)
            bestActions.append(action)
        # Return a randomly chosen action among the best actions
        return random.choice(bestActions)
        # YL: if there are multiple best actions, then choose a random one
        util.raiseNotDefined()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        
        # Do a coinflip
        if util.flipCoin(self.epsilon):
          # If legalActions has items in it
          if legalActions:
            # Choose a random action from the legal list
            action = random.choice(legalActions)
            # (If legalActions was empty its action would not be updated, leaving it at None, which is what gets returned)
        else:
          # Does the best policy action if the coinflip was false
          action = self.getPolicy(state)
        # util.raiseNotDefined()

        # Returns either a [random action/None] or [best action] depending on the coinflip
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here
          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        # The maximum Q-value is extracted by crosschecking all of the possible Q-values by way of all current legal actions in a particular state (which is done through the computeValueFromQValue method)
        max_Qval = self.computeValueFromQValues(nextState)
        # The current Q-value is taken from the getQValue method that has been implemented (This will be considered as the old value when it comes to the Q-Learning Formula)
        current_Qval = self.getQValue(state, action)
        # The updated Q-value is then updated/calculated by using the general formula: Q(s,a) <-- (1 - alpha) Q(s,a) + alpha(reward + discount(max(Q-value)))
        updated_Qval = (1 - self.alpha) * (current_Qval) + (self.alpha) * (reward + self.discount * max_Qval)
        # The method performs the intended process by updating the q value as an action is taken to a particular state
        self.qvalues[(state,action)] = updated_Qval

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action

class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent
       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def final(self, state):
        """Called at the end of each game."""
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
