import numpy as np
import pandas as pd

class SarsaTable:
    def __init__(self, actions, learningRate = 0.02, rewardDecay = 0.9, eGreed = 0.9):
        self.actions = actions  # a list for different actions
        self.lr = learningRate
        self.gamma = rewardDecay
        self.epsilon = eGreed
        self.qTable = pd.DataFrame(columns = self.actions, dtype=np.float64)


    def chooseAction(self, state):
        self.check_state_exist(state)

        if np.random.uniform() < self.epsilon:
            state_action = self.qTable.loc[state, :]
            # reindex in case of same reward for different state_action
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.idxmax()
        else:
            action = np.random.choice(self.actions)

        return action


    def learn(self, state, action, reward, state_, action_):  # state_ : state after take last action
        self.check_state_exist(state_)

        qPredict = self.qTable.loc[state, action]
        if state_ != "teminal":
            # qObserve = reward + self.gamma * self.qTable.loc[state_, :].max()  # key expression
            qObserve = reward + self.gamma * self.qTable.loc[state_, action_]  # learn after make next decision : action_
        else:
            qObserve = reward

        self.qTable.loc[state, action] += self.lr * (qObserve - qPredict)


    def check_state_exist(self, state):
        if state not in self.qTable.index:
            self.qTable = self.qTable.append(
                pd.Series(
                    [0]*len(self.actions),
                    index = self.qTable.columns,
                    name = state
                )
            )
