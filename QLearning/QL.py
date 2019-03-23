import numpy as np
import pandas as pd

class QLearningTable:
    def __init__(self, actions, learningRate = 0.02, rewardDecay = 0.9, eGreed = 0.9):
        self.actions = actions  # a list for different actions
        self.lr = learningRate
        self.gamma = rewardDecay
        self.epsilon = eGreed
        self.q_table = pd.DataFrame(columns = self.actions, dtype=np.float64)


    def chooseAction(self, state):
        self.check_state_exist(state)

        if np.random.uniform() < self.epsilon:
            state_action = self.q_table.loc[state, :]
            # reindex in case of same reward for different state_action
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.idxmax()
        else:
            action = np.random.choice(self.actions)

        return action


    def learn(self, state, action, reward, state_):  # state_ : state after take last action
        self.check_state_exist(state_)

        qPredict = self.q_table.loc[state, action]
        if state_ != "teminal":
            qTarget = reward + self.gamma * self.q_table.loc[state_, :].max()  # key expression
        else:
            qTarget = reward
        self.q_table.loc[state, action] += self.lr * (qTarget - qPredict)


    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index = self.q_table.columns,
                    name = state
                )
            )
