import torch
import torch.nn as nn

import numpy as np
import random


class BaseAgent():

    def __init__(self):
        pass

    def action_decision(self, obs):
        return random.randrange(0,3,1)


    def take_action(self, obs):
        #pass the obs to the logger
        #pass the obs to the visualizer

        #return the action of the action_decision function
        return self.action_decision(obs)

class AdvancedAgent(BaseAgent):
    """
    Observation must contain:
    -volatility_bbli
    -volatility_bbhi
    -volatility_kcli
    -volatility_dcli
    -trend_psar_up
    -trend_psar_down
    windowsize must be atleast 15
    """

    def __init__(self):
        self.mode = 1
    
    def action_decision(self, obs):
        #hold is default
        action = 0

        if self.mode == 1: #buymode

            #prepare decision
            if obs['volatility_bbli'].tail(15).sum() > 0:
                buy_signal_bbli = 1
            else:
                buy_signal_bbli = 0
            if obs['volatility_kcli'].tail(15).sum() > 0:
                buy_signal_kcli = 1
            else:
                buy_signal_kcli = 0
            if obs['volatility_dcli'].tail(15).sum() > 0:
                buy_signal_dcli = 1
            else:
                buy_signal_dcli = 0

            #buy1
            if ((obs['trend_psar_up'].tail(1).iloc[0] - obs['trend_psar_up'].tail(2).head(1).iloc[0]) < 0) & ((obs['trend_psar_down'].tail(1).iloc[0] - obs['trend_psar_down'].tail(2).head(1).iloc[0]) == 0.0):
                if (buy_signal_bbli+buy_signal_dcli+buy_signal_kcli) >= 2:
                    action = 1
                    self.mode = 2
            #buy2            
            elif (obs['volatility_bbli'].tail(2).head(1).iloc[0] == 1.0) & (obs['volatility_bbli'].tail(1).iloc[0] == 0.0):
                action = 1
                self.mode = 2

        else: #sellmode

            #prepare two_spacial_bbhi
            two_spacial_bbhi = False
            values = obs['volatility_bbhi'].tail(10).values.tolist()
            is_achieved = False
            seen_one = False
            before = values[0]
            if before == 1:
                seen_one = True
            counter = 0
            for element in values[1:]:
                if is_achieved:
                    if element == 1:
                        two_spacial_bbhi = True
                        break
                    else:
                        pass
                else:
                    if element == 1:
                        seen_one = True
                    if seen_one:
                        if (before == element) & (element == 0.0):
                            counter +=1
                            if counter == 3:
                                is_achieved = True
                        else:
                            counter = 0
                before = element
            
            #sell1
            if (((obs['trend_psar_up'].tail(1).iloc[0] - obs['trend_psar_up'].tail(2).head(1).iloc[0]) == 0) & ((obs['trend_psar_up'].tail(2).head(1).iloc[0] - obs['trend_psar_up'].tail(3).head(1).iloc[0]) != 0)) & ((obs['trend_psar_down'].tail(1).iloc[0] - obs['trend_psar_down'].tail(2).head(1).iloc[0]) != 0.0):
                action = 2
                self.mode = 1
            
            #sell2
            elif obs['volatility_bbhi'].tail(10).sum() >= 4:
                action = 2
                self.mode = 1
            
            #sell3
            elif (obs['volatility_bbhi'].tail(10).sum() >= 2) & two_spacial_bbhi:
                action = 2
                self.mode = 1

        return action

class BasicNNAgent(BaseAgent):

    def __init__(self, networkpath):
        self.network = torch.load(networkpath).to('cuda')
    
    def action_decision(self, obs):
        obs = np.array(obs)
        t = torch.tensor(obs).to('cuda').unsqueeze(dim=0)
        decision = self.network(t)
        return torch.softmax(decision, dim=1).argmax(dim=1).item()