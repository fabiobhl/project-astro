import torch.nn as nn
import torch
from agents.nnagent import BasicNNAgent
from src.environment import LiveEnvironment
from datetime import datetime
import time
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def wait_for_minute():
    while True:
        seconds = int(datetime.now().strftime('%S'))
        if seconds == 0:
            break
    time.sleep(1)
    return

def wait_for_halfminute():
    while True:
        seconds = int(datetime.now().strftime('%S'))
        if seconds == 30:
            break
    return

class network(nn.Module):

    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=10, hidden_size=10, batch_first=True, num_layers=2, dropout=0.2)
        #self.dropout1 = nn.Dropout(0.2)
        self.batchnorm1 = nn.BatchNorm1d(60)
        #self.lstm2 = nn.LSTM(input_size=30, hidden_size=30, batch_first=True)
        #self.dropout2 = nn.Dropout(0.2)
        #self.batchnorm2 = nn.BatchNorm1d(60)
        self.out = nn.Linear(in_features=10, out_features=3)

    def forward(self, t):
        #lstm1 layer
        t, waste = self.lstm1(t)
        t = torch.tanh(t)
        #t = self.dropout1(t)
        t = self.batchnorm1(t)

        #lstm2 layer
        #t, waste = self.lstm2(t, waste)
        #t = torch.tanh(t)
        #t = self.dropout2(t)
        #t = self.batchnorm2(t)

        #outputlayer lstm
        t = t[:,-1,:]
        t = self.out(t)
        #t = torch.softmax(t, dim=1)

        return t

features = ['close', 'volume', 'volatility_bbhi', 'volatility_bbli', 'volatility_kchi',
                    'volatility_kcli', 'volatility_dchi', 'volatility_dcli', 'trend_psar_up_indicator',
                    'trend_psar_down_indicator']

print("Building Agent...")
agent = BasicNNAgent("./Neural Nets/models/750")
print("built")

print("Waiting for halfminute...")
wait_for_halfminute()
print("Building Environment...")
env = LiveEnvironment(windowsize=60, symbol="BTCUSDT", feature_list=features, nnprocessing=True)
print("built")

#plotting
fig, ax = plt.subplots()
fig.show()
data = pd.DataFrame()
data['close'] = env.window['close'].iloc[-60:len(env.window)]
data['buy'] = np.nan
data['sell'] = np.nan
data['hold'] = np.nan


#tradingloop
while True:
    print("waiting for minute to pass...")
    wait_for_minute()
    print("updating...")
    update = env.update_environment()
    print("deciding...")
    decision = agent.take_action(update)
    print("Decision: ", decision)

    #plotting
    data.reset_index(drop=True, inplace=True)
    data.loc[len(data)] = [env.window['close'].iloc[-1], None, None, None]
    data = data.iloc[1:]
    if decision == 0:
        data['hold'].iloc[-1] = data['close'].iloc[-1]
    elif decision == 1:
        data['buy'].iloc[-1] = data['close'].iloc[-1]
    elif decision == 2:
        data['sell'].iloc[-1] = data['close'].iloc[-1]

    ax.cla()
    ax.plot(data['close'])
    ax.plot(data['buy'], marker='o', color="green")
    ax.plot(data['sell'], marker='o', color="red")
    ax.plot(data['hold'], marker='o', color="grey")
    fig.canvas.draw()
    plt.pause(0.001)