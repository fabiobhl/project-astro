#add filepath to path
import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

#external library imports
import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
from collections import OrderedDict, namedtuple
from itertools import product

#python libraries import
import math
import pickle

#pytorch import
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter as DefaultSummaryWriter
import torch.nn.functional as F
from torch.utils.tensorboard.summary import hparams


class SummaryWriter(DefaultSummaryWriter):
    def add_hparams(self, hparam_dict, metric_dict):
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict)

        logdir = self._get_file_writer().get_logdir()
        
        with SummaryWriter(log_dir=logdir) as w_hp:
            w_hp.file_writer.add_summary(exp)
            w_hp.file_writer.add_summary(ssi)
            w_hp.file_writer.add_summary(sei)
            for k, v in metric_dict.items():
                w_hp.add_scalar(k, v)

def calculate_profit(input_array, trading_fee):
    """
    Description:
        This function takes a 2xn numpy array, where the first column consists of n prices and the second column consists of n, corresponding labels (0: Hold, 1: Buy, 2: Sell)
    Arguments:
        - input_array (2xn np.array):
            0: prices
            1:labels
        - trading_fee (float): The trading fee of the market in percentage!!
    Return:
        - specific profit (float): the specific profit, calculated from the labels
        - ret_array (5xn np.array): 
            0: prices
            1: the price if the action was hold
            2: the price if the action was buy
            3: the price if the action was sell
            4: labels
    """
    
    #check if the array has the correct shape:
    if len(input_array.shape) != 2 or input_array.shape[1] != 2:
        raise Exception("Your provided input_array does not satisfy the correct shape conditions")
    
    #convert trading fee from percentage to decimal
    trading_fee = trading_fee/100

    #extend the input_array
    output_array = np.zeros(shape=(input_array.shape[0], 5))
    output_array[:,0] = input_array[:,0]
    output_array[:,1] = np.nan
    output_array[:,2] = np.nan
    output_array[:,3] = np.nan
    output_array[:,4] = input_array[:,1]

    #create the specific_profit variable
    specific_profit = 0

    #set the mode to buy
    mode = 'buy'

    #calculate the profit
    for i in range(0, output_array.shape[0]):
        #get the pred
        pred = output_array[i,4]

        #save the action
        if pred == 0:
            output_array[i, 1] = output_array[i, 0]
            action = 'hold'
        elif pred == 1:
            output_array[i, 2] = output_array[i, 0]
            action = 'buy'
        elif pred == 2:
            output_array[i, 3] = output_array[i, 0]
            action = 'sell'

        #do the trading
        if mode == 'buy' and action == 'buy':
            tc_buyprice = output_array[i, 0]                                     #tc = tradingcoin
            """
            brutto_tcamount = trading_amount/tc_buyprice
            netto_tcamount = brutto_tcamount - brutto_tcamount*trading_fee
            """
            mode = 'sell'

        elif mode == 'sell' and action == 'sell':
            tc_sellprice = output_array[i, 0]
            """
            brutto_bcamount = tc_sellprice*netto_tcamount                        #bc = basecoin
            netto_bcamount = brutto_bcamount - brutto_bcamount*trading_fee
            localprofit = netto_bcamount-trading_amount
            """
            local_specific_profit = (tc_sellprice/tc_buyprice)*(1-trading_fee)*(1-trading_fee)-1
            specific_profit += local_specific_profit
            mode = 'buy'
    
    return specific_profit, output_array

class RunBuilder():
    @staticmethod
    def get_runs(params):
        
        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))
        
        return runs
    
    @staticmethod
    def get_runs_iterator(params):
        
        run = namedtuple('Run', params.keys())

        for v in product(*params.values()):
            yield run(*v)

class Epoch():
    """
    Class for managing data for 1 epoch, in the RunManager
    """
    def __init__(self):
        #permanent variables
        self.count = -1

        #non permanent variables
        self.train_loss = 0
        self.train_num_correct = 0
        self.test_num_correct = 0
        self.test_choices = torch.Tensor()
    
    def reset_permanent(self):
        self.count = -1
        
    def reset_nonpermanent(self):
        self.train_loss = 0
        self.train_num_correct = 0
        self.test_num_correct = 0
        self.test_choices = torch.Tensor()

class Run():
    """
    Class for managing data for 1 run, in the RunManager
    """
    def __init__(self):
        #permanent variables
        self.count = -1

        #reset variables (get reset after every run)
        self.hyperparameters = None
        self.best_test_accuracy = 0
        self.best_specific_profit_stability = -100

    def reset_permanent(self):
        self.count = -1
    
    def reset_nonpermanent(self):
        self.hyperparameters = None
        self.best_test_accuracy = 0
        self.best_specific_profit_stability = -100

class RunManager():

    def __init__(self):
        self.epoch = Epoch()            #for managing data of 1 epoch
        self.run = Run()                #for managing data of 1 run
        self.tb = None                  #the tensorboard writer

    def begin_run(self, logfolder_name, run, network, example_data):
        #reset the permanent parameters of epoch
        self.epoch.reset_permanent()

        #save hyperparameters and update runcount
        self.run.hyperparameters = run
        self.run.count += 1

        #create the tb for the run and save the graph of the network
        self.tb = SummaryWriter(log_dir=f"C:/Users/fabio/Desktop/projectastro/pretrain_runs/{logfolder_name}/{self.run.hyperparameters}")
        self.tb.add_graph(network, input_to_model=example_data)

    def end_run(self, run):
        #save the hyperparameters
        metrics = {
            "ZMax Test Accuracy": self.run.best_test_accuracy,
            "ZMax Specific Profit Stability": self.run.best_specific_profit_stability
        }
        self.tb.add_hparams(hparam_dict=dict(run._asdict()), metric_dict=metrics)

        #reset non permanent run varibles
        self.run.reset_nonpermanent()

        #close the tensorboard
        self.tb.flush()
        self.tb.close()

    def begin_epoch(self):
        #update epoch count
        self.epoch.count += 1

    def end_epoch(self):
        #reset the non permanent epoch variables
        self.epoch.reset_nonpermanent()
    
    def begin_training(self):
        pass

    def end_training(self, num_train_samples):
        #calculate the metrics
        loss = (self.epoch.train_loss/num_train_samples)*1000
        accuracy = (self.epoch.train_num_correct / num_train_samples)*100

        #add the metrics to the tensorboard
        self.tb.add_scalar('Train Loss', loss, self.epoch.count)
        self.tb.add_scalar('Train Accuracy', accuracy, self.epoch.count)

    def begin_testing(self):
        pass

    def end_testing(self, num_test_samples, performance_data=None, trading_activity_interval=(60,120)): 
        #calculate the metrics
        accuracy = (self.epoch.test_num_correct / num_test_samples) * 100
        specific_profit = performance_data["specific_profit"]
        specific_profit_rate = performance_data["specific_profit_rate"]
        specific_profit_stability = performance_data["specific_profit_stability"]

        #update best variables
        self.run.best_test_accuracy = accuracy if (self.run.best_test_accuracy < accuracy) else self.run.best_test_accuracy
        self.run.best_specific_profit_stability = specific_profit_stability if (self.run.best_specific_profit_stability < specific_profit_stability) else self.run.best_specific_profit_stability

        #add the metrics to the tensorboard
        self.tb.add_scalar('Test Accuracy', accuracy, self.epoch.count)
        self.tb.add_histogram("Choices", self.epoch.test_choices, self.epoch.count)

        #add the performance data
        self.tb.add_scalar('Specific Profit', specific_profit, self.epoch.count)
        self.tb.add_scalar('Specific Profit Rate', specific_profit_rate, self.epoch.count)
        self.tb.add_scalar('Specific Profit Stability', specific_profit_stability, self.epoch.count)

        #Specific Profit per Interval
        amount_of_intervals = len(performance_data["interval_infos"])
        fig, ax = plt.subplots(nrows=2)

        for i in range(amount_of_intervals):
            
            y = performance_data["trading_per_interval"][i][:,5].astype(np.double)
            mask = np.isfinite(y)
            x = np.arange(0,len(y),1)
            ax[0].plot(x[mask], y[mask], label=f"{i}")


            y = performance_data["trading_per_interval"][i][:,6].astype(np.double)
            mask = np.isfinite(y)
            x = np.arange(0,len(y),1)
            ax[1].plot(x[mask], y[mask], label=f"{i}", drawstyle="steps")

        ax[0].set_title("SProfit", fontsize="7")
        ax[1].set_title("Accumulated SProfit", fontsize="7")
        ax[0].tick_params(labelsize=7)
        ax[1].tick_params(labelsize=7)
        ax[0].set_xlim(left=0)
        ax[1].set_xlim(left=0)
        fig.legend()
        fig.tight_layout()
        
        self.tb.add_figure("SProfit per Interval", fig, self.epoch.count)

        #trading Activity per Interval
        amount_of_intervals = len(performance_data["interval_infos"])
        fig, ax = plt.subplots(nrows=math.ceil(amount_of_intervals/2), ncols=2)

        tas = trading_activity_interval[0]
        tae = trading_activity_interval[1]

        for i in range(amount_of_intervals):
            
            ax[math.floor(i/2), i%2].plot(performance_data["trading_per_interval"][i][tas:tae,0], color="black")
            ax[math.floor(i/2), i%2].plot(performance_data["trading_per_interval"][i][tas:tae,1], marker='o', linestyle="", color="gray", markersize=4)
            ax[math.floor(i/2), i%2].plot(performance_data["trading_per_interval"][i][tas:tae,2], marker='o', linestyle="", color="green", markersize=4)
            ax[math.floor(i/2), i%2].plot(performance_data["trading_per_interval"][i][tas:tae,3], marker='o', linestyle="", color="red", markersize=4)

            title = f"{i}" + f" M: {performance_data['interval_infos'][i]['movement']}, L: {performance_data['interval_infos'][i]['duration']}, D: {performance_data['interval_infos'][i]['date_interval']}"
            ax[math.floor(i/2), i%2].set_title(title, fontsize="7")
            ax[math.floor(i/2), i%2].tick_params(labelsize=7)
            fig.tight_layout()
        
        self.tb.add_figure("Trading Activity", fig, self.epoch.count)


    def track_train_metrics(self, loss, preds, labels):
        #track train loss
        self.epoch.train_loss += loss

        #track train num correct
        self.epoch.train_num_correct += self._get_num_correct(preds, labels)

    def track_test_metrics(self, preds, labels):
        #track test num correct
        self.epoch.test_num_correct += self._get_num_correct(preds, labels)

        #track choice distribution
        self.epoch.test_choices = torch.cat((self.epoch.test_choices, preds.argmax(dim=1).to('cpu')), dim=0)

    @torch.no_grad()
    def _get_num_correct(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()


if __name__ == "__main__":
    pass