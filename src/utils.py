#external library imports
import pandas as pd 
import numpy as np

#pytorch import
import torch
from torch.utils.tensorboard import SummaryWriter as DefaultSummaryWriter
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

if __name__ == "__main__":
    pass