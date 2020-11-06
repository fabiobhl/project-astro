#external library imports
import pandas as pd 
import numpy as np 
import ta
from binance.client import Client
from sklearn import preprocessing
from matplotlib import pyplot as plt
from scipy import signal
from collections import OrderedDict, namedtuple
from itertools import product

#python libraries import
import datetime
import time
import math
import pickle
import random
import warnings

#pytorch import
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter as DefaultSummaryWriter
import torch.nn.functional as F
from torch.utils.tensorboard.summary import hparams

#tune imports
import ray
from ray import tune
from ray.tune.utils import pin_in_object_store, get_pinned_object
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.hyperopt import HyperOptSearch

#"keys" file import
import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
import keys

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

class DataBase():
    """
    Description:
        This is the base Database class, on which every other Database Objects builds upon.
    Arguments:
        -symbol:        The Currencies you want to trade
        -date_list:     List of datetime.date objects in the form: [[startdate, enddate], [startdate, enddate], ...]
    """
    def __init__(self, symbol, date_list):
        self.symbol = symbol
        self.date_list = date_list
        self.raw_data = self._download(self.date_list)
        self.derived_data = self._derive()

    def _download(self, date_list):
        """
        Download the data and add the tas, at the end save all the dataframe in a list and return
        """
        #create the client
        client = Client(api_key=keys.key, api_secret=keys.secret)

        #create list to append to
        raw_data_list = []

        #mainloop
        for timespan in date_list:

            #get the dates
            startdate = timespan[0].strftime("%d %b, %Y")
            enddate = timespan[1].strftime("%d %b, %Y")

            #download the data and safe it in a dataframe
            raw_data = client.get_historical_klines(symbol="BTCUSDT", interval='1m', start_str=startdate, end_str=enddate)
            data = pd.DataFrame(raw_data)

            #clean the dataframe
            data = data.astype(float)
            data.drop(data.columns[[7,8,9,10,11]], axis=1, inplace=True)
            data.rename(columns = {0:'open_time', 1:'open', 2:'high', 3:'low', 4:'close', 5:'volume', 6:'close_time'}, inplace = True)

            #set the correct times
            data['close_time'] += 1
            data['close_time'] = pd.to_datetime(data['close_time'], unit='ms')
            data['open_time'] = pd.to_datetime(data['open_time'], unit='ms')

            #check for nan values
            if data.isna().values.any():
                raise Exception("Nan values in data, please discard this object and try again")
            
            #add the technical analysis data
            data = ta.add_all_ta_features(data, open='open', high="high", low="low", close="close", volume="volume", fillna=True)

            #drop first 60 rows
            data = data.iloc[60:]

            #reset the index
            data.reset_index(inplace=True, drop=True)

            #append data to list
            raw_data_list.append(data)
        
        return raw_data_list

    def _derive(self):
        """
        Diff, pct_change or do nothing with the features, to give the agent a better sense of time
        """

        data_list = []

        for index, df in enumerate(self.raw_data):

            data = pd.DataFrame()
            raw_data = df.copy()

            pct_change = ["open", "high", "low", "close", "volume_adi", "volume_obv", "volume_nvi", "volume_vwap", "volatility_atr",
                          "volatility_bbm", "volatility_bbh", "volatility_bbl", "volatility_kcc", "volatility_kch", "volatility_kcl",
                          "trend_sma_fast", "trend_sma_slow", "trend_ema_fast", "trend_ema_slow", "trend_adx", "trend_mass_index",
                          "trend_ichimoku_conv", "trend_ichimoku_base", "trend_ichimoku_a", "trend_ichimoku_b", "trend_visual_ichimoku_a",
                          "trend_visual_ichimoku_b", "trend_psar_up", "trend_psar_down", "momentum_kama"] 

            diff = ["volume", "volume_cmf", "volume_fi", "momentum_mfi", "volume_em", "volume_sma_em", "volume_vpt",
                    "volatility_bbw", "volatility_bbp", "volatility_kcw", "volatility_kcp", "volatility_dcl", "volatility_dch",
                    "trend_macd", "trend_macd_signal", "trend_adx_pos", "trend_adx_neg", "trend_vortex_ind_pos", "trend_vortex_ind_neg",
                    "trend_trix", "trend_cci", "trend_dpo", "trend_kst", "trend_kst_sig", "trend_aroon_up", "trend_aroon_down", "momentum_rsi",
                    "momentum_tsi", "momentum_uo", "momentum_stoch", "momentum_stoch_signal", "momentum_wr", "momentum_ao", "momentum_roc",
                    "others_dr", "others_dlr", "others_cr"]

            no_change = ["open_time", "close_time", "volatility_bbhi", "volatility_bbli", "volatility_kchi", "volatility_kcli",
                         "trend_macd_diff", "trend_vortex_ind_diff", "trend_kst_diff", "trend_aroon_ind", "trend_psar_up_indicator",
                         "trend_psar_down_indicator"]

            #pct_change
            data[pct_change] = raw_data[pct_change].pct_change()

            #diff
            data[diff] = raw_data[diff].diff()

            #no change
            data[no_change] = raw_data[no_change]

            #delete first row
            data = data.iloc[1:]
            self.raw_data[index] = self.raw_data[index].iloc[1:]

            #reset index
            data.reset_index(inplace=True, drop=True)
            self.raw_data[index].reset_index(inplace=True, drop=True)

            data_list.append(data)
        
        return data_list

class PerformanceAnalyticsDatabase(DataBase):
    """
    Description:
        This is a Database Class for the PerformanceAnalytics class
    Arguments:
        -symbol:        The Currencies you want to trade
        -date_list:     List of datetime.date objects in the form: [[startdate, enddate], [startdate, enddate], ...]
    """
    def __init__(self, symbol, date_list):
        super().__init__(symbol, date_list)


    def get_data(self, feature_list, derived=True, scaled=True, feature_range=(-1,1), batch_size=200, window_size=60):

        if not derived:
            data_list = self.raw_data.copy()
        else:
            data_list = self.derived_data.copy()

        ret_list = []

        for df in data_list:
            
            #select the features
            data = df[feature_list].to_numpy()

            #scale if desired
            scaler = preprocessing.MinMaxScaler(feature_range=feature_range, copy=False)
            scaler.fit_transform(data)

            #sample/batch the data
            shape = data.shape
            length = shape[0]

            #flatten the array
            data = data.flatten()

            features_amount = len(feature_list)
            #amount of elements in one window
            window_elements = window_size*features_amount
            #amount of elements to skip in next window
            elements_skip = features_amount
            #amount of windows to generate
            windows_amount = length-window_size

            #define the indexer
            indexer = np.arange(window_elements)[None, :] + elements_skip*np.arange(windows_amount)[:, None]

            #get the samples
            batches = data[indexer].reshape(windows_amount, window_size, features_amount)
            batches_amount = math.floor(batches.shape[0]/batch_size)
            batches = batches[0:batches_amount*batch_size,:,:]
            batches = batches.reshape(batches_amount, batch_size, window_size, features_amount)

            #append to ret_list
            ret_list.append(batches)

        return ret_list

class TrainDatabase(DataBase):
    """
    Description:
        This class is here to create data, on which you can train AI models.
    Arguments:
        -symbol (string): The Currencies you want to trade (Binance Code)
        -date_list (list): List of datetime.date objects in the form: [[startdate, enddate], [startdate, enddate], ...]
        -trading_fee (float): The tradingfee of your trading platform
    """
    def __init__(self, symbol, date_list, trading_fee=0.075):
        #calling the inheritance
        super().__init__(symbol, date_list)

        #save the trading_fee
        self.trading_fee = trading_fee

        #save the labeling methods currently available
        self.labeling_methods = ["feature_extraction"]
        self.auto_labeling_methods = ["feature_extraction"]
        
        #create the auto labelin methods (alm) dictionaries
        self.alm_range = {
            "feature_extraction": {
                "hold_factor": tune.uniform(1,20),
                "threshold": tune.uniform(0,100),
                "distance": tune.uniform(1,10),
                "prominence": tune.uniform(0,10),
                "width": tune.uniform(1,20)
            }
        }
        self.alm_optimal = {}

        #run the optimization
        self.optimize(trial_amount=100)

        #create the scalerslist
        self.scalers = []

    def get_data(self, feature_list, feature_range=(-1,1), derived=True, batch_size=200, window_size=60, labeling_method="feature_extraction", test_percentage=0.2):
        
        #select derived_data or raw_data
        if derived:
            predata = self.derived_data
        else:
            predata = self.raw_data
        
        #create the lists to save data
        train_data_list = []
        train_labels_list = []
        test_data_list = []
        test_labels_list = []

        #reset the scalerslist
        self.scalers = []

        #get the label_list
        label_list = self._labeling(labeling_method=labeling_method)

        #main data loop
        for index, df in enumerate(predata):

            #select the features
            data = df[feature_list].to_numpy()

            #scale
            self.scalers.append(preprocessing.MinMaxScaler(feature_range=feature_range, copy=False))
            self.scalers[index].fit_transform(data)
            
            #roll the data (rolling window)
            shape = data.shape
            length = shape[0]

            #flatten the array
            data = data.flatten()

            features_amount = len(feature_list)
            #amount of elements in one window
            window_elements = window_size*features_amount
            #amount of elements to skip in next window
            elements_skip = features_amount
            #amount of windows to generate
            windows_amount = length-window_size+1

            #define the indexer
            indexer = np.arange(window_elements)[None, :] + elements_skip*np.arange(windows_amount)[:, None]

            #get the rolling windows
            windows = data[indexer].reshape(windows_amount, window_size, features_amount)
            
            #cut first (window_size-1) elements from labels
            labels = label_list[index].iloc[window_size-1:].to_numpy()

            #balance (needs to be implemented)

            #batch the windows
            batches_amount = math.floor(windows.shape[0]/batch_size)
            windows = windows[0:batches_amount*batch_size,:,:]
            batches = windows.reshape(batches_amount, batch_size, window_size, features_amount)

            #batch the labels
            labels = labels[0:batches_amount*batch_size]
            labels = labels.reshape(batches_amount, batch_size)

            #split into train/test data
            train_amount = batches_amount - math.floor(batches_amount*test_percentage)

            train_data_list.append(batches[0:train_amount])
            test_data_list.append(batches[train_amount:])

            train_labels_list.append(labels[0:train_amount])
            test_labels_list.append(labels[train_amount:])

        #append all the dataframes
        train_data = np.concatenate(train_data_list, axis=0).astype(np.float64)
        train_labels = np.concatenate(train_labels_list, axis=0).astype(np.int64)
        test_data = np.concatenate(test_data_list, axis=0).astype(np.float64)
        test_labels = np.concatenate(test_labels_list, axis=0).astype(np.int64)

        return (train_data, train_labels, test_data, test_labels)

    def _labeling(self, labeling_method):
        
        if labeling_method in self.labeling_methods:
            #get the labeling method
            labeler = getattr(self, f"_{labeling_method}_labeling")
            #get the label_list
            ret_list = labeler(parameter_dict=self.alm_optimal[labeling_method]["parameters"])
        else:
            raise Exception("Your chosen labelingmethod is not available, please try again with another method")

        return ret_list
    
    """
    All the different labeling methods follow:
        Special Syntax:
            -the name of the labeling method must be: "_" + "name" + "_labeling"
        Arguments:
            -parameter_dict (kwargs)
    """
    def _feature_extraction_labeling(self, parameter_dict):
        
        #setup the return list
        ret_list = []

        #the features with which we are going to determine the peaks and lows
        feature_list = ["close", "volume_obv", "volume_cmf", "volume_fi", "momentum_mfi",
                    "volume_vwap", "volatility_bbm", "volatility_bbh", "volatility_bbp",
                    "volatility_kcc", "volatility_kcp", "volatility_dch", "trend_macd",
                    "trend_macd_diff", "trend_ema_fast", "trend_adx_pos", "trend_vortex_ind_diff",
                    "trend_cci", "trend_kst_diff", "momentum_rsi", "momentum_tsi", "momentum_uo",
                    "momentum_stoch_signal", "momentum_wr", "momentum_ao", "momentum_roc"]

        #mainloop
        for index, df in enumerate(self.raw_data): 
            #create the data were are going to work on
            data = pd.DataFrame()
            data["close"] = df["close"]
            data["hold"] = 0
            data["buy"] = 0
            data["sell"] = 0
            data["bsh"] = np.nan

            #labeling loop
            for feature in feature_list:
                
                #get the feature array
                array = np.array(df[feature])

                #find indeces of peaks and lows
                peaks, _ = signal.find_peaks(array, threshold=parameter_dict["threshold"], distance=parameter_dict["distance"], prominence=parameter_dict["prominence"], width=parameter_dict["width"])
                lows, _ = signal.find_peaks(-array, threshold=parameter_dict["threshold"], distance=parameter_dict["distance"], prominence=parameter_dict["prominence"], width=parameter_dict["width"])

                #update scores in the data
                data.iloc[lows, 2] += 1                                 #update the buy column
                data.iloc[peaks, 3] += 1                                #update the sell column
                hold = np.ones(len(data))
                hold[peaks] = 0
                hold[lows] = 0
                data["hold"] += hold/parameter_dict["hold_factor"]      #update the hold column

            #set the right index at bsh
            data["bsh"] = data.iloc[:,1:4].idxmax(axis=1)
            data.loc[data["bsh"] == "hold", "bsh"] = 0 
            data.loc[data["bsh"] == "buy", "bsh"] = 1 
            data.loc[data["bsh"] == "sell", "bsh"] = 2

            #append the labels to the return list
            ret_list.append(data["bsh"].copy())
        
        return ret_list

    def optimize(self, trial_amount, search_algo="bayesopt", max_workers=8, verbose=1):
        """
        Description:
            This method goes through all labeling methods in self.automatic_labeling_methods and optimizes their parameters for maximal specific profit. It is built with ray tune therefore it is highly scalable.
        Arguments:
            - trial_amount (int): the amount of parameter-combinations that are going to be tested
            - search_algo (string): the search algorithm with which it's going to get optimized [hyperopt, bayesopt]
            - max_workers (int): how many parallel tasks it should run
            - verbose (int): The Level of logging in the console (0, 1, 2)
        Return:
            - nothing
            - updates the alm_optimal dictionary if the new best config is better than the old one
            - for new optimizations to be saved, you need to save the object with the .save() method
        """

        for index, labeling_method in enumerate(self.auto_labeling_methods):
            
            #get the labeling method
            labeler = getattr(self, f"_{labeling_method}_labeling")
            
            #get he price data
            price_array = pd.DataFrame()
            for data in self.raw_data:
                df = pd.DataFrame()
                df["close"] = data["close"].copy()
                price_array = pd.concat([price_array, df], axis=0)
            price_array = price_array.to_numpy()

            #store the needed objects in raytune mainprocess
            ray.init()
            labeler_id = pin_in_object_store(labeler)
            price_array_id = pin_in_object_store(price_array)
            calculate_profit_id = pin_in_object_store(calculate_profit)

            #function that gets passed to the bayesian optimizer
            def objective(config):
                
                #get objects from central tune storage
                labeler = get_pinned_object(labeler_id)
                price_array = get_pinned_object(price_array_id)
                calculate_profit = get_pinned_object(calculate_profit_id)

                #get the labels from the specific method
                label_list = labeler(config)

                #create the array tha gets passed to the profit calculator
                label_array = pd.concat(label_list, axis=0).to_numpy()
                label_array = np.expand_dims(label_array, axis=1)
                array = np.concatenate([price_array, label_array], axis=1)

                specific_profit, _ = calculate_profit(array, self.trading_fee)

                tune.report(specific_profit=specific_profit)
                time.sleep(0.1)

            #setup the searchalgo
            if search_algo == "bayesopt":
                search_alg = BayesOptSearch(random_search_steps=trial_amount/10)
                search_alg = ConcurrencyLimiter(search_alg, max_concurrent=max_workers)
            elif search_algo == "hyperopt":
                #get the current best params
                best_config = None
                if labeling_method in self.alm_optimal.keys():
                    best_config = [self.alm_optimal[labeling_method]["parameters"]]
                
                search_alg = HyperOptSearch(n_initial_points=trial_amount/10, points_to_evaluate=best_config)
                search_alg = ConcurrencyLimiter(search_alg, max_concurrent=max_workers)
            else:
                raise Exception("You chose a Search Algorithm that is not available, please choose from this list: bayesopt, hyperopt")

            #run the optimization
            result = tune.run(objective, config=self.alm_range[labeling_method], metric="specific_profit", search_alg=search_alg, mode="max", num_samples=trial_amount, verbose=verbose)
            
            #save the best config in self.alm_optimal
            if labeling_method not in self.alm_optimal.keys() or self.alm_optimal[labeling_method]["specific_profit"] < result.best_result["specific_profit"]:
                self.alm_optimal[labeling_method] = {"parameters": result.get_best_config(),
                                                    "specific_profit": result.best_result["specific_profit"]}

            print(result.get_best_config())
            print(result.best_result["specific_profit"])

    def check_labeling(self, labeling_method="feature_extraction"):
        """
        Description:
            Displays the best current labeling config in a plot
        """
        #get the labels
        label_list = self._labeling(labeling_method="feature_extraction")
        
        #get the price data
        data = pd.DataFrame()
        for index, raw_data in enumerate(self.raw_data):
            df = pd.DataFrame()
            df["close"] = raw_data["close"].copy()
            df["bsh"] = label_list[index]
            data = pd.concat([data, df], axis=0)
        
        data["hold"] = np.nan
        data["buy"] = np.nan
        data["sell"] = np.nan
        data.loc[data["bsh"] == 0, "hold"] = data["close"]
        data.loc[data["bsh"] == 1, "buy"] = data["close"]
        data.loc[data["bsh"] == 2, "sell"] = data["close"]

        fig, ax = plt.subplots()

        ax.plot(data.loc[:,"close"])
        ax.plot(data.loc[:,"hold"], linestyle="", marker="o", color="gray")
        ax.plot(data.loc[:,"buy"], linestyle="", marker="o", color="green")
        ax.plot(data.loc[:,"sell"], linestyle="", marker="o", color="red")

        plt.show()

    def get_amount_trades(self, labeling_method="feature_extraction"):
        """
        Description:
            Counts the amount of positive and negative trades
        Return:
            tuple of length 2 with: (amount_of_pos_trades, amount_of_neg_trades)
        """

        #get the labels
        label_list = self._labeling(labeling_method="feature_extraction")
        
        #get the price data
        data = pd.DataFrame()
        for index, raw_data in enumerate(self.raw_data):
            df = pd.DataFrame()
            df["close"] = raw_data["close"].copy()
            df["bsh"] = label_list[index]
            data = pd.concat([data, df], axis=0)
        
        input_array = data.to_numpy()

        #convert trading fee from percentage to decimal
        trading_fee = self.trading_fee/100

        #extend the input_array
        output_array = np.zeros(shape=(input_array.shape[0], 5))
        output_array[:,0] = input_array[:,0]
        output_array[:,1] = np.nan
        output_array[:,2] = np.nan
        output_array[:,3] = np.nan
        output_array[:,4] = input_array[:,1]

        #create the count variable
        pos_trades = 0
        neg_trades = 0

        #set the mode to buy
        mode = 'buy'

        #calculate the amounts
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
                mode = 'sell'

            elif mode == 'sell' and action == 'sell':
                tc_sellprice = output_array[i, 0]
                local_specific_profit = (tc_sellprice/tc_buyprice)*(1-trading_fee)*(1-trading_fee)-1
                if local_specific_profit >= 0:
                    pos_trades += 1
                else:
                    neg_trades += 1
                mode = 'buy'
        
        return (pos_trades, neg_trades)

    def save(self, object_path):
        """
        Method for saving the object
        """
        pickle.dump(self, open(object_path, "wb"))
    
    @staticmethod
    def load(object_path):
        """
        Alternativ constructor to load a object
        """
        obj = pickle.load(open(object_path, "rb"))
        if isinstance(obj, TrainDatabase):
            return obj
        else:
            raise Exception('You tried to load a file which is not an instance of the TrainDatabase class')

class PerformanceAnalytics(PerformanceAnalyticsDatabase):
    """
    Description:
        This class can be used to determine the performance (profit) of the actor. 
    Constructor Arguments:
        -date_list:         list of dateintervals on which the actor gets tested (need to be Datetime Objects)
        -trade_amount:      the amount of money the actor can deal with
        -trading_fees:      the tradingfees of the tradingplatform
    """
    def __init__(self, symbol, date_list):
        super().__init__(symbol, date_list)
    
    def evaluate_network(self, network, feature_list, feature_range=(-1,1), window_size=60, batch_size=200, trading_fee=0.075, verbose=False):
        """
        Evaluates the performance of the network.

        Arguments:
            network (nn.Module): The network, to be tested.
            feature_list (list): The list of features, that the network is expecting.
            feature_range (tuple): The range, to which the data should be scaled to e.g. (-1,1)
            window_size (int): The size of the rolling window, that the network is expecting,  Default: 60
            batch_size (int): The size of the batches, which get fed to the network
            trading_fee (float): The tradingfee of the coinmarket you are trading on, in % e.g. 0.1%.  Default: 0.075%
            verbose (Bool): If true, the method opens a plot and shows every window
        
        Return:
            Returns a dictionary with all the performance results.
        """
        #convert trading fee to decimals
        trading_fee = trading_fee/100

        #check if there is a cuda gpu available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #move our network to cpu/gpu
        network = network.to(device)
        #get the data
        data = self.get_data(feature_list=feature_list, derived=True, scaled=True, feature_range=feature_range, batch_size=batch_size, window_size=60)

        #create the return dict
        return_dict = {
            "specific_profit": 0,
            "specific_profit_rate": 0,
            "specific_profit_stability": 0,
            "trading_per_interval": [],
            "interval_infos": []
        }

        #mainloop
        for index, array in enumerate(data):
            #setup variables
            mode = 'buy'
            specific_profit = 0
            
            #create bsh array (The array, where the predictions are going to be safed)
            bsh = np.empty((window_size-1))
            bsh[:] = np.nan
            bsh = torch.as_tensor(bsh).to(device)

            #move batches to tensor
            batches = torch.as_tensor(array).to(device)
            
            #getting the predictions
            for i in range(array.shape[0]):
                #feed data to nn
                batch = batches[i]
                prediction = network(batch)
                #update data
                prediction = torch.softmax(prediction, dim=1)
                pred = prediction.argmax(dim=1)
                #append to bsh array
                bsh = torch.cat((bsh, pred),0)
            
            #move bsh to cpu
            bsh = bsh.to('cpu').numpy()

            #append the bsh array to data
            profit_data = pd.DataFrame()
            profit_data["close"] = self.raw_data[index]["close"].copy()
            profit_data["hold"] = np.nan
            profit_data["buy"] = np.nan
            profit_data["sell"] = np.nan
            #shorten the data to predictions length
            profit_data = profit_data.iloc[0:bsh.shape[0],:]
            profit_data["bsh"] = bsh

            if verbose:
                fig, ax = plt.subplots()
                fig.show()

            #convert data to numpy array
            profit_data = profit_data.to_numpy()
            
            #calculate the profit
            for i in range(window_size-1, profit_data.shape[0]):
                #get the pred
                pred = profit_data[i,4]

                #save the action
                if pred == 0:
                    profit_data[i, 1] = profit_data[i, 0]
                    action = 'hold'
                elif pred == 1:
                    profit_data[i, 2] = profit_data[i, 0]
                    action = 'buy'
                elif pred == 2:
                    profit_data[i, 3] = profit_data[i, 0]
                    action = 'sell'

                #do the trading
                if mode == 'buy':
                    if action == 'buy':
                        tc_buyprice = profit_data[i, 0]                                     #tc = tradingcoin
                        """
                        brutto_tcamount = trading_amount/tc_buyprice
                        netto_tcamount = brutto_tcamount - brutto_tcamount*trading_fee
                        """
                        mode = 'sell'

                elif mode == 'sell':
                    if action == 'sell':
                        tc_sellprice = profit_data[i, 0]
                        """
                        brutto_bcamount = tc_sellprice*netto_tcamount                        #bc = basecoin
                        netto_bcamount = brutto_bcamount - brutto_bcamount*trading_fee
                        localprofit = netto_bcamount-trading_amount
                        """
                        local_specific_profit = (tc_sellprice/tc_buyprice)*(1-trading_fee)*(1-trading_fee)-1
                        specific_profit += local_specific_profit
                        mode = 'buy'
            

                if verbose:
                    #get plotting window
                    window = profit_data[i-window_size-1:i]

                    ax.cla()
                    ax.plot(window[:,0])
                    ax.plot(window[:,1], marker='o', color='gray')
                    ax.plot(window[:,2], marker='o', color='green')
                    ax.plot(window[:,3], marker='o', color='red')
                    fig.canvas.draw()
                    plt.pause(0.001)
                    input()

            """
            Calculate and save the metrics
            """
            #specific profit
            return_dict["specific_profit"] += specific_profit
            
            #specific profit rate
            amount_of_hours = (profit_data.shape[0]-window_size)/60
            specific_profit_rate = specific_profit/amount_of_hours
            return_dict["specific_profit_rate"] += specific_profit_rate

            #specific profit stability
            movement = 2*(profit_data[-1,0] - profit_data[0,0])/(profit_data[-1,0] + profit_data[0,0])
            specific_profit_stability = specific_profit_rate/(1+movement)
            return_dict["specific_profit_stability"] += specific_profit_stability

            #interval infos
            interval_info = {
                "movement": round(movement*100,2),
                "duration": round(amount_of_hours,1),
                "date_interval": f"{self.date_list[index][0].strftime('%d/%m/%y')}--{self.date_list[index][1].strftime('%d/%m/%y')}"
            }
            return_dict["interval_infos"].append(interval_info)

            #trading per interval
            return_dict["trading_per_interval"].append(profit_data)
        
        #divide the specific metrics by amount of intervals
        return_dict["specific_profit_rate"] = return_dict["specific_profit_rate"]/len(data)
        return_dict["specific_profit_stability"] = return_dict["specific_profit_stability"]/len(data)

        return return_dict

    def save(self, object_path):
        """
        Method for saving the Randprof object
        """
        pickle.dump(self, open(object_path, "wb"))

    @staticmethod
    def load(object_path):
        """
        Alternativ constructor to load a Randprof object
        """
        obj = pickle.load(open(object_path, "rb"))
        if isinstance(obj, PerformanceAnalytics):
            return obj
        else:
            raise Exception('You tried to load a file which is not an instance of the Randprof class')

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

        #trading Activity per Interval
        amount_of_intervals = len(performance_data["interval_infos"])
        fig, ax = plt.subplots(nrows=math.ceil(amount_of_intervals/2), ncols=2)

        for i in range(amount_of_intervals):
            
            ax[math.floor(i/2), i%2].plot(performance_data["trading_per_interval"][i][tas:tae,0])
            ax[math.floor(i/2), i%2].plot(performance_data["trading_per_interval"][i][tas:tae,1])
            ax[math.floor(i/2), i%2].plot(performance_data["trading_per_interval"][i][tas:tae,2])
            ax[math.floor(i/2), i%2].plot(performance_data["trading_per_interval"][i][tas:tae,3])

            title = f"{i}" + f" M: {performance_data['interval_infos'][i]['movement']}, L: {performance_data['interval_infos'][i]['duration']}, D: {performance_data['interval_infos'][i]['date_interval']}"
            ax[math.floor(i/2), i%2].set_title(title, fontsize="7")
            ax[math.floor(i/2), i%2].tick_params(labelsize=7)
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