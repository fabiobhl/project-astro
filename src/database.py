#add filepath to path
import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

#file imports
import keys
import feature_derivations
from utils import calculate_profit

#python libraries import
import json
import math
import datetime
import warnings
import uuid
import shutil
import gc

#external libraries import
import numpy as np
import pandas as pd
from binance.client import Client
from scipy import signal
from matplotlib import pyplot as plt
from sklearn import preprocessing
import ta

#ray imports
import ray
from ray import tune
from ray.tune.utils import pin_in_object_store, get_pinned_object
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.hyperopt import HyperOptSearch

#pytorch imports
import torch

class dbidReader():

    def __init__(self, path):
        self.path = f"{path}/dbid.json"

        #load in the dbid
        with open(self.path) as json_file:
            self.dbid = json.load(json_file)
        
    def __getitem__(self, key):
        return self.dbid[key]

    def __setitem__(self, key, item):
        #change the dict in the ram
        self.dbid[key] = item

        #save changes to json file
        with open(self.path, 'w') as fp:
            json.dump(self.dbid, fp,  indent=4)

    def dump(self):
        #save changes to json file
        with open(self.path, 'w') as fp:
            json.dump(self.dbid, fp,  indent=4)

class DataBase():
    """
    Description:
        This is the base Database class, on which every other Database Objects builds upon.
    Arguments:
    """
    def __init__(self, path):
        #save the path
        self.path = path

        #check if the path exists and is a database
        if not os.path.isdir(path):
            raise Exception("The path you chose is not existing")
        if not os.path.isfile(f"{path}/dbid.json"):
            raise Exception("The path you chose is not a DataBase")
        
        #setup dbid
        self.dbid = dbidReader(path=self.path)
        
        #check if database is of type database
        if self.dbid["type"] != self.__class__.__name__:
            raise Exception(f"Your path does not link to a DataBase of type {self.__class__.__name__}")
    
    @classmethod
    def create(cls, save_path, symbol, date_list, candlestick_interval="1m"):
        """
        Description:
            This method creates a DataBase-Folder at a given location with the specified data.           
        Arguments:
            -save_path (string):        The location, where the folder gets created (Note: The name of the folder should be in the save_path e.g: "C:/.../desired_name")
            -symbol (string):           The Cryptocurrency you want to trade (Note: With accordance to the Binance API)
            -date_list (list):          List of datetime.date objects in the form: [[startdate, enddate], [startdate, enddate], ...]
            -candle_interval (string):  On what interval the candlestick data should be downloaded   
        Return:
            - nothing, creates a folder and multiple with multiple files inside
        """
        #check if the specified directory already exists
        if os.path.isdir(save_path):
            raise Exception("Please choose a directory, that does not already exist")
        
        """
        Download the data and add the tas, and add it to the raw_data_list which then is returned
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
            raw_data = client.get_historical_klines(symbol=symbol, interval=candlestick_interval, start_str=startdate, end_str=enddate)
            data = pd.DataFrame(raw_data)

            #clean the dataframe
            data = data.astype(float)
            data.drop(data.columns[[7,8,9,10,11]], axis=1, inplace=True)
            data.rename(columns = {0:'open_time', 1:'open', 2:'high', 3:'low', 4:'close', 5:'volume', 6:'close_time'}, inplace=True)

            #set the correct times
            data['close_time'] += 1
            data['close_time'] = pd.to_datetime(data['close_time'], unit='ms')
            data['open_time'] = pd.to_datetime(data['open_time'], unit='ms')

            #check for nan values
            if data.isna().values.any():
                raise Exception("Nan values in data, please discard this object and try again")
            
            #add the technical analysis data
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                data = ta.add_all_ta_features(data, open='open', high="high", low="low", close="close", volume="volume", fillna=True)

            #drop first 60 rows
            data = data.iloc[60:]

            #reset the index
            data.reset_index(inplace=True, drop=True)

            #append data to list
            raw_data_list.append(data)


        """
        Diff, pct_change or do nothing with the features, to give the agent a better sense of time and add it to the derived_data_list, which is returned in the end
        """
        #create list to append to
        derived_data_list = []

        #mainloop
        for index, df in enumerate(raw_data_list):
            
            #copy the data
            data = pd.DataFrame()
            raw_data = df.copy()

            #get the different lists, to know how to handle them
            pct_change = feature_derivations.pct_change
            diff = feature_derivations.diff
            no_change = feature_derivations.no_change

            #pct_change
            data[pct_change] = raw_data[pct_change].pct_change()

            #diff
            data[diff] = raw_data[diff].diff()

            #no change
            data[no_change] = raw_data[no_change]

            #delete first row
            data = data.iloc[1:]
            raw_data_list[index] = raw_data_list[index].iloc[1:]

            #reset index
            data.reset_index(inplace=True, drop=True)
            raw_data_list[index].reset_index(inplace=True, drop=True)

            derived_data_list.append(data)
        

        """
        create the directory and save the csv's
        """
        #create the directory
        os.mkdir(save_path)

        #save the data to csv's
        for index in range(len(raw_data_list)):
            raw_data_list[index].to_csv(path_or_buf=f"{save_path}/raw_data_{index}", index_label="index")
            derived_data_list[index].to_csv(path_or_buf=f"{save_path}/derived_data_{index}", index_label="index")
        
        #creating the dbid
        dbid = {
            "type": cls.__name__,
            "symbol": symbol,
            "date_list": [[interval[0].strftime("%d/%m/%Y"), interval[1].strftime("%d/%m/%Y")] for interval in date_list],
            "candlestick_interval": candlestick_interval
        }

        #save the dbid
        with open(f"{save_path}/dbid.json", 'w') as fp:
            json.dump(dbid, fp,  indent=4)
        
        return cls(path=save_path)

    def __getitem__(self, indices):
        
        #check if data is available
        if len(os.listdir(self.path)) <= 1:
            raise Exception("There is no data in your database")
        
        #check if indices is tuple of length 4
        if type(indices) != tuple or len(indices) != 4:
            raise Exception("Please make sure your keys are in the form: [datatype, interval, rows, columns]")

        #check that all the keys are the correct datatype
        if type(indices[0]) != str:
            raise Exception("Make sure your datatype-key is of type string")
        elif type(indices[1]) != slice and type(indices[1]) != int:
            raise Exception("Make sure your interval-key is either of type int or of type slice")
        elif type(indices[2]) != slice and type(indices[2]) != int:
            raise Exception("Make sure your row-key is either of type int or of type slice")
        elif type(indices[3]) != list:
            raise Exception("Make sure your column-key is a list of strings")

        #one date interval access
        if type(indices[1]) == int:
            #check if datatype is available
            if not os.path.isfile(f"{self.path}/{indices[0]}_0"):
                raise Exception("Your chosen datatype is not available in this DataBase")
            #check if date interval is available
            if indices[1] >= len(self.dbid["date_list"]) or indices[1] < 0:
                raise Exception("Your chosen date interval is out of bonds")
        
            #load in the data
            data = pd.read_csv(filepath_or_buffer=f"{self.path}/{indices[0]}_{indices[1]}", usecols=indices[3])

            data = data.iloc[indices[2],:]

            #convert the date columns
            if "close_time" in data.columns:
                data["close_time"]= pd.to_datetime(data["close_time"])
            if "open_time" in data.columns:
                data["open_time"]= pd.to_datetime(data["open_time"])

            return data

        
        #multiple date interval access
        else:
            raise Exception("Mutpile date interval acces has not been implemented yet")

class TrainDataBase(DataBase):
    """
    Description:
        This is a Class that Wraps around a TrainDataBase. This kind of DataBase can be created with the create() method.
    Arguments:
        -symbol (string): The Currencies you want to trade (Binance Code)
        -date_list (list): List of datetime.date objects in the form: [[startdate, enddate], [startdate, enddate], ...]
        -trading_fee (float): The tradingfee of your trading platform
    """
    def __init__(self, path):
        #calling the inheritance
        super().__init__(path)

        #save the labeling methods currently available
        self.labeling_methods = ["feature_extraction"]
        self.auto_labeling_methods = ["feature_extraction"]

        #create the auto-labeling-methods (alm) dictionaries
        self.alm_range = {
            "feature_extraction": {
                "hold_factor": tune.uniform(1,20),
                "threshold": tune.uniform(0,100),
                "distance": tune.uniform(1,10),
                "prominence": tune.uniform(0,10),
                "width": tune.uniform(1,20)
            }
        }

    def get_wrapper(self, feature_list, feature_range=(-1,1), scaling_mode="globally", data_type="derived_data", batch_size=200, window_size=60, labeling_method="feature_extraction", test_percentage=0.2):
        
        #check if TDB was optimized
        if not os.path.isfile(f"{self.path}/labels_0"):
            raise Exception("Before you can get a Wrapper you need to optimize your DataBase atleast once")

        wrapper = TrainDataBaseWrapper(path=self.path, feature_list=feature_list, feature_range=feature_range, scaling_mode=scaling_mode,
                                       data_type=data_type, batch_size=batch_size, window_size=window_size, labeling_method=labeling_method,
                                       test_percentage=test_percentage)
        
        return wrapper

    def _labeling(self, labeling_method):

        #check if tdb was optimized
        if len(self.dbid["alm_optimal"]) == 0:
            raise Exception("You have to optimize your TrainDataBase before you can label your data")
        
        if labeling_method in self.labeling_methods:
            #get the labeling method
            labeler = getattr(self, f"_{labeling_method}_labeling")
            #get the label_list
            ret_list = labeler(parameter_dict=self.dbid["alm_optimal"][labeling_method]["parameters"])
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
        for index in range(len(self.dbid["date_list"])): 
            
            #load in the data
            df = self["raw_data", index, :, feature_list]

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
            ret_list.append(data["bsh"])
        
        return ret_list

    def optimize(self, trial_amount, search_algo="bayesopt", max_workers=8, trading_fee=0.075, verbose=1):
        """
        Description:
            This method goes through all labeling methods in self.automatic_labeling_methods and optimizes their parameters for maximal specific profit. It is built with ray tune therefore it is highly scalable.
        Arguments:
            - trial_amount (int):       the amount of parameter-combinations that are going to be tested
            - search_algo (string):     the search algorithm with which it's going to get optimized [hyperopt, bayesopt]
            - max_workers (int):        how many parallel tasks it should run
            - trading_fee (float):      Tradingfee of your cryptoexchange
            - verbose (int):            The Level of logging in the console (0, 1, 2)
        Return:
            - nothing
            - updates the alm_optimal dictionary if the new best config is better than the old one
        """

        for index, labeling_method in enumerate(self.auto_labeling_methods):
            
            #get the labeling method
            labeler = getattr(self, f"_{labeling_method}_labeling")

            #function that gets passed to the bayesian optimizer
            def objective(config):
                #get the labels from the specific method
                label_list = labeler(config)

                #load the price data
                price_array = pd.DataFrame()
                for index in range(len(self.dbid["date_list"])):
                    df = self["raw_data", index, :, ["close"]]
                    price_array = pd.concat([price_array, df], axis=0)
                price_array = price_array.to_numpy()

                #create the array that gets passed to the profit calculator
                label_array = pd.concat(label_list, axis=0).to_numpy()
                label_array = np.expand_dims(label_array, axis=1)
                array = np.concatenate([price_array, label_array], axis=1)

                specific_profit, _ = calculate_profit(array, trading_fee)

                tune.report(specific_profit=specific_profit)

            #setup the searchalgo
            if search_algo == "bayesopt":
                search_alg = BayesOptSearch(random_search_steps=trial_amount/10)
                search_alg = ConcurrencyLimiter(search_alg, max_concurrent=max_workers)
            elif search_algo == "hyperopt":
                #get the current best params
                best_config = None
                if labeling_method in self.dbid["alm_optimal"].keys():
                    best_config = [self.dbid["alm_optimal"][labeling_method]["parameters"]]
                
                search_alg = HyperOptSearch(n_initial_points=trial_amount/10, points_to_evaluate=best_config)
                search_alg = ConcurrencyLimiter(search_alg, max_concurrent=max_workers)
            else:
                raise Exception("You chose a Search Algorithm that is not available, please choose from this list: bayesopt, hyperopt")

            #run the optimization
            result = tune.run(objective, config=self.alm_range[labeling_method], metric="specific_profit", search_alg=search_alg, mode="max", num_samples=trial_amount, verbose=verbose)
            
            #save the best config in self.alm_optimal
            if labeling_method not in self.dbid["alm_optimal"].keys() or self.dbid["alm_optimal"][labeling_method]["specific_profit"] < result.best_result["specific_profit"]:
                self.dbid["alm_optimal"][labeling_method] = {"parameters": result.get_best_config(),
                                                             "specific_profit": result.best_result["specific_profit"],
                                                             "trading_fee": trading_fee}
                #save the dbid because of chained settting of items
                self.dbid.dump()

            print("Best Config: ", result.get_best_config())
            print("Best SProfit: ", result.best_result["specific_profit"])

        #get all the labels
        optimal_label_dict = {}
        for index, labeling_method in enumerate(self.auto_labeling_methods):
            
            #get the labeler
            labeler = getattr(self, f"_{labeling_method}_labeling")

            #get the labels
            optimal_label_dict[labeling_method] = labeler(self.dbid["alm_optimal"][labeling_method]["parameters"])

        #save all the optimal labels in a csv
        for index in range(len(self.dbid["date_list"])):
            #create a dataframe
            df = pd.DataFrame()
            
            for i, labeling_method in enumerate(self.auto_labeling_methods):
                #add labels to dataframe
                df[labeling_method] = optimal_label_dict[labeling_method][index]

            #save to csv
            df.to_csv(path_or_buf=f"{self.path}/labels_{index}", index_label="index")

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

    @classmethod
    def create(cls, save_path, symbol, date_list, candlestick_interval="1m"):

        #restrict the length of the datelist to 1
        if len(date_list) > 1:
            raise Exception("A TrainDataBase can only consist of 1 dateinterval")

        obj = super().create(save_path, symbol, date_list, candlestick_interval)

        #add things to the dbid
        obj.dbid["alm_optimal"] = {}

        return obj

class TrainDataBaseWrapper(DataBase):

    def __init__(self, path, feature_list, feature_range, scaling_mode, data_type, batch_size, window_size, labeling_method, test_percentage, device=None):
        #save the path
        self.path = path
        #setup dbid
        self.dbid = dbidReader(path=self.path)
        #get the device
        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        #save the variables
        self.feature_list = feature_list
        self.feature_range = feature_range
        self.scaling_mode = scaling_mode
        self.data_type = data_type
        self.batch_size = batch_size
        self.window_size = window_size
        self.labeling_method = labeling_method
        self.test_percentage = test_percentage

        #get id
        self.id = uuid.uuid1()

        #create temporary folder
        os.mkdir(f"{self.path}/{self.id}")

        #prepare the data
        self._prepare_data()

        #load in the mmap
        self.train_data = np.load(file=f"{self.path}/{self.id}/train_data.npy", mmap_mode="r")
        self.train_labels = np.load(file=f"{self.path}/{self.id}/train_labels.npy", mmap_mode="r")
        self.test_data = np.load(file=f"{self.path}/{self.id}/test_data.npy", mmap_mode="r")
        self.test_labels = np.load(file=f"{self.path}/{self.id}/test_labels.npy", mmap_mode="r")
    
    def _prepare_data(self):
        
        #get the labels
        labels = self["labels", 0, :, [self.labeling_method]]

        #select the features
        data = self[self.data_type, 0, :, self.feature_list].to_numpy()

        if self.scaling_mode == "globally":
            #scale
            scaler = preprocessing.MinMaxScaler(feature_range=self.feature_range, copy=False)
            scaler.fit_transform(data)
        else:
            raise Exception("Your scaling mode has not been implemented yet")
            
        #roll the data (rolling window)
        shape = data.shape
        length = shape[0]

        #flatten the array
        data = data.flatten()

        features_amount = len(self.feature_list)
        #amount of elements in one window
        window_elements = self.window_size*features_amount
        #amount of elements to skip in next window
        elements_skip = features_amount
        #amount of windows to generate
        windows_amount = length-self.window_size+1

        #define the indexer
        indexer = np.arange(window_elements)[None, :] + elements_skip*np.arange(windows_amount)[:, None]

        #get the rolling windows
        windows = data[indexer].reshape(windows_amount, self.window_size, features_amount)
        
        #cut first (window_size-1) elements from labels
        labels = labels.iloc[self.window_size-1:].to_numpy()

        #balance (needs to be implemented)

        #batch the windows
        batches_amount = math.floor(windows.shape[0]/self.batch_size)
        windows = windows[0:batches_amount*self.batch_size,:,:]
        batches = windows.reshape(batches_amount, self.batch_size, self.window_size, features_amount)

        #batch the labels
        labels = labels[0:batches_amount*self.batch_size]
        labels = labels.reshape(batches_amount, self.batch_size)

        #split into train/test data
        train_amount = batches_amount - math.floor(batches_amount*self.test_percentage)

        train_data = batches[0:train_amount]
        test_data = batches[train_amount:]

        train_labels = labels[0:train_amount]
        test_labels = labels[train_amount:]

        #save into folder
        np.save(f"{self.path}/{self.id}/train_data",train_data)
        np.save(f"{self.path}/{self.id}/test_data",test_data)
        np.save(f"{self.path}/{self.id}/train_labels",train_labels)
        np.save(f"{self.path}/{self.id}/test_labels",test_labels)

    def train(self):

        for i in range(self.train_data.shape[0]):
            data = np.array(self.train_data[i])
            labels = np.array(self.train_labels[i])
            
            yield torch.tensor(data).to(self.device), torch.tensor(labels).to(self.device)
    
    def test(self):

        for i in range(self.test_data.shape[0]):
            data = np.array(self.test_data[i])
            labels = np.array(self.test_labels[i])

            yield torch.tensor(data).to(self.device), torch.tensor(labels).to(self.device)

    def __del__(self):
        del self.train_data
        del self.train_labels
        del self.test_data
        del self.test_labels
        shutil.rmtree(f"{self.path}/{self.id}")

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

    @classmethod
    def create(self, save_path, symbol, date_list, candlestick_interval="1m"):
        #restrict the length of the datelist to 1
        if len(date_list) > 1:
            raise Exception("A TrainDataBase can only consist of 1 dateinterval")

        obj = super().create(save_path, symbol, date_list, candlestick_interval)

        #add things to the dbid
        obj.dbid["alm_optimal"] = {}

        return obj

if __name__ == "__main__":
    pass