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

#dash imports
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objs as go

#pytorch imports
import torch

class dbidReader():
    """
    Description:
        Class which can be used like a dictionary. It writes all the changes to the harddisk.
        You must dump the dbidReader, after you changed a variable in the dictionary through chained access (i.e dbid[alm_optimal][feature_extraction] = 0).
        Otherwise your changed variable wont be saved.
    Arguments:
        -path (string):     Path of the database
    """

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
        -path (string): Path of the Database
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
            -save_path (string):            The location, where the folder gets created (Note: The name of the folder should be in the save_path e.g: "C:/.../desired_name")
            -symbol (string):               The Cryptocurrency you want to trade (Note: With accordance to the Binance API)
            -date_list (list):              List of datetime.date objects in the form: [[startdate, enddate], [startdate, enddate], ...]
            -candlestick_interval (string): On what interval the candlestick data should be downloaded   
        Return:
            - nothing, creates a folder with multiple files inside
        """
        #check if the specified directory already exists
        if os.path.isdir(save_path):
            raise Exception("Please choose a directory, that does not already exist")
        
        """
        Download the data, add the tas
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
        Diff, pct_change
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

            data = data.iloc[indices[2],:].reset_index(drop=True)

            #convert the date columns
            if "close_time" in data.columns:
                data["close_time"]= pd.to_datetime(data["close_time"])
            if "open_time" in data.columns:
                data["open_time"]= pd.to_datetime(data["open_time"])

            return data

        
        #multiple date interval access
        else:
            raise Exception("Mutpile date interval access has not been implemented yet")

class Wrapper(DataBase):
    """
    Description:
        Wrapper is used to wrap around a database and prepare the data to your liking: sampling, rolling, shuffling, scaling, balancing, ...
        A new folder is created for every instance and gets deleted, as soon as the instance goes out of scope.
    Arguments:
        path (string):      Path of the Database you want your wrapper to wrap around
        device (string):    Device on which your Wrapper should work on (i.e. cpu, gpu), if this arguments is not set it uses the gpu, if there is one available
    """

    def __init__(self, path, device=None):
        #save the path
        self.path = path
        #setup dbid
        self.dbid = dbidReader(path=self.path)
        #get the device
        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        #get id
        self.id = uuid.uuid1()
        #create temporary folder
        os.mkdir(f"{self.path}/{self.id}")
    
    def __del__(self):
        shutil.rmtree(f"{self.path}/{self.id}")

class TrainDataBase(DataBase):
    """
    Description:
        This Database can be used for supervised training of time-series models. To create a TrainDataBase, use the create method.
    Arguments:
        -path (string):     The path of your TrainDataBase
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
                "width": tune.uniform(1,20),
                "window_size": tune.uniform(60, 1500)
            }
        }

    def get_wrapper(self, feature_list, feature_range=(-1,1), scaling_mode="globally", data_type="derived_data", batch_size=200, window_size=60, labeling_method="feature_extraction", test_percentage=0.2, device=None):
        """
        Description:
            This method returns a wrapper with the prepared data.
        Arguments:
            -feature_list (list):               A list of features you want to feed to your Neural Net
            -feature_range (tuple):             To what range the features get scaled
            -scaling_mode (string):             Wheter to scale every window by itself, or scale globally over the whole database
            -data_type (string):                What kind of data you wanna use (i.e. raw_data, changes, ...)
            -batch_size (int):                  Size of the batches
            -window_sie (int):                  Size of the rolling window
            -labeling_method (string):          Method with, which the labels were created
            -test_percentage (float):           What percentage of the data should be used as test data
            -device (string):                   On what device should the wrapper operate
        """
        #check if TDB was optimized
        if not os.path.isfile(f"{self.path}/labels_0"):
            raise Exception("Before you can get a Wrapper you need to optimize your DataBase atleast once")

        wrapper = TrainDataBaseWrapper(path=self.path, feature_list=feature_list, feature_range=feature_range, scaling_mode=scaling_mode,
                                       data_type=data_type, batch_size=batch_size, window_size=window_size, labeling_method=labeling_method,
                                       test_percentage=test_percentage, device=device)
        
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
    def _feature_extraction_labeling(self, window_size, trading_fee, search_algo, trial_amount, max_workers, verbose):
        
        #the features with which we are going to determine the peaks and lows
        feature_list = ["close", "volume_obv", "volume_cmf", "volume_fi", "momentum_mfi",
                    "volume_vwap", "volatility_bbm", "volatility_bbh", "volatility_bbp",
                    "volatility_kcc", "volatility_kcp", "volatility_dch", "trend_macd",
                    "trend_macd_diff", "trend_ema_fast", "trend_adx_pos", "trend_vortex_ind_diff",
                    "trend_cci", "trend_kst_diff", "momentum_rsi", "momentum_tsi", "momentum_uo",
                    "momentum_stoch_signal", "momentum_wr", "momentum_ao", "momentum_roc"]

        #the range of the parameters of the labeling process
        parameter_range = {
            "hold_factor": tune.uniform(1,20),
            "threshold": tune.uniform(0,100),
            "distance": tune.uniform(1,10),
            "prominence": tune.uniform(0,10),
            "width": tune.uniform(1,20)
        }

        #check if old labels are available
        old_labels_available = True if "feature_extraction" in self.dbid["alm_optimal"].keys() else False

        #load in the data
        df = self["raw_data", 0, :, feature_list]

        #create the data were are going to work on
        data = pd.DataFrame()
        data["close"] = df["close"]
        data["hold"] = 0
        data["buy"] = 0
        data["sell"] = 0
        data["bsh"] = np.nan

        #calculate the amount of windows needed
        amount_of_windows = math.ceil(data.shape[0]/window_size)

        #outer labeling loop
        for i in range(amount_of_windows):
            
            def objective(parameter_dict, only_labeling=False):
                
                #get the window
                window = df.iloc[i*window_size:(i+1)*window_size,:].copy().reset_index(drop=True)
                data_window = data.iloc[i*window_size:(i+1)*window_size,:].copy().reset_index(drop=True)

                #inner labeling loop
                for feature in feature_list:
                    
                    #get the feature array
                    array = np.array(window[feature])

                    #find indeces of peaks and lows
                    peaks, _ = signal.find_peaks(array, threshold=parameter_dict["threshold"], distance=parameter_dict["distance"], prominence=parameter_dict["prominence"], width=parameter_dict["width"])
                    lows, _ = signal.find_peaks(-array, threshold=parameter_dict["threshold"], distance=parameter_dict["distance"], prominence=parameter_dict["prominence"], width=parameter_dict["width"])

                    #update scores in the data
                    data_window.iloc[lows, 2] += 1                                 #update the buy column
                    data_window.iloc[peaks, 3] += 1                                #update the sell column
                    hold = np.ones(len(data_window))
                    hold[peaks] = 0
                    hold[lows] = 0
                    data_window["hold"] += hold/parameter_dict["hold_factor"]      #update the hold column
                
                #set the right index at bsh
                data_window["bsh"] = data_window.iloc[:,1:4].idxmax(axis=1)
                data_window.loc[data_window["bsh"] == "hold", "bsh"] = 0 
                data_window.loc[data_window["bsh"] == "buy", "bsh"] = 1 
                data_window.loc[data_window["bsh"] == "sell", "bsh"] = 2
                
                if not only_labeling:
                    #create array for profit calcs
                    profit_array = np.array(data_window.iloc[:,[0, 4]])

                    #run profit calcs
                    specific_profit, _ = calculate_profit(input_array=profit_array, trading_fee=trading_fee)

                    #tune.report(specific_profit=specific_profit)
                    return specific_profit
                
                else:
                    return data_window.iloc[:,4]

            class Objective(tune.Trainable):
                def setup(self, config):
                    # config (dict): A dict of hyperparameters
                    self.parameter_dict = config
                
                def step(self):
                    specific_profit = objective(self.parameter_dict)

                    return {"specific_profit": specific_profit}

            #run ray tune
            #setup the searchalgo
            if search_algo == "bayesopt":
                search_alg = BayesOptSearch(random_search_steps=trial_amount/10)
                search_alg = ConcurrencyLimiter(search_alg, max_concurrent=max_workers)
            elif search_algo == "hyperopt":
                search_alg = HyperOptSearch(n_initial_points=trial_amount/10, points_to_evaluate=None)
                search_alg = ConcurrencyLimiter(search_alg, max_concurrent=max_workers)
            else:
                raise Exception("You chose a Search Algorithm that is not available, please choose from this list: bayesopt, hyperopt")

            #run the optimization
            #result = tune.run(objective, config=parameter_range, metric="specific_profit", search_alg=search_alg, mode="max", num_samples=trial_amount, verbose=verbose, keep_checkpoints_num=0, checkpoint_score_attr="specific_profit", log_to_file=False, server_port=8080, reuse_actors=False)
            result = tune.run(Objective, stop={"training_iteration": 1}, config=parameter_range, num_samples=trial_amount, search_alg=search_alg, mode="max", metric="specific_profit")

            #get old labels profit
            if old_labels_available:
                window = data.iloc[i*window_size:(i+1)*window_size,[0]].copy().reset_index(drop=True)
                window["bsh"] = self["labels", 0, i*window_size:(i+1)*window_size, ["feature_extraction"]]

                old_specific_profit, _ = calculate_profit(np.array(window), trading_fee)

                #save the new labels
                if old_specific_profit < result.best_result["specific_profit"]:
                    #get the new labels
                    labels = objective(result.get_best_config(), only_labeling=True)
                    #save them
                    data.iloc[i*window_size:(i+1)*window_size,4] = np.array(labels)

                #save the old labels
                else:
                    data.iloc[i*window_size:(i+1)*window_size,4] = np.array(window["bsh"])

            #incase there are no old labels:
            else:
                #get the new labels
                labels = objective(result.get_best_config(), only_labeling=True)
                #save them
                data.iloc[i*window_size:(i+1)*window_size,4] = np.array(labels)

        #save the labels
        if os.path.isfile(f"{self.path}/labels_0"):
            #read in the old labels
            labels = pd.read_csv(f"{self.path}/labels_0", index_col=0)
            #replace the labels
            labels["feature_extraction"] = data["bsh"]
            #save csv
            labels.to_csv(path_or_buf=f"{self.path}/labels_0",index_label="index")
        else:
            #create labels df
            labels = pd.DataFrame()
            labels["feature_extraction"] = data["bsh"]
            #save csv
            labels.to_csv(path_or_buf=f"{self.path}/labels_0",index_label="index")

        #get the performance
        profit, _ = calculate_profit(input_array=np.array(data.iloc[:,[0,4]]), trading_fee=trading_fee)

        #save the performance
        self.dbid["alm_optimal"]["feature_extraction"] = {"specific_profit": profit, "trading_fee": trading_fee}
        self.dbid.dump()

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
            def objective(config, checkpoint_dir=None):
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
            result = tune.run(objective, config=self.alm_range[labeling_method], metric="specific_profit", search_alg=search_alg, mode="max", num_samples=trial_amount, verbose=verbose, keep_checkpoints_num=0)
            
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

    def dashboard(self):
        app = dash.Dash()

        app.layout = html.Div([
            dcc.Tabs(id="tabs", value="overview-tab", children=[
                dcc.Tab(label="Overview", value="overview-tab"),
                dcc.Tab(label="Manual Labeling", value="manual-labeling-tab"),
                dcc.Tab(label="Manual Labeling Opt", value="manual-labeling-opt-tab"),
            ]),
            html.Div(id="page-content")
        ])
        
        data = self["raw_data", 0, :, ["close_time", "open", "high", "low", "close"]]
        data2 = self["raw_data", 0, :, ["close_time", "close"]]

        graph_config = {
            "scrollZoom": True,
            "showAxisDragHandles": True,
            "showAxisRangeEntryBoxes": True
        }
        fig = px.line(data2, x="close_time", y="close", range_x=["2020-05-01", "2020-07-30"])

        graph = dcc.Graph(figure=fig, id="graph", config=graph_config)

        page1_overview_layout = html.Div([
            html.H1("Overview"),
            graph
        ])

        page2_manual_labeling_layout = html.Div([
            html.H1("Manual Labeling")
        ])

        page3_manual_labling_opt_layout = html.Div([
            html.H1("Manual Labeling Opt")
        ])

        page_404_layout = html.Div([
            html.H1("404 your link does not exist")
        ])

        #page tabs callback function
        @app.callback(dash.dependencies.Output("page-content", "children"),
                      dash.dependencies.Input("tabs", "value"))
        def display_page(tab):
            if tab == "overview-tab":
                return page1_overview_layout
            elif tab == "manual-labeling-tab":
                return page2_manual_labeling_layout
            elif tab == "manual-labeling-opt-tab":
                return page3_manual_labling_opt_layout

        app.run_server(debug=True)

    @classmethod
    def create(cls, save_path, symbol, date_list, candlestick_interval="1m"):

        #restrict the length of the datelist to 1
        if len(date_list) > 1:
            raise Exception("A TrainDataBase can only consist of 1 dateinterval")

        obj = super().create(save_path, symbol, date_list, candlestick_interval)

        #add things to the dbid
        obj.dbid["alm_optimal"] = {}

        return obj

class TrainDataBaseWrapper(Wrapper):

    def __init__(self, path, feature_list, feature_range, scaling_mode, data_type, batch_size, window_size, labeling_method, test_percentage, device):
        super().__init__(path=path, device=device)

        #save the variables
        self.feature_list = feature_list
        self.feature_range = feature_range
        self.scaling_mode = scaling_mode
        self.data_type = data_type
        self.batch_size = batch_size
        self.window_size = window_size
        self.labeling_method = labeling_method
        self.test_percentage = test_percentage

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
    def __init__(self, path):
        super().__init__(path)

    def get_wrapper(self, feature_list, feature_range=(-1,1), scaling_mode="locally", data_type="derived_data", batch_size=200, window_size=60, device=None):

        wrapper = PerformanceAnalyticsDatabaseWrapper(path=self.path, feature_list=feature_list, feature_range=feature_range, scaling_mode=scaling_mode,
                                                      data_type=data_type, batch_size=batch_size, window_size=window_size, device=device)

        return wrapper

class PerformanceAnalyticsDatabaseWrapper(Wrapper):
    
    def __init__(self, path, feature_list, feature_range, scaling_mode, data_type, batch_size, window_size, device=None):
        super().__init__(path=path, device=device)

        #save the variables
        self.feature_list = feature_list
        self.feature_range = feature_range
        self.scaling_mode = scaling_mode
        self.data_type = data_type
        self.batch_size = batch_size
        self.window_size = window_size
        
        #prepare the data
        self._prepare_data()

        #load in the mmaps
        self.mmaps = [np.load(file=f"{self.path}/{self.id}/data{i}.npy", mmap_mode="r") for i in range(len(self.dbid["date_list"]))]

    def _prepare_data(self):

        for i in range(len(self.dbid["date_list"])):
            
            #load in data/select the features
            data = self[self.data_type, i, :, self.feature_list].to_numpy()

            if self.scaling_mode == "locally":
                #scale if desired
                scaler = preprocessing.MinMaxScaler(feature_range=self.feature_range, copy=False)
                scaler.fit_transform(data)
            else:
                raise Exception("Your desired scaling mode has not been implemented yet")

            #sample/batch the data
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
            windows_amount = length-self.window_size

            #define the indexer
            indexer = np.arange(window_elements)[None, :] + elements_skip*np.arange(windows_amount)[:, None]

            #get the samples
            batches = data[indexer].reshape(windows_amount, self.window_size, features_amount)
            batches_amount = math.floor(batches.shape[0]/self.batch_size)
            batches = batches[0:batches_amount*self.batch_size,:,:]
            batches = batches.reshape(batches_amount, self.batch_size, self.window_size, features_amount)

            #save to disk
            np.save(f"{self.path}/{self.id}/data{i}",batches)

    def data(self, index):
        data = self.mmaps[index]

        for i in range(data.shape[0]):
            batch = np.array(data[i])
    
            yield torch.tensor(batch).to(self.device)

    def __del__(self):
        del self.mmaps
        shutil.rmtree(f"{self.path}/{self.id}")

if __name__ == "__main__":
    tdb = TrainDataBase("C:/Users/fabio/Desktop/projectastro/databases/TrainDatabases/test_tdb")
    tdb._feature_extraction_labeling(window_size=60, trading_fee=0.075, search_algo="hyperopt", trial_amount=10, max_workers=1, verbose=2)