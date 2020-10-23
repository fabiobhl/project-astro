import numpy as np
import pandas as pd
import pickle
import random
import os
from binance.client import Client
import ta
import time
from datetime import datetime
import os
import json
import warnings
from sklearn import preprocessing

import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
import keys

"""
Description:
    This environment is only for Training AI's! It is not able to trade live!
    This class is an environment for an agent to act in. The action_space is 3: that means on every step
    the agent takes, it can only choose between 3 actions: buy, sell, hold. In this base version of the environment
    the agent can not decide how much he would like to buy. He can only go full in.
    The episode_length is fixed on initialization. The agent has to take an action on every step, he receives the 
    new state of the environment directly after he took the action.
    This environment is coded in such a way that one can use it as a OpenAI environment. This was done on purpose.
"""
class TrainingEnvironment(DataBase):
    """
    Constructorarguments:
        size:           size of the dataset (amount of minutes) // integer
        symbol:         the currency/symbol to trade needs to be in binance standards e.g.BTCUSDT // string
        features:       the datafeatures that you want to give your agent // list
        datatype:       the type of data you want to show to your agent (e.g. ta_data) // string
        windowsize:     the size of the window that the agent sees on every step // integer
        episode_length: how many 1-minute steps a agent takes in every episode // integer
        tradeamount:    the amount of currency the agent has available to trade with // integer
        tradefees:      the fee that gets charged on every trade (in percentage) // float
    """
    def __init__(self, size, symbol, features, datatype, windowsize=60, episode_length=60, tradeamount=1000, tradefees=0):
        #check if the overall size is big enough for the environment
        if (windowsize + episode_length) > size:
            raise EnvironmentError('The size of the Environment is too small')
            
        #call the __init__ from the DataBase Class to initialize a database
        super().__init__(size, symbol)

        self.features = features #the datafeatures that the observation will contain
        self.datatype = datatype #the datatype that the observation will contain
        self.windowsize = windowsize #the size of the window (observation) that the agent can see
        self.episode_length = episode_length #how long the agent trades per episode (1 = 1min)
        self.tradeamount = tradeamount #the amount of money the agent can use to trade with
        self.tradefees = tradefees #the tradefees that the platform has
        self.act_space = 3 #the different action an agent can take (buy, sell, hold)
        self.obs_space = (self.windowsize, len(self.features)) #the dimension of the observation

        #episode-specific variables (get reset on every new episode)
        self.done = True
        self.episode_counter = 0
        self.action_log = pd.DataFrame()
        self.episode_index = False
        self.mode = 'buy'
        self.cc_amount = 0
        self.profit = 0

        #sample the data
        self.sampled_data = self.sample()

    """
    Split the chosen datatype (from the DataBase Class) into samples and save them in a array (self.sampled_data)
    This algorithm uses the "rolling window" technique
    """
    def sample(self):
        #create empty dataframe
        data = pd.DataFrame()

        #select the chosendatatype
        if self.datatype == "ta_data":
            #choose the columns
            for feature in self.features:
                data[feature] = self.ta_data[feature].copy()
        elif self.datatype == "derive_data":
            #choose the columns
            for feature in self.features:
                data[feature] = self.derive_data[feature].copy()
        elif self.datatype == "scaled_data":
            #choose the columns
            for feature in self.features:
                data[feature] = self.scaled_data[feature].copy()
        else:
            raise EnvironmentError("You chose an invalid datatype")

        #save the shape of data and convert into array
        shape = data.shape
        array = np.array(data)

        #flatten the array
        array = array.flatten()

        features_amount = shape[1]
        #amount of elements in one window
        window_elements = self.windowsize*features_amount
        #amount of elements to skip in next window
        elements_skip = features_amount
        #amount of windows to generate
        windows_amount = self.size-self.windowsize

        #define the indexer
        indexer = np.arange(window_elements)[None, :] + elements_skip*np.arange(windows_amount)[:, None]

        #get the samples
        samples = array[indexer].reshape(windows_amount, self.windowsize, features_amount)

        self.length = samples.shape[0]

        return samples

    """
    reset the environment and return start state
    """
    def reset(self):

        self.done = False
        self.episode_counter = 0
        self.action_log = pd.DataFrame()
        self.mode = 'buy'
        self.cc_amount = 0
        self.profit = 0
        self.episode_index = random.randrange(0, self.length-self.episode_length, 1) #get random index

        #get startstate
        startstate = self.sampled_data[self.episode_index]

        #setup actionlog
        self.action_log['close'] = self.ta_data.copy()['close'][self.episode_index:self.episode_index+self.windowsize]
        self.action_log['bsh'] = np.nan
        self.action_log['hold'] = np.nan
        self.action_log['buy'] = np.nan
        self.action_log['sell'] = np.nan
        self.action_log.reset_index(drop=True, inplace=True)

        #return observation
        return startstate

    """
    -take action and calculate profit & update actionlog
    -return next obs
    """
    def step(self, action): 
        #make sure environment got reset before stepping
        if self.done:
            raise EnvironmentError('Please reset the environment before stepping over it')

        #update the action_log
        index = self.action_log.tail(1).index
        if action == 0:
            self.action_log['bsh'][index] = 0
            self.action_log['hold'][index] = self.action_log.copy()['close'][index]
        elif action == 1:
            self.action_log['bsh'][index] = 1
            self.action_log['buy'][index] = self.action_log.copy()['close'][index]
        elif action == 2:
            self.action_log['bsh'][index] = 2
            self.action_log['sell'][index] = self.action_log.copy()['close'][index]
        else:
            raise EnvironmentError("You chose an invalid action valid actions are: 0: Hold, 1: Buy, 2: Sell")
        
        #caculate the reward
        reward = 0
        if self.mode == 'buy':
            if action == 1:
                price = self.action_log['close'][index].item()
                self.cc_amount = self.tradeamount/price
                self.mode = 'sell'

        elif self.mode == 'sell':
            if action == 2:
                price = self.action_log['close'][index].item()
                amount = price*self.cc_amount
                profit = amount - self.tradeamount
                reward = profit
                self.profit += profit
                self.mode = 'buy'

        #update the episodecounter/index
        self.episode_counter += 1
        self.episode_index += 1

        if self.episode_counter <= self.episode_length:
            #add new line to action_log
            appender = pd.DataFrame({'close':self.ta_data.copy()['close'][self.episode_index+self.windowsize-1].item(),
                                     'bsh': None, 'hold': None, 'buy': None, 'sell': None},
                                     index=[self.windowsize])
            self.action_log = self.action_log.append(appender, ignore_index=True)
            self.action_log.reset_index(drop=True, inplace=True)
            self.action_log = self.action_log.iloc[1:]
            self.action_log.reset_index(drop=True, inplace=True)
            #get the new observation
            observation = self.sampled_data[self.episode_index]

        else:
            #update the doneflag
            self.done = True
            observation = False


        return (observation, reward, self.done)

    #alternative constructor
    @staticmethod
    def from_database(databasepath):
        obj = pickle.load(open(databasepath, "rb"))
        if isinstance(obj, TrainingEnvironment):
            return obj
        else:
            raise EnvironmentError('You tried to load a file which is not an instance of Environment')

    #save the object
    def save(self, databasepath):
        pickle.dump(self, open(databasepath, "wb"))

"""
Description:
    This environment is meant to be used as a backtesting environment.
    You can pair it with any agent you want. You need to set a size at the beginning.
    After initialization, reset the environment and step through it, until the done-flag
    is true. When this point is reached you completed your backtest. You can reset and do
    it as many times as, you wish.
"""
class BacktestingEnvironment(DataBase):
    """
    Constructorarguments:
        size:           size of the dataset (amount of minutes) // integer
        symbol:         the currency/symbol to trade, needs to be in binance standards e.g.BTCUSDT // string
        features:       the datafeatures that you want to give your agent // list
        datatype:       the type of data you want to show to your agent (e.g. ta_data) // string
        windowsize:     the size of the window that the agent sees on every step // integer
        loggingpath:    the path where the logfiles sould be saved // string
        idn:            this id is used to identify the environment // string
        tradeamount:    the amount of currency the agent has available to trade with // integer
        tradefees:      the fee that gets charged on every trade (in percentage) // float
    """
    def __init__(self, size, symbol, features, datatype, loggingpath, idn, windowsize=60, tradeamount=1000, tradefees=0):
        #check if the overall size is big enough for the environment
        if (windowsize) > size:
            raise EnvironmentError('The size of the Environment is too small')
        
        #call the __init__ from the DataBase Class to initialize a database
        super().__init__(size, symbol)

        self.features = features #the datafeatures that the observation will contain
        self.datatype = datatype #the datatype that the observation will contain
        self.windowsize = windowsize #the size of the window (observation) that the agent can see
        self.tradeamount = tradeamount #the amount of money the agent can use to trade with
        self.tradefees = tradefees #the tradefees that the platform has
        self.id = idn #the identification string

        #run-specific variables (get reset when reset() method is called)
        self.done = True
        self.run_counter = -1
        self.action_log = pd.DataFrame()
        self.episode_index = False
        self.mode = 'buy'
        self.cc_amount = 0
        self.profit = 0

        #logging specific variables
        self.loggingpath = loggingpath + "/"
        self.setup_loggingspace() #setup the loggingpsace

        #setup the data
        self.data = self.get_data(self.datatype, self.features)

    def get_data(self, datatype, datafeatures):
        """
        gets the right data from the database
        gets called in the constructor
        """
        #choose the datatype
        if datatype == 'ta_data':
            df = self.ta_data
        elif datatype == 'derive_data':
            df = self.derive_data
        elif datatype == 'scaled_data':
            df = self.scaled_data
        else:
            raise Exception('Please make sure to choose a valid datatype')

        #choose the datafeatures
        data = pd.DataFrame()
        for feature in self.features:
            data[feature] = df[feature]

        return data

    def get_actionlog(self):
        """
        resets the action log. is used in the reset() method.
        """
        #get the prices
        self.action_log['close'] = self.data['close'].copy()

        #add the additional columns
        self.action_log['bsh'] = np.nan
        self.action_log['hold'] = np.nan
        self.action_log['buy'] = np.nan
        self.action_log['sell'] = np.nan
    
    def get_loggingname(self):
        """
        returns the name of the file to which the is going to be written to
        """
        
        name = str(self.run_counter) + "-" + str(self.windowsize) + "-" + str(self.tradeamount) + "-" + self.datatype + "-" + self.symbol + "-"

        for feature in self.features:
            name += feature
            name += "#"

        name = name[:-1] + ".csv"

        return name

    def setup_loggingspace(self):
        path = self.loggingpath + self.id
        if os.path.exists(path):
            raise Exception("This environment was already created please choose another id")
        else:
            os.makedirs(path)

        self.loggingpath = path + "/"

    def reset(self):
        """
        reset the environment and return start state
        """

        #check if the environment gets reset before reaching the end
        if self.done == False:
            #log the data before it gets reset
            data = self.data.copy()
            data['bsh'] = self.action_log['bsh']
            data['buy'] = self.action_log['buy']
            data['sell'] = self.action_log['sell']
            data['hold'] = self.action_log['hold']

            data = data.iloc[:self.episode_index+self.windowsize-1]

            data.to_csv(self.loggingpath + self.get_loggingname())


        self.done = False
        self.run_counter += 1
        self.action_log = pd.DataFrame()
        self.mode = 'buy'
        self.cc_amount = 0
        self.profit = 0
        self.episode_index = 0

        #get startstate
        startstate = self.data.iloc[self.episode_index:self.episode_index+self.windowsize]

        #setup actionlog
        self.get_actionlog()

        #return observation
        return startstate

    def step(self, action):
        """
        -take action and calculate profit & update actionlog
        -return next obs
        """

        #make sure environment got reset before stepping
        if self.done:
            raise EnvironmentError('Please reset the environment before stepping over it')

        #update the action_log
        index = self.episode_index+self.windowsize-1

        if action == 0:
            self.action_log['bsh'].iloc[index] = 0
            self.action_log['hold'].iloc[index] = self.action_log.copy()['close'].iloc[index]
        elif action == 1:
            self.action_log['bsh'].iloc[index] = 1
            self.action_log['buy'].iloc[index] = self.action_log.copy()['close'].iloc[index]
        elif action == 2:
            self.action_log['bsh'].iloc[index] = 2
            self.action_log['sell'].iloc[index] = self.action_log.copy()['close'].iloc[index]
        else:
            raise EnvironmentError(f'You chose an invalid action: {action}, valid actions are: 0: Hold, 1: Buy, 2: Sell')

        #update the episodeindex
        self.episode_index += 1

        if self.episode_index <= self.size-self.windowsize:
            #get the new observation
            observation = self.data.iloc[self.episode_index:self.episode_index+self.windowsize]

        else:
            #update the doneflag
            self.done = True
            observation = False

            #log to file
            data = self.data.copy()
            data['bsh'] = self.action_log['bsh']
            data['buy'] = self.action_log['buy']
            data['sell'] = self.action_log['sell']
            data['hold'] = self.action_log['hold']

            data.to_csv(self.loggingpath + self.get_loggingname())
            print("logged")
            print(self.get_loggingname())

        return (observation, self.done)

class LiveEnvironmentLogger():
    """
    Logging Data:

        Basic Data:
            -open_time                  //raw data from binance
            -open                       //raw data from binance
            -high                       //raw data from binance
            -low                        //raw data from binance
            -close                      //raw data from binance
            -volume                     //raw data from binance
            -close_time                 //raw data from binance
            -open_time_edited           //converted open time to readable format
            -close_time_edited          //converted close time to readable format
        
        Update Data:
            -update                     //the update that got added to the window
            -update_time_start          //the time when the update function got called
            -update_time_end            //the time when the update function returned the new window
            -update_time_delta          //the time the update function needed to create the new window
            -processing_time_start      //the time when the dataprocessing started
            -processing_time_end        //the time when the dataprocessing ended
            -processing_time_delta      //the time the processing took
        
        Action Data:
            -bsh                        //the action the agent took
            -action_time_start          //the time when the action function got called
            -action_time_end            //the time thwn the action function returned
            -action_time_delta          //the time it took the actionfunction
        
    """
    def __init__(self, windowsize, symbol, first_window):
        #save the variables
        self.windowsize = windowsize
        self.symbol = symbol

        #create the log dataframe
        self.log = pd.DataFrame()
        self.log["open_time"] = None
        self.log["open"] = None
        self.log["high"] = None
        self.log["low"] = None
        self.log["close"] = None
        self.log["volume"] = None
        self.log["close_time"] = None
        self.log["open_time_edited"] = None
        self.log["close_time_edited"] = None

        self.log["update_time_start"] = None
        self.log["update_time_end"] = None
        self.log["update_time_delta"] = None
        self.log["processing_time_start"] = None
        self.log["processing_time_end"] = None
        self.log["processing_time_delta"] = None
        
        self.log["bsh"] = None
        self.log["action_time_start"] = None
        self.log["action_time_end"] = None
        self.log["action_time_delta"] = None
        
        basic_columns = ["open_time", "open", "high", "low", "close", "volume", "close_time", "open_time_edited", "close_time_edited"]
        update_columns = ["update_time_start", "update_time_end", "update_time_delta", "processing_time_start", "processing_time_end", "processing_time_delta"]
        action_columns = ["bsh", "action_time_start", "action_time_end", "action_time_delta"]
        
        self.columns = basic_columns + update_columns + action_columns 

        #create a new directory
        self.path = f"C:/Users/fabio/Desktop/livelogs/{datetime.now().strftime('%d_%m_%Y__%H_%M_%S')}"
        os.mkdir(self.path)

        #write the basic info
        info = {
            'windowsize': self.windowsize,
            'symbol': self.symbol
        }
        with open(self.path + '/info.json', 'w') as outfile:
            json.dump(info, outfile)

        #write the first window
        data = first_window
        data['open_time_edited'] = pd.to_datetime(data['open_time'] + 1, unit='ms')
        data['close_time_edited'] = pd.to_datetime(data['close_time'] + 1, unit='ms')
        data['start_time'] = np.nan
        data['end_time'] = np.nan
        data['time_delta'] = np.nan
        data['update_time_delta'] = np.nan
        data['processing_time_delta'] = np.nan
        data['action_time_delta'] = np.nan
        data['bsh_start_time'] = np.nan
        data['bsh_end_time'] = np.nan
        data['bsh_time_delta'] = np.nan
        data['bsh'] = np.nan

        data.to_csv(self.path + "/log.csv", index=False)

    def log(self):
        


        #set all variales to none
        self.update = None
        self.update_time_start = None
        self.update_time_end = None
        self.update_time_delta = None
        self.processing_time_start = None
        self.processing_time_end = None
        self.processing_time_delta = None
        
        self.bsh = None
        self.bsh_time_start = None
        self.bsh_time_end = None
        self.bsh_time_delta = None

"""
Description:
    
"""
class LiveEnvironment():
    """
    Constructorarguments:
        windowsize:     size of the window (amount of minutes) // integer
        symbol:         the currency/symbol to trade, needs to be in binance standards e.g. BTCUSDT // string
    Description:
        This constructor saves all the data from the constructorcall and than creates a first window and
        a first logwindow
    """

    def __init__(self, windowsize, symbol, feature_list, nnprocessing):
        #save the variables
        self.windowsize = windowsize
        self.symbol = symbol
        self.feature_list = feature_list
        self.nnprocessing = nnprocessing
        #create a scaler if nnprocessing is activated
        if self.nnprocessing:
            self.scaler = preprocessing.MinMaxScaler((-1,1))
        
        #create a binance client
        self.client = Client(api_key=keys.key, api_secret=keys.secret)
        
        #get the the first window
        #get the raw_data plus 60 more for adding the tas later in the process
        raw_data = self.client.get_klines(symbol=self.symbol, interval='1m', limit=self.windowsize+61)
        data = pd.DataFrame(raw_data)
        data = data.astype(float)
        data.drop(data.columns[[7,8,9,10,11]], axis=1, inplace=True)
        data.rename(columns = {0:'open_time', 1:'open', 2:'high', 3:'low', 4:'close', 5:'volume', 6:'close_time'}, inplace = True)
        #drop the last unfinished line
        data = data.iloc[0:-1]

        self.window = data
        
        """
        self.logger = LiveEnvironmentLogger(windowsize=self.windowsize, symbol=self.symbol, first_window=self.window.copy())
        """

        #the datarows that are getting normalized with the .pct_change method
        self.pctchange = ['open', 'high', 'low', 'close', 'volume', 'volume_nvi', 'volatility_bbm', 'volatility_bbh', 'volatility_bbl', 'volatility_bbw', 'volatility_kcc',
                          'volatility_kch', 'volatility_kcl', 'volatility_dcl', 'volatility_dch', 'trend_ema_fast', 'trend_ema_slow', 'trend_adx_pos', 'trend_adx_neg',
                          'trend_mass_index', 'trend_ichimoku_a', 'trend_ichimoku_b', 'trend_visual_ichimoku_a', 'trend_visual_ichimoku_b', 'trend_aroon_up', 'trend_aroon_down',
                          'trend_psar', 'trend_psar_up', 'trend_psar_down', 'momentum_rsi', 'momentum_uo', 'momentum_kama']
        #the datarows that are getting normalized with the .diff method
        self.diff = ['volume_adi', 'volume_obv', 'volume_cmf', 'volume_fi', 'volume_em', 'volume_sma_em', 'volume_vpt', 'trend_macd', 'trend_macd_signal', 'trend_macd_diff',
                     'trend_trix', 'trend_cci', 'trend_dpo', 'trend_kst', 'trend_kst_sig', 'trend_kst_diff', 'momentum_mfi', 'momentum_tsi', 'momentum_stoch','momentum_stoch_signal',
                     'momentum_wr', 'momentum_ao', 'momentum_roc']

    def update_environment(self):
        #save the starttime
        update_time_start = time.time()

        #get the update
        update = self.client.get_klines(symbol=self.symbol, interval='1m', limit=2)[0][0:7]

        #check if the update is a new candlestick
        if update[0] == self.window.iloc[-1,0]:
            raise Exception("The update had no effect on the window, time it better")
        else:
            #append the update
            self.window.loc[len(self.window)] = update
            #delete the first row
            self.window.drop(labels=0, axis="index", inplace=True)
            self.window.reset_index(inplace=True, drop=True)
            self.window = self.window.astype(float)
            """
            #add the update to the logger
            self.logger.log.iloc[0,0:7] = self.update
            """
            return_window = self._preprocess()

        """
        #save the times to the logger
        update_time_end = time.time()
        self.logger.log["update_time_end"] = update_time_end
        self.logger.log["update_time_start"] = update_time_start
        self.logger.log["update_time_delta"] = update_time_end-update_time_start
        """
        
        #return the window
        return return_window
    
    def _preprocess(self):
        #the return_window
        return_window = pd.DataFrame()

        #add the tas
        with warnings.catch_warnings(record=True):
            data = ta.add_all_ta_features(self.window.copy(), open="open", high="high", low="low", close="close", volume="volume", fillna=True)

        #select the tas from the ta_list
        for feature in self.feature_list:
            return_window[feature] = data[feature]

        if self.nnprocessing:
            """
            derive the data
            """
            #pct_change
            for column in return_window:
                if column in self.pctchange:
                    return_window[column] = return_window[column].pct_change()

            #diff
            for column in return_window:
                if column in self.diff:
                    return_window[column] = return_window[column].diff()

            """
            scale the data
            """
            #reset the indices
            return_window.reset_index(inplace=True, drop=True)

            self.scaler.fit(return_window)

            #scale
            return_window = pd.DataFrame(self.scaler.transform(return_window))

        #reduce to windowsize
        return_window = return_window.iloc[-(self.windowsize):len(return_window)]

        return return_window


if __name__ == "__main__":
    features = ['close', 'volume', 'volatility_bbhi', 'volatility_bbli', 'volatility_kchi',
                    'volatility_kcli', 'volatility_dchi', 'volatility_dcli', 'trend_psar_up_indicator',
                    'trend_psar_down_indicator']


    env = LiveEnvironment(windowsize=60, symbol="BTCUSDT", feature_list=features, nnprocessing=True)
    
