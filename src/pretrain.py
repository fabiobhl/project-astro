#add filepath to path
import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

#fileimports
from database import PerformanceAnalyticsDatabase

#external libraries import
from matplotlib import pyplot as plt
from collections import OrderedDict, namedtuple
from itertools import product

#pytorch imports
import torch

class PerformanceAnalytics():
    
    def __init__(self, path, device=None):
        #create the database
        self.pdb = PerformanceAnalyticsDatabase(path)

        #get the device
        if device == None:
            #check if there is a cuda gpu available
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
    
    def evaluate_network(self, network, feature_list, feature_range=(-1,1), scaling_mode="locally", data_type="derived_data", batch_size=200, window_size=60, trading_fee=0.075, verbose=False):
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

        #move our network to device
        network = network.to(self.device)

        #get the wrapper
        wrapper = self.pdb.get_wrapper(feature_list=feature_list, feature_range=feature_range, scaling_mode=scaling_mode, data_type=data_type, batch_size=batch_size, window_size=60, device=self.device)

        #create the return dict
        return_dict = {
            "specific_profit": 0,
            "specific_profit_rate": 0,
            "specific_profit_stability": 0,
            "trading_per_interval": [],
            "interval_infos": []
        }

        #mainloop
        for i in range(len(self.pdb.dbid["date_list"])):
            #setup variables
            mode = 'buy'
            specific_profit = 0
            
            #create bsh array (The array, where the predictions are going to be safed)
            bsh = np.empty((window_size-1))
            bsh[:] = np.nan
            bsh = torch.as_tensor(bsh).to(self.device)

            #get the data
            batches = wrapper.data(i)
            
            #getting the predictions
            for batch in batches:
                #feed data to nn
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
            profit_data["close"] = self.pdb["raw_data", i, :, ["close"]]
            profit_data["hold"] = np.nan
            profit_data["buy"] = np.nan
            profit_data["sell"] = np.nan
            #shorten the data to predictions length
            profit_data = profit_data.iloc[0:bsh.shape[0],:]
            profit_data["bsh"] = bsh
            profit_data["sprofit"] = np.nan
            profit_data["sprofit_accumulated"] = np.nan

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

                        profit_data[i,5] = local_specific_profit
                        profit_data[i,6] = specific_profit
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