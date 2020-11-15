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
