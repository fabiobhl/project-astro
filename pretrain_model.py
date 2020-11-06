import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from src.utils import TrainDatabase, PerformanceAnalytics, RunBuilder, RunManager

import datetime
import time
import math

#definition of the network
class network(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, dropout, batch_norm, advanced_linear):
        super().__init__()
        #save the parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.advanced_linear = advanced_linear
        #create the lstm layer
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, num_layers=num_layers, dropout=dropout)
        
        #create the batchnorm layer
        if self.batch_norm:
            self.batchnorm1 = nn.BatchNorm1d(60)

        #create the relu
        self.relu = nn.ReLU()

        #create the linear layers
        self.out_advanced = nn.Linear(10, 3)
        self.out_simple = nn.Linear(self.hidden_size, 3)

        if self.hidden_size > 1000:
            self.linear1000 = nn.Linear(self.hidden_size, 1000)
            self.linear500 = nn.Linear(1000, 500)
            self.linear250 = nn.Linear(500, 250)
            self.linear100 = nn.Linear(250, 100)
            self.linear50 = nn.Linear(100, 50)
            self.linear10 = nn.Linear(50, 10)
        elif self.hidden_size > 500:
            self.linear500 = nn.Linear(self.hidden_size, 500)
            self.linear250 = nn.Linear(500, 250)
            self.linear100 = nn.Linear(250, 100)
            self.linear50 = nn.Linear(100, 50)
            self.linear10 = nn.Linear(50, 10)
        elif self.hidden_size > 250:
            self.linear250 = nn.Linear(self.hidden_size, 250)
            self.linear100 = nn.Linear(250, 100)
            self.linear50 = nn.Linear(100, 50)
            self.linear10 = nn.Linear(50, 10)
        elif self.hidden_size > 100:
            self.linear100 = nn.Linear(self.hidden_size, 100)
            self.linear50 = nn.Linear(100, 50)
            self.linear10 = nn.Linear(50, 10)
        else:
            self.linear50 = nn.Linear(self.hidden_size, 50)
            self.linear10 = nn.Linear(50, 10)


    def forward(self, t):
        #lstm1 layer
        t, waste = self.lstm1(t)

        if self.batch_norm:
            t = self.batchnorm1(t)

        #linear layers
        t = t[:,-1,:]
        
        if self.advanced_linear:
            #pass through the linear layers
            if self.hidden_size > 1000:
                t = self.relu(self.linear1000(t))
                t = self.relu(self.linear500(t))
                t = self.relu(self.linear250(t))
                t = self.relu(self.linear100(t))
                t = self.relu(self.linear50(t))
                t = self.relu(self.linear10(t))
            elif self.hidden_size > 500:
                t = self.relu(self.linear500(t))
                t = self.relu(self.linear250(t))
                t = self.relu(self.linear100(t))
                t = self.relu(self.linear50(t))
                t = self.relu(self.linear10(t))
            elif self.hidden_size > 250:
                t = self.relu(self.linear250(t))
                t = self.relu(self.linear100(t))
                t = self.relu(self.linear50(t))
                t = self.relu(self.linear10(t))
            elif self.hidden_size > 100:
                t = self.relu(self.linear100(t))
                t = self.relu(self.linear50(t))
                t = self.relu(self.linear10(t))
            else:
                t = self.relu(self.linear50(t))
                t = self.relu(self.linear10(t))
            
            t = self.out_advanced(t)
        else:
            t = self.out_simple(t)

        return t


params = OrderedDict(
    num_layers = [2, 5],
    lr = [0.01, 0.1],
    batch_size = [500, 200],
    epochs = [40],
    hidden_size = [50, 100],
    dropout = [0.2, 0],
    batch_norm = [True, False],
    advanced_linear = [True],
    balancing_method = [1]     #0: No Balancing, 1: Weigth tensor
)

params2 = OrderedDict(
    num_layers = [2],
    lr = [0.01],
    batch_size = [200],
    epochs = [10],
    hidden_size = [100],
    dropout = [0.2],
    batch_norm = [False],
    advanced_linear = [True],
    balancing_method = [1]     #0: No Balancing, 1: Weigth tensor
)


def train(run_iterable, feature_list, name, performanceanayltics_path, traindatabase_path):

    #check if gpu is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Working on {device}")

    #load a PerfomanceAnayltics object
    pa = PerformanceAnalytics.load(performanceanayltics_path)

    #load a TrainData object
    tdb = TrainDatabase.load(traindatabase_path)

    #create the RunManager
    runman = RunManager()

    #get first run
    run = next(run_iterable)

    #run loop (for every possible configuration)
    while run != False:

        #get the data
        train_data, train_labels, test_data, test_labels = tdb.get_data(feature_list=feature_list, batch_size=run.batch_size)
        
        #move the data to tensors and the desired device
        train_data = torch.from_numpy(train_data).to(device)
        train_labels = torch.from_numpy(train_labels).to(device)
        test_data = torch.from_numpy(test_data).to(device)
        test_labels = torch.from_numpy(test_labels).to(device)

        #create the network
        net = network(input_size=len(feature_list), hidden_size=run.hidden_size, num_layers=run.num_layers, dropout=run.dropout, batch_norm=run.batch_norm, advanced_linear=run.advanced_linear)
        net = net.double().to(device)

        #create the optimizer
        optimizer = torch.optim.Adam(net.parameters(), lr=run.lr)

        #create the criterion (Loss Calculator)
        if run.balancing_method == 1:
            #create the weight tensor
            weights = torch.tensor([train_labels.eq(0).sum(), train_labels.eq(1).sum(), train_labels.eq(2).sum()], dtype=torch.float64)
            weights = weights / weights.sum()
            weights = 1.0 / weights
            weights = weights / weights.sum()

            #create the the criterion
            criterion = nn.CrossEntropyLoss(weight=weights.to(device))
        else:
            criterion = nn.CrossEntropyLoss()

        #start the run in the RunManager
        example = train_data[0,0,:,:].unsqueeze(dim=0)
        runman.begin_run(logfolder_name=name, run=run, network=net, example_data=example)

        #epoch loop
        for epoch in range(run.epochs):
            #start the epoch in the runmanager
            runman.begin_epoch()

            """
            Training
            """
            #set the network to trainmode
            net = net.train()
            #start the training in the runman
            runman.begin_training()

            #training loop
            for i in range(train_data.shape[0]):
                #the the samples and labels
                samples = train_data[i,:,:,:]
                labels = train_labels[i,:]

                #zero out the optimizer
                optimizer.zero_grad()

                #get the predictions
                preds = net(samples)

                #calculate the loss
                loss = criterion(preds, labels)

                #update the weights
                loss.backward()
                optimizer.step()

                #log the data
                runman.track_train_metrics(loss=loss.item(), preds=preds, labels=labels)

            #end the training in the runman
            runman.end_training(num_train_samples=train_labels.shape[0]*train_labels.shape[1])

            """
            Testing
            """

            #set the network to evalutation mode
            net = net.eval()
            #start the testing in the runman
            runman.begin_testing()
            
            #testing loop
            for i in range(test_data.shape[0]):
                
                #get the samples and labels
                samples = test_data[i,:,:,:]
                labels = test_labels[i,:]
                
                #get the predictions
                preds = net(samples)

                #log the data
                runman.track_test_metrics(preds=preds, labels=labels)
            
            #evaluate network performance
            performance_data = pa.evaluate_network(network=net, feature_list=feature_list, trading_fee=0.075)

            #end the testing in the runman
            runman.end_testing(num_test_samples=test_labels.shape[0]*test_labels.shape[1], performance_data=performance_data)

            #end the epoch in the runmanager            
            runman.end_epoch()

            #update status in Terminal
            print("Run: ", run,"Epoch: ", epoch)

        #end the run in the RunManager
        runman.end_run(run=run)


        #get the next run
        try:
            run = next(run_iterable)
        except:
            run = False
        



train(run_iterable=RunBuilder.get_runs_iterator(params), feature_list=["close", "volume", "open", "volatility_bbhi", "volatility_bbli", "volatility_kchi", "volatility_kcli", "trend_macd_diff", "trend_vortex_ind_diff", "trend_kst_diff", "trend_aroon_ind", "trend_psar_up_indicator", "trend_psar_down_indicator"], name=datetime.datetime.now().strftime("%d-%m-%y--%H-%M-%S"), performanceanayltics_path="C:/Users/fabio/Desktop/projectastro/databases/PaDatabases/pa_4weeks-Jul-Aug-Sep-Oct", traindatabase_path="C:/Users/fabio/Desktop/projectastro/databases/TrainDatabases/tdb_march")


"""
ToDo:

    RunManager:
        -Logging of specific_profit per interval in matplotlib graph or as a scalar
        (-Find Solution for profit_per_intervall Logging)
        
    PerformanceAnalytics:
        -Implement max prof calc for pa class

    train function:
        -Implement callbacks for saving the best models
        (-Implement a progressbar in the terminal)

    RunBuilder:
        -Implement Bayesion Optimization in RunBuilder

    Train Database:
        -Implement balancing (Oversampling)
"""