import numpy as np
import time
import pandas as pd
import sys
from Strategy_Evaluation import Strategy_Evaluation

class Market1(object):
    def __init__(self,begin_date="2016/1/1",end_date="2017/5/5"):
        self.action_space = [-1,0,1]
        #action record price,time,action
        self.lstm_length=10
        self.get_price=self.price_data(begin_date,end_date)
        self.action_record=pd.DataFrame(columns=['Date', 'time','price','Vol','action'])
        self.ot=list(self.get_price.__next__())
        self.count_episode=0
        self.n_actions=3
        
    
    def price_data(self,begin_date,end_date):
        series=pd.read_csv('parse_over_data.csv')
        for x in range(self.lstm_length):
            series["price.lag"+str(x+1)]=series.price.shift(x+1)
            series["Vol.lag"+str(x+1)]=series.Vol.shift(x+1)
        series=series[(series.Date>begin_date)&(series.Date<end_date)]
        series=series.dropna()
        self.n_features=series.shape[1]-3
        for index,row in series.iterrows():
            yield row
    

    def step(self, action):
        
        # record St,Ot,At
        self.action_record.loc[self.count_episode]=self.ot[1:5]+[action]
        # reward function
        self.ot=list(self.get_price.__next__())
        s_ = self.ot[3:]
        reward=sum((s_[0]-self.action_record[self.action_record.action != 0].price)*self.action_record[self.action_record.action != 0].action)
        self.count_episode+=1
        return s_,reward


class Market2(object):
    def __init__(self,begin_date="2016/1/1",end_date="2017/5/5"):
        self.action_space = [-1,0,1]
        #action record price,time,action
        self.lstm_length=10
        self.get_price=self.price_data(begin_date,end_date)
        self.action_record=pd.DataFrame(columns=['Date', 'time','price','Vol','action'])
        self.ot=list(self.get_price.__next__())
        self.count_episode=0
        self.n_actions=3
        self.acc_value=[]
        self.sharp_ratio=[]
        self.mdd=[]
        
    
    def price_data(self,begin_date,end_date):
        series=pd.read_csv('parse_over_data.csv')
        for x in range(self.lstm_length):
            series["price.lag"+str(x+1)]=series.price.shift(x+1)
            series["Vol.lag"+str(x+1)]=series.Vol.shift(x+1)
        series=series[(series.Date>begin_date)&(series.Date<end_date)]
        series=series.dropna()
        self.n_features=series.shape[1]-3
        for index,row in series.iterrows():
            yield row
    

    def step(self, action):
        
        # record St,Ot,At

        self.action_record.loc[self.count_episode]=self.ot[1:5]+[action]
        
        # reward function
        self.ot=list(self.get_price.__next__())
        s_ = self.ot[3:]
        Evaluation=Strategy_Evaluation(self)
        Evaluation.cal_win_rate()
        reward=sum((s_[0]-self.action_record[self.action_record.action != 0].price)*self.action_record[self.action_record.action != 0].action)
        if(self.count_episode>50):
            self.mdd+=[Evaluation.max_drawdown()]
            self.sharp_ratio+=[Evaluation.sharp_ratio()]

        self.acc_value+=[reward]
        self.count_episode+=1
        return s_,reward

class Market3(object):
    def __init__(self,begin_date="2016/1/1",end_date="2017/5/5"):
        self.action_space = [-1,0,1]
        #action record price,time,action
        self.lstm_length=10
        self.get_price=self.price_data(begin_date,end_date)
        self.action_record=pd.DataFrame(columns=['Date', 'time','price','Vol','action'])
        self.ot=list(self.get_price.__next__())
        self.count_episode=0
        self.n_actions=3
        self.acc_value=[]
        self.sharp_ratio=[]
        self.mdd=[]
        
    
    def price_data(self,begin_date,end_date):
        series=pd.read_csv('daily.csv')
        for x in range(self.lstm_length):
            series["price.lag"+str(x+1)]=series.price.shift(x+1)
#             series["Vol.lag"+str(x+1)]=series.Vol.shift(x+1)
        series=series[(series.Date>begin_date)&(series.Date<end_date)]
        series=series.dropna()
        self.n_features=series.shape[1]-3
        for index,row in series.iterrows():
            yield row
    

    def step(self, action):
        
        # record St,Ot,At

        self.action_record.loc[self.count_episode]=self.ot[1:5]+[action]
        
        # reward function
        self.ot=list(self.get_price.__next__())
        s_ = self.ot[3:]
        Evaluation=Strategy_Evaluation(self)
        Evaluation.cal_win_rate()
        reward=sum((s_[0]-self.action_record[self.action_record.action != 0].price)*self.action_record[self.action_record.action != 0].action)
        if(self.count_episode>50):
            self.mdd+=[Evaluation.max_drawdown()]
            self.sharp_ratio+=[Evaluation.sharp_ratio()]

        self.acc_value+=[reward]
        self.count_episode+=1
        return s_,reward

## 更新版 Environment Step輸出為一個pd data frame，且需使用的資料只要格式跟price-2.CSV一樣就可以取用
class Market4(object):
    def __init__(self,begin_date="2012/1/1",end_date="2017/5/5",lstm_length=10):
        self.action_space = [-1,0,1]
        #action record price,time,action
        self.lstm_length=lstm_length
        self.get_price=self.price_data(begin_date,end_date)
        self.ot=self.get_price.__next__()
        self.count_episode=0
        self.n_actions=3
        self.acc_value=[]
        self.acc_position=0
        
    
    def price_data(self,begin_date,end_date):
        series=pd.read_csv('price-3.csv')
        index_list=series.columns
        self.n_features=len(index_list)-1
        self.action_record=pd.DataFrame(columns=list(index_list)+["action"])
        for i in index_list:
            for x in range(self.lstm_length-1):
                series[i+"lag"+str(x+1)]=series[i].shift(x+1)
        series=series[(series.Date>begin_date)&(series.Date<end_date)]
        series=series.dropna()

        
        for index,row in series.iterrows():
            b=row
            xx=pd.concat([b.select(lambda col: col.startswith(i)).reset_index(drop=True)  for i in index_list],axis=1)
            xx.columns=index_list

            yield xx
    

    def step(self, action):
        # record St,Ot,At
        self.ot=self.get_price.__next__()
        self.acc_position+=action
        self.action_record.loc[self.count_episode]=list(self.ot.iloc[0,:])+[action]
        
        # 所有對策略的評價全部交給Strategy_Evaluation來做
        # 裡面有MDD , Sharp Ratio , Average Holding Period 等 Metric
        self.evaluation=Strategy_Evaluation(self,initial_account_value=200000)
        # reward function
        self.evaluation.cal_win_rate()
        
        s_ = self.ot
        # reward 1 ,僅考慮累積報酬
#         reward=sum((s_.ret[0]-self.action_record[self.action_record.action != 0].ret)*self.action_record[self.action_record.action != 0].action)
        # reward 2 ,逞罰部位數過大
#         reward=1/(np.abs(self.acc_position)+1)*sum((s_.ret[0]-self.action_record[self.action_record.action != 0].ret)*self.action_record[self.action_record.action != 0].action)
        # reward 3, sharp ratio
        reward=self.evaluation.ExpectReturn
        print("Expect=",reward)
        self.acc_value+=[sum((s_.price[0]-self.action_record[self.action_record.action != 0].price)*self.action_record[self.action_record.action != 0].action)]
        self.count_episode+=1
        print(self.count_episode)
        return s_,reward