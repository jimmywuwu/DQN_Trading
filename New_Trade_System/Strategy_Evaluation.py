import numpy as np
import time
import pandas as pd
import sys
from datetime import datetime, timedelta

class Strategy_Evaluation(object):
    def __init__(self,Market,fee_per_trans=25,price_per_point=50,initial_account_value=200000):
        Transection=Market.action_record
        x=Transection[np.abs(Transection.action)==1]
        if(len(Transection[np.abs(Transection.action)>1])>0):
            print(Transection[np.abs(Transection.action)!=1])
            x_=Transection[np.abs(Transection.action)!=1]
            dd=[np.abs(row[1].action)*[row[1]] for row in x_.iterrows()]
            print(dd)
            dd=[item for sublist in dd for item in sublist]
            x_=pd.DataFrame(columns=['Date', 'time','price','Vol','action'])
            x_=x_.append(dd)
            x_['action'] = [1 if x >0 else -1 for x in x_['action']]
            self.transection=x.append(x_).sort_values(['Date', 'time'], ascending=[1, 1])
        self.transection=x
        self.fee=fee_per_trans*self.transection.shape[0]
        self.acc_value=Market.acc_value
        self.initial_account_value=initial_account_value
        self.price_per_point=price_per_point
    
    def cal_win_rate(self):
        win_=[]
        x_=self.transection.reset_index(drop=True)
        ttt=min(x_[['price','action']][x_.action==1].shape[0],
        x_[['price','action']][x_.action==-1].shape[0])
        for i in range(ttt):
            inBuy=x_[x_.action==1].iloc[i,:]
            inSell=x_[x_.action==-1].iloc[i,:]
            if(inBuy.Date>inSell.Date):
                win_=win_+[inBuy.price-inSell.price]
            elif(inBuy.Date==inSell.Date):
                if(inBuy.time>inSell.time):
                    win_=win_+[inBuy.price-inSell.price]
                else:
                    win_=win_+[inSell.price-inBuy.price]
            else:
                win_=win_+[inSell.price-inBuy.price]
        
        # 下面變數若為nan代表還未有平倉的紀錄出現
        self.Earn=np.array(win_)
        self.Win_rate=np.mean(self.Earn>0)
        self.win_time=np.sum(self.Earn>0)
        self.lose_time=np.sum(self.Earn<0)
        self.Aver_win=np.mean(self.Earn[self.Earn>0])
        self.Aver_lose=-np.mean(self.Earn[self.Earn<0])
        self.ExpectReturn=self.Win_rate*self.Aver_win/self.Aver_lose-(1-self.Win_rate)

    def max_drawdown(self):
        timeseries=np.array(self.acc_value)*self.price_per_point+self.initial_account_value
        i = np.argmax(np.maximum.accumulate(timeseries) - timeseries)
        j = np.argmax(timeseries[:i])
        return (float(timeseries[i]) / timeseries[j]) - 1.
    
    def sharp_ratio(self):
        timeseries=np.array(self.acc_value)*self.price_per_point+self.initial_account_value
        return np.mean(np.diff(np.log(timeseries)))/np.std(np.diff(np.log(timeseries)))
    
    def Aver_Holding_Period(self):
        x_=self.transection.reset_index(drop=True)
        ttt=min(x_[['price','action']][x_.action==1].shape[0],
        x_[['price','action']][x_.action==-1].shape[0])
        inBuy=x_[x_.action==1].iloc[:ttt,:]
        inSell=x_[x_.action==-1].iloc[:ttt,:]
        time1=pd.Series(pd.to_datetime(inBuy['Date'].astype(str).astype(str),format="%Y/%m/%d")).values.astype("timedelta64[s]")
        print(time1)
        time2=pd.Series(pd.to_datetime(inSell['Date'].astype(str).astype(str),format="%Y/%m/%d %H:%M:%S")).values.astype("timedelta64[s]")
        print(time2)
        #  pd datetime格式一直被強迫轉成nano second等級，故輸出的時候除以10e8 
        return np.mean(np.abs(time1-time2)/np.timedelta64(1, 'h'))/10e8
        
        

