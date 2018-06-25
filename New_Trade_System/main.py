import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from Market import Market1,Market2,Market3,Market4
from Agent import DeepQNetwork1,DeepQNetwork2


def run():
    rw=[]
    step = 0
    observation = env.ot.drop(columns=['Date'])
    for episode in range(100):
        # RL choose action based on observation
        action = RL.choose_action(observation)

        # RL take action and get next observation and reward
        observation_, reward = env.step(action)
        observation_=observation_.drop(columns=['Date'])
        rw=rw+[reward]
        
        RL.store_transition(observation, action, reward, observation_)

        if (step > 200) and (step % 5 == 0):
            RL.learn()

        # swap observation
        observation = observation_

        step += 1
    print("交易次數：",sum(env.action_record.action!=0))
    print("手續費：",sum(env.action_record.action!=0)*0.5*50)
    print("賺賠點數：",env.acc_value[-1])
    print("累計損益：",env.acc_value[-1]*50)
    print("累計損益+手續費：",env.acc_value[-1]*50-sum(env.action_record.action!=0)*0.5*50)
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')
    plt.plot(env.action_record[["price"]])
    plt.plot(rw)
    plt.savefig("price_pl.png")
    plt.close()
    plt.plot(env.action_record[["action"]].cumsum())
    plt.savefig("action.png")
    plt.close()
    print(env.action_record[env.action_record.action!=0])
    
    RL.plot_cost()

    
def run2():
    rw=[]
    step = 0
    observation = env.ot.drop(columns=['Date'])

    while(True):
        try:

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward = env.step(action)
            observation_=observation_.drop(columns=['Date'])
            rw=rw+[reward]

            RL.store_transition(observation, action, reward, observation_)

            if (step > 50) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            step += 1
        except StopIteration:
            break
    print("交易次數：",sum(env.action_record.action!=0))
    print("手續費：",sum(env.action_record.action!=0)*0.5*50)
    print("賺賠點數：",env.acc_value[-1])
    print("累計損益：",env.acc_value[-1]*50)
    print("累計損益+手續費：",env.acc_value[-1]*50-sum(env.action_record.action!=0)*0.5*50)
    print("賺次數 :",env.evaluation.win_time)
    print("賠次數 :",env.evaluation.lose_time)
    print("Average Win:",env.evaluation.Aver_win)
    print("Average Lose:",env.evaluation.Aver_lose)
    print("Expected Return:",env.evaluation.ExpectReturn)
    print("Earn:",env.evaluation.Earn)
    print(env.action_record)
    
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')
    plt.plot(env.action_record[["price"]])
    plt.plot(rw)
    plt.savefig("price_pl.png")
    plt.close()
    plt.plot(env.action_record[["action"]].cumsum())
    plt.savefig("action.png")
    plt.close()
#     plt.plot(env.mdd)
#     plt.savefig("mdd.png")
#     plt.close()
#     plt.plot(env.sharp_ratio)
#     plt.savefig("sharp_ratio.png")
#     plt.close()
#     print(env.mdd)
#     print(env.sharp_ratio)
    
    RL.plot_cost()

if __name__ == "__main__":
    # maze game
    lstm_length=20
    env = Market4(begin_date="2016/1/1",end_date="2017/5/5",lstm_length=lstm_length)
    RL = DeepQNetwork2(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=10,
                      memory_size=100,
                      output_graph=True,
                      max_position=1,
                      lstm_length=lstm_length,
                      hidden_unit=10
                      )
    run2()

    
## TO DO 
## 將 env 改成 Market4
## 收到程式碼後run 結果 將Lstm 輸入改成可以接收更多變數
## 調整memory size tune結果
## 
 