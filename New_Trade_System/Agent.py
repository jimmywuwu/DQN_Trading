"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.7.3
"""

import numpy as np
import pandas as pd
import tensorflow as tf

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# np.random.seed(1)
# tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork1:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
        max_position=5,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.cum_position=0
        # total learning step
        self.learn_step_counter = 0
        self.max_position=max_position
        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()
        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation.drop(columns=['Date'])
        print("choose action\n\n\n",observation)
        observation = np.array([observation])
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        if abs(self.cum_position)<self.max_position:
            if np.random.uniform() < self.epsilon:
                # forward feed the observation and get q value for every actions
                action = np.argmax(actions_value)-1
            else:
                action = np.random.randint(0, self.n_actions)-1
        elif abs(self.cum_position)==self.max_position:
            if self.cum_position>0:
                action = np.argmax(actions_value[0][0:2])-1
            else:
                action = np.argmax(actions_value[0][1:3])
        self.cum_position+=action
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        print("batch_memory is ",batch_memory)
        xxxxx
        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]

        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]

        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """

        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.switch_backend('agg')
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.savefig("image.png")
        plt.close()
        
class DeepQNetwork2:
    def __init__(
            self,
            n_actions,
            n_features,
            lstm_length=15,
            hidden_unit=20,
            num_layer=1,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=1,
            e_greedy_increment=None,
            output_graph=False,
        max_position=5,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.cum_position=0
        # total learning step
        self.learn_step_counter = 0
        self.max_position=max_position
        
        # lstm setting
        self.lstm_length=lstm_length
        self.hidden_unit=hidden_unit
        self.num_layer=num_layer
        
        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, self.n_features * self.lstm_length*2 + 2))
        
        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()
        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, shape=(None, self.lstm_length,self.n_features), name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers
            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                # **步驟2：定義一層 LSTM_cell，只需要説明 hidden_size, 它會自動匹配輸入的 X 的維度

                lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_unit, forget_bias=1.0, state_is_tuple=True)

                # **步驟3：添加 dropout layer, 一般只設置 output_keep_prob

                lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=0.5)

                # **步驟4：調用 MultiRNNCell 來實現多層 LSTM
                
                mlstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * self.num_layer, state_is_tuple=True)
                # **步驟5：用全零來初始化state

                init_state = mlstm_cell.zero_state(self.batch_size, dtype=tf.float32)
                # **步驟6：方法一，調用 dynamic_rnn() 來讓我們構建好的網絡運行起來
                # ** 當 time_major==False 時， outputs.shape = [batch_size, timestep_size, hidden_size] 
                # ** 所以，可以取 h_state = outputs[:, -1, :] 作為最後輸出
                # ** state.shape = [layer_num, 2, batch_size, hidden_size], 
                # ** 或者，可以取 h_state = state[-1][1] 作為最後輸出
                # ** 最後輸出維度是 [batch_size, hidden_size]

                outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=self.s, initial_state=init_state, time_major=False)


                h_state = outputs[:, -1, :]  # 或者 h_state = state[-1][1]

                # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):

                w2 = tf.get_variable('w2', [self.hidden_unit, self.n_actions], initializer=w_initializer, collections=c_names)

                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)

                self.q_eval = tf.matmul(h_state, w2) + b2

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.lstm_length,self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
             # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                # **步驟2：定義一層 LSTM_cell，只需要説明 hidden_size, 它會自動匹配輸入的 X 的維度

                lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_unit, forget_bias=1.0, state_is_tuple=True)

                # **步驟3：添加 dropout layer, 一般只設置 output_keep_prob

                lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=0.5)

                # **步驟4：調用 MultiRNNCell 來實現多層 LSTM
                
                mlstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * self.num_layer, state_is_tuple=True)
                # **步驟5：用全零來初始化state

                init_state = mlstm_cell.zero_state(self.batch_size, dtype=tf.float32)
                # **步驟6：方法一，調用 dynamic_rnn() 來讓我們構建好的網絡運行起來
                # ** 當 time_major==False 時， outputs.shape = [batch_size, timestep_size, hidden_size] 
                # ** 所以，可以取 h_state = outputs[:, -1, :] 作為最後輸出
                # ** state.shape = [layer_num, 2, batch_size, hidden_size], 
                # ** 或者，可以取 h_state = state[-1][1] 作為最後輸出
                # ** 最後輸出維度是 [batch_size, hidden_size]

                outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=self.s, initial_state=init_state, time_major=False)

                h_state = outputs[:, -1, :]  # 或者 h_state = state[-1][1]
                # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):

                w2 = tf.get_variable('w2', [self.hidden_unit, self.n_actions], initializer=w_initializer, collections=c_names)

                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)

                self.q_next = tf.matmul(h_state, w2) + b2


    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        s=np.array(s)
        s=s.flatten()
        s_=np.array(s_)
        s_=s_.flatten()
        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = np.array(observation)[np.newaxis,:]
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        if abs(self.cum_position)<self.max_position:
            if np.random.uniform() < self.epsilon:
                # forward feed the observation and get q value for every actions
                action = np.argmax(actions_value)-1
            else:
                action = np.random.randint(0, self.n_actions)-1
        elif abs(self.cum_position)==self.max_position:
            if self.cum_position>0:
                action = np.argmax(actions_value[0][0:2])-1
            else:
                action = np.argmax(actions_value[0][1:3])
        self.cum_position+=action
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        print("s \n\n\n\n\n",self.s.get_shape())
        print("s_ \n\n\n\n\n",self.s_.get_shape())
        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features*self.lstm_length:].reshape((-1,self.lstm_length,self.n_features)),  # fixed params
                self.s: batch_memory[:, :self.n_features*self.lstm_length].reshape((-1,self.lstm_length,self.n_features)),  # newest params
            })
        

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]

        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]

        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """

        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features*self.lstm_length:].reshape((-1,self.lstm_length,self.n_features)),
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.switch_backend('agg')
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.savefig("image.png")
        plt.close()