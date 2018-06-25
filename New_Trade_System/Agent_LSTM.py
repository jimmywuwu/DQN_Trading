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
class DeepQNetwork_LSTM:
    def __init__(
            self,
            n_actions,
            n_features,
            n_inputs=2,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=1,
            n_hidden_units=10,
            e_greedy_increment=None,
            output_graph=False,
        max_position=5,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.n_inputs = n_inputs
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.n_hidden_units = n_hidden_units
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
        tf.reset_default_graph() 
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss       
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            
            weights = {'in': tf.Variable(tf.random_normal([self.n_inputs, self.n_hidden_units]), collections=c_names),'out': tf.Variable(tf.random_normal([self.n_hidden_units, 3]), collections=c_names)}
            biases = {'in': tf.Variable(tf.constant(0.1, shape=[self.n_hidden_units, ]), collections=c_names),'out': tf.Variable(tf.constant(0.1, shape=[3, ]), collections=c_names)}
            s_evaluate = tf.reshape(self.s,[-1,int(self.n_features/2),self.n_inputs])
            s_evaluate = tf.reshape(s_evaluate,[-1,self.n_inputs])
            s_evaluate = tf.matmul(s_evaluate, weights['in']) + biases['in']
            s_evaluate = tf.reshape(s_evaluate,[-1,int(self.n_features/2),self.n_hidden_units])
            print(s_evaluate)
            
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden_units, forget_bias=1.0, state_is_tuple=True)
            init_state = lstm_cell.zero_state(batch_size=1, dtype=tf.float32) # 初始化全零 state
            
            outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, s_evaluate, initial_state=init_state, time_major=False)
#            outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
            self.aaaa = s_evaluate
            self.q_eval = tf.matmul(final_state[1], weights['out']) + biases['out']
        
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)        
        
        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input    
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            weights = {'in': tf.Variable(tf.random_normal([self.n_inputs, self.n_hidden_units]), collections=c_names),'out': tf.Variable(tf.random_normal([self.n_hidden_units, 3]), collections=c_names)}
            biases = {'in': tf.Variable(tf.constant(0.1, shape=[self.n_hidden_units, ]), collections=c_names),'out': tf.Variable(tf.constant(0.1, shape=[3, ]), collections=c_names)}
            s_target = tf.reshape(self.s_,[-1,int(self.n_features/2),self.n_inputs])
            s_target = tf.reshape(s_target,[-1,self.n_inputs])                      
            s_target = tf.matmul(s_target, weights['in']) + biases['in']
            s_target = tf.reshape(s_target,[-1,int(self.n_features/2),self.n_hidden_units])
            
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden_units, forget_bias=1.0, state_is_tuple=True)
            init_state = lstm_cell.zero_state(batch_size=1, dtype=tf.float32) # 初始化全零 state
            
            outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, s_target, initial_state=init_state, time_major=False)
            self.q_next = tf.matmul(final_state[1], weights['out']) + biases['out'] 

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
        observation = np.array([observation])
#        print(observation)
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
#        test = self.sess.run(self.aaaa, feed_dict={self.s: observation})
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
        return actions_value, action

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
        plt.savefig("lstm/lstm_image10.png")
        plt.close()