'''

Things to do: Change env.reset() to provide a value from 0 to len(x_train) - 302

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import random

from collections import deque
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn import metrics

file = open("1400_4_layer.txt", "a+")

tf.get_logger().setLevel('WARNING')


df = pd.read_csv("Permission_Dataset.csv")

feature_cols = []
for col in df.columns:
    feature_cols.append(col)
#feature_cols.remove("Unnamed: 0")
feature_cols.remove("class")

X = df[feature_cols]
Y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 1)
'''
X_train_np = np.array(X_train_n)
X_test_np = np.array(X_test_n)
X_train = X_train_np.reshape(X_train_n.shape[0], 2, 2, 1)
X_test = X_test_np.reshape(X_test_n.shape[0], 2, 2, 1)
'''

def discount_rate():
    return 0.95

def learning_rate():
    return 0.001

def batch_size():
    return 24

EPISODES = 28

def get_state(i):
    row_index = i
    row_values = X_train.loc[row_index]
    row_values_list = row_values.tolist()
    return row_values_list

class DDQN():
    def __init__(self, states, actions, alpha , gamma, epsilon, epsilon_min, epsilon_decay):
        self.ns = states
        self.na = actions
        self.memory = deque([], maxlen = 2500)  #not sure why it is needed yet.
        self.alpha = alpha
        self.gamma = gamma() 
        # Explore/ Exploit
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = self.build_model()          # primary model
        self.target_model = self.build_model()   # taget model
        self.update_target_from_model()
        self.loss = []  
        #self. train_local_model()

    #def train_local_model(self):
    #    self.model.fit(X_train, y_train, epochs= 200, verbose = 1)

    def build_model(self):
        model = keras.Sequential()
        # Dense Neural Network
        model.add(keras.layers.Dense(32, input_dim = self.ns))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        model.add(keras.layers.Dense(64))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        #model.add(keras.layers.Dense(128))
        #model.add(keras.layers.LeakyReLU(alpha=0.2))
        model.add(keras.layers.Dense(64))
        model.add(keras.layers.LeakyReLU(alpha=0.2))
        # final layer number of neurons must be the same as the number of actions. I believe 2 in this case viz. malicious benign
        model.add(keras.layers.Dense(self.na, activation = 'sigmoid')) 

        
        ''' CNN  (has some erros with input shape to each layer)
        model.add(keras.layers.Conv2D(32, (3, 3), activation = "relu", input_shape = (2,2,1)))
        model.add(keras.layers.MaxPooling2D(2, 2))
        model.add(keras.layers.Conv2D(64, (3, 3), activation="relu"))
        model.add(keras.layers.MaxPooling2D(2, 2))
        model.add(keras.layers.Conv1D(64, 3, activation = "relu"))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dense(1, activation='sigmoid'))
        '''

        model.compile(loss = "mse", optimizer = keras.optimizers.Adam(learning_rate= self.alpha), metrics= ['accuracy'])

        return model

    def update_target_from_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.na) #Explore
        action_vals = self.model.predict(state) #Exploit: Use the NN to predict the correct action from this state
        return np.argmax(action_vals[0])    
    
    def test_action(self, state): #Exploit
        action_vals = self.model.predict(state)
        return np.argmax(action_vals[0])
    
    def store(self, state, action, reward, nstate, done):
        #Store the experience in memory
        self.memory.append( (state, action, reward, nstate, done))

    def experience_replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        x = []
        y = []       
        np_array = np.array(minibatch)
        # states
        st = np.zeros((0,self.ns))
        # next states
        nst = np.zeros((0, self.ns))

        # creating state and next state numpy arrays
        for i in range(len(np_array)):
            st = np.append(st, np_array[i,0], axis = 0)
            nst = np.append(nst, np_array[i, 3], axis = 0)

        st_predict = self.model.predict(st)    
        nst_predict = self.model.predict(nst)
        nst_predict_target = self.target_model.predict(nst)

        index = 0

        for state, action, reward, nstate, done in minibatch:
            x.append(state)
            nst_action_predict_target = nst_predict_target[index]
            nst_action_predict_model = nst_predict[index]
            #print(type(nst_action_predict_model[np.argmax(nst_action_predict_model)]), type(self.gamma))

            if done == True:
                target = reward
            else:
                target = reward + self.gamma * nst_action_predict_target[np.argmax(nst_action_predict_model)]

            target_f = st_predict[index]         
            target_f[action] = target
            y.append(target_f)
            index += 1

        # reshaping for keras fit
        x_reshape = np.array(x).reshape(batch_size, self.ns)    
        y_reshape = np.array(y)
        epoch_count = 1
        hist = self.model.fit(x_reshape, y_reshape, epochs=epoch_count, verbose=1)

        # data for graphing the loss per epoch
        for i in range(epoch_count):
            self.loss.append(hist.history['loss'][i])

        # Decaying epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay    


class Environment():

    def __init__(self, data, label, history_t = 5):
        self.label = label
        self.data = data
        self.history_t = history_t
        self.t = 0
        self.reset()

    def test_reset(self):
        self.t = 0
        self.done = False
        self.history = []
        index = 0
        for col in X_test.columns:
            self.history.append(X_test.iloc[index][col])
        return self.history

    def reset(self):
        self.t = 0
        self.done = False
        #self.profits = 0
        self.history = []
        #self.history = [0 for _ in range(self.history_t)]
        #index = random.randint(0, len(X_train))
        index = 0
        for col in X_train.columns:
            self.history.append(X_train.iloc[index][col])
        return self.history
    
    
    def step(self, act):
        
        label_row = self.label.iloc[self.t, :]
        label_class = label_row['class']
        #reward = abs(act - label_class)
        reward = abs(act - self.label.iloc[self.t]['class'])
        #reward = abs(act - self.label.iloc[self.t,:]['class'])
        #print(type(self.label.iloc[self.t][self.t]), self.label.iloc[self.t][self.t])
        #print((self.label.iloc[self.t]['class']))
        if reward > 0:
            reward = -1
        else:
            reward = 1

        #print(self.t)
        self.t += 1

        self.history = []

        #append all column values to history 
        for col in X_train.columns:
            self.history.append(self.data.iloc[self.t][col])

        #print(self.history)
        if (self.t == len(self.data) -1):
            self.done = True
        return self.history, reward, self.done    
    

action_space = [1,0]

Y_train = pd.DataFrame(y_train)

ns = np.array(get_state(2744)).shape[0]
na = np.array(action_space).shape[0]
ddqn = DDQN(ns, na, learning_rate(), discount_rate, 1, 0.01, 0.995)

batch_size = batch_size()

env = Environment(X_train, Y_train)
rewards = []
epsilons = []
accuracy = []

def write_lists_to_file(list1, list2, filename):
    try:
        with open(filename, '+a') as file:
            file.write("Rewards:\n")
            for value in list1:
                file.write(f"{value}\n")

            file.write("\nEpsilons:\n")
            for value in list2:
                file.write(f"{value}\n")

        print(f"Lists written to '{filename}' successfully.")
    except Exception as e:
        print(f"Error: {e}")

def training():
    # Training
    state_count = 0
    for e in range(27):
        print("Starting Episode " + str(e))
        if state_count == 0:
            state = env.reset()
        #print(state)
        state = np.reshape(state, [1, ns])          # reshape to store in memory to pass to model.predict()
        tot_rewards = 0
        for time in range(300):
            state_count += 1
            action = ddqn.action(state)
            nstate, reward, done = env.step(action)
            nstate = np.reshape(nstate, [1, ns])
            tot_rewards += reward
            ddqn.store(state, action, reward, nstate, done)
            state = nstate

            if done or time == 299:
                rewards.append(tot_rewards)
                epsilons.append(ddqn.epsilon)
                log = "episode: {}/{}, score: {}, e: {}".format(e, EPISODES, tot_rewards, ddqn.epsilon)
                print(log)
                file.write(log)
                file.write("\n")
                break

            # Experience Replay
            if len(ddqn.memory) > batch_size:
                ddqn.experience_replay(batch_size)
            
            #if state_count == (len(X_train) - 1):
            #    state_count = 0
            #    break
                
            
        # update weights after each episode
        ddqn.update_target_from_model()

        # terminal conditions
        if len(rewards) > 12 and np.average(rewards[-3:]) > 240:
            print(rewards)
            break


#print(df)

#training()
def model_save():
    ddqn.model.save("DDQN_local_model.h5")
    ddqn.target_model.save("DDQN_target_model.h5")
Y_test = pd.DataFrame(y_test)
env_t = Environment(X_test, Y_test)



def testing():
    false_negative_count = 0
    false_positive_count = 0
    state_count = 0
    episode_reward_threshold_count = 0
    for e_test in range(35):
        if state_count == 0:
            state = env_t.reset()
        state = np.reshape(state, [1,ns])
        tot_rewards = 0
        for t_test in range(100):
            state_count += 1
            action = ddqn.test_action(state)
            nstate, reward, done = env_t.step(action)

            nstate = np.reshape(nstate, [1,ns])
            tot_rewards += reward
            state = nstate
            
            if action == 0 and env_t.label.iloc[env_t.t]['class'] == 1:
                false_negative_count += 1
            if action == 1 and env_t.label.iloc[env_t.t]['class'] == 0:
                false_positive_count += 1

            if done or t_test == 99:
                rewards.append(tot_rewards)
                epsilons.append(0)
                #accuracy.append(correct_count/total_count)
                print("episode: {}/{}, score: {}, e: {},".format(e_test, 40, tot_rewards, 0))
                if tot_rewards >= 80:
                    episode_reward_threshold_count += 1
                break
            if state_count >= len(X_test) - 1:
                state_count = 0
    print("False positive count: ", false_positive_count)
    print("False negative count: ", false_negative_count)
        #if episode_reward_threshold_count >= 8:
        #    break
        
def plotting():
        #Plotting
    rolling_average = np.convolve(rewards, np.ones(100)/100)

    plt.plot(rewards)
    plt.plot(rolling_average, color='black')
    #plt.axhline(y=195, color='r', linestyle='-') #Solved Line
    #Scale Epsilon (0.001 - 1.0) to match reward (0 - 200) range
    eps_graph = [300*x for x in epsilons]
    plt.plot(eps_graph, color='g', linestyle='-')
    #Plot the line where TESTING begins
    #plt.axvline(x=50, color='y', linestyle='-')
    plt.xlim( (0,EPISODES) )
    plt.ylim( (0,300) )
    plt.show()
    plt.savefig("DDQN.png")
    
running = True
def print_accuracy():
    print(accuracy)

while running:
    print("1. Start training     2. Start testing    3. Save model    4. Quit Program   5. Print Accuracy   6. Log session  7.Plot graph    8. Do Everything")
    choice = int(input("Enter choice"))
    if choice == 1:
        training()
    elif choice == 2:
        testing()    
    elif choice == 3:
        model_save()
    elif choice == 4:
        running = False    
    elif choice == 5:
        print(accuracy)
    elif choice == 6:
        write_lists_to_file(rewards, epsilons, "DDQN_log.txt")
        file.close()
    elif choice == 7:
        plotting()
    elif choice == 8:
        training()
        testing()
        model_save()
        write_lists_to_file(rewards, epsilons, "DDQN_log.txt")
        plotting()
        file.close()
        running = False
    else: 
        print("Enter valid choice")    
