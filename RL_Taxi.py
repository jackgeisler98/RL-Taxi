#Jack Geisler 
#imports done manually in the demo

from IPython.display import HTML 
import numpy as np
import gym
import random

#globals manually entered in the demo 
env = gym.make("Taxi-v2")
env.render()

action_size = env.action_space.n
print("Action size ", action_size)

state_size = env.observation_space.n
print("State size", state_size)

qtable = np.zeros((state_size, action_size))
print(qtable)

total_episodes = 50000
total_test_episodes = 100   # Total episodes
max_steps = 99              #max steps per episode

learning_rate = 0.7
gamma = 0.618                #discounting rate

#exploration parameters
epsilon = 1.0               #exploration rate
max_epsilon = 1.0           #Exploration probability at start
min_epsilon = 0.01          #minimum exploration probability
decay_rate = 0.01           #Exponential decay rate for exploration probability

#List of Rewards
#rewards = []

#2 for life or until learning is stopped
for episode in range(total_episodes):
    #reset environment
    state = env.reset()
    step = 0
    done = False

    for step in range (max_steps):
        #3. choose an action a in the current world state (s)
        ##Fist we randomize a number
        exp_exp_tradeoff = random.uniform(0, 1)

        ##If this number > greater than epsilon --> exploitation (taking the bggest Q value for this state)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[state, :])

        #else doing a random choice --> exploraton
        else:
            action = env.action_space.sample()

        #take the action (a) and observe the outcome state(s') and reward (r)
        new_state, reward, done, info = env.step(action)

        #update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        #qtable[new_state,:] : all the action we can take from new state
        qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])

        #our new state is state
        state = new_state

        #if done (if we're dead) : finish episode
        if done ==True:
            break
        
    episode += 1

    #Reduce epsion (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)

env.reset()
rewards = []
total_rewards = 0 

for episodes in range (total_test_episodes):
    state = env.reset()
    step = 0
    done = False
    print("**********************************************")
    print("EPISODE", episode)

    for step in range(max_steps):
        env.render()
        #take the action (index) that have the maximum expected future reward given that state
        action = np.argmax(qtable[state,:])
        
        new_state, reward, done, info = env.step(action)
        
        total_rewards += reward 
        
        if done:
            rewards.append(total_rewards)
            print ("Score", reward)
            break
        state = new_state
env.close()
print ("Score over time: " + str(sum(rewards)/total_test_episodes))
print (qtable)
