print("\nTASK # 1\n")
# takes lot of time since environment is 40x40.
import gym
import numpy as np

env = gym.make("MountainCar-v0")

# hyperparameters
alpha = 0.8  
gamma = 0.99 
epsilon = 0.1 

num_states = 40 
num_actions = 3  # left, stay, right
q_table = np.zeros((num_states, num_states, num_actions))

# convert continuous state space into discrete state space
def discretize_state(state):
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_distance = (env_high - env_low) / num_states
    pos = int((state[0] - env_low[0]) / env_distance[0])
    vel = int((state[1] - env_low[1]) / env_distance[1])
    return pos, vel

# training
num_episodes = 10000
for i in range(num_episodes):
    state = env.reset()
    pos, vel = discretize_state(state)
    done = False
    while not done:
        # exploration vs exploitation
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[pos][vel])
        # take action
        next_state, reward, done, info = env.step(action)
        next_pos, next_vel = discretize_state(next_state)
        # update Q-value
        q_table[pos][vel][action] = q_table[pos][vel][action] + alpha * (reward + gamma * np.max(q_table[next_pos][next_vel]) - q_table[pos][vel][action])
        pos, vel = next_pos, next_vel

# testing
total_reward = 0
num_trials = 100
for i in range(num_trials):
    state = env.reset()
    pos, vel = discretize_state(state)
    done = False
    while not done:
        action = np.argmax(q_table[pos][vel])
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        pos, vel = discretize_state(next_state)
        print("State:", next_state, "Action:", action, "Reward:", reward)
print("Average reward per trial:", total_reward / num_trials)


################################### TASK # 1 ENDS ##################################

print("\nTASK # 2\n")

import gym
import random
import numpy as np

# Taxi-v2 not supported
env = gym.make('Taxi-v3')

# Set up the Q-learning agent
Q = np.zeros((env.observation_space.n, env.action_space.n))
lr = 0.8
gamma = 0.95
epsilon = 0.1

# Define the reward function
def get_reward(state, action, next_state):
    if next_state == 4:
        return 10
    elif state[0] == next_state[2] and state[1] == next_state[3] and action == 5:
        return -10
    else:
        return -1

# Train the agent
for i in range(10000):
    state = env.reset()
    done = False

    while not done:
       
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])
\
        next_state, reward, done, _ = env.step(action)

        Q[state][action] += lr * (reward + gamma * np.max(Q[next_state]) - Q[state][action])
        state = next_state

# Test the agent
state = env.reset()
done = False

while not done:
   
    action = np.argmax(Q[state])
    next_state, reward, done, _ = env.step(action)
    # Print the current state, action, reward, and next state
    print("State:", state, "Action:", action, "Reward:", reward, "Next state:", next_state)
    state = next_state

################################### TASK # 2 ENDS ##################################

print("\nTASK # 3\n")

import numpy as np

# Define the state space
state_space = [0, 1, 2, 3, 4]

# Define the action space
action_space = ['transmit', 'conserve_energy']

# Define the reward function
positive_reward = 1
negative_reward = -1
threshold = 0.5

def get_reward(action, cost):
    if action == 'transmit' and cost < threshold:
        return positive_reward
    elif action == 'transmit' and cost >= threshold:
        return negative_reward
    else:
        return positive_reward

# Define the Q-learning algorithm
def q_learning(state_space, action_space, get_reward, alpha, gamma, epsilon, max_iterations):
    
    q_values = np.zeros((len(state_space), len(action_space)))
    initial_state = state_space[0]
    current_state = initial_state

    def execute_action(action):
        # Simulate the cost of the action
        if action == 'transmit':
            cost = np.random.uniform(0, 1)
        else:
            cost = 0
        
        # Simulate the next state
        if current_state == state_space[-1]:
            next_state = current_state
        else:
            next_state = state_space[state_space.index(current_state) + 1]
        
        return next_state, cost
    
    # Initialize the total reward
    total_reward = 0
    
    
    for i in range(max_iterations):
        # Choose an action using epsilon-greedy policy
        if np.random.uniform() < epsilon:
            action = np.random.choice(action_space)
        else:
            action = action_space[np.argmax(q_values[current_state])]
        
        next_state, cost = execute_action(action)
        reward = get_reward(action, cost)
        q_values[current_state, action_space.index(action)] += alpha * (reward + gamma * np.max(q_values[next_state]) - q_values[current_state, action_space.index(action)])
        
        # Update the total reward and current state
        total_reward += reward
        current_state = next_state
    
    # Return the optimal policy
    policy = {}
    for state in state_space:
        policy[state] = action_space[np.argmax(q_values[state])]
        
    return policy

policy = q_learning(state_space=state_space,
                    action_space=action_space,
                    get_reward=get_reward,
                    alpha=0.1,
                    gamma=0.9,
                    epsilon=0.1,
                    max_iterations=10000)

print(policy)
