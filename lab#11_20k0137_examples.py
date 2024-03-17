import numpy as np
import gym
# Create the environment
env = gym.make('FrozenLake-v1')

# Set the hyperparameters
alpha = 0.1
gamma = 0.99
num_episodes = 5000

# Initialize the Q-table
num_states = env.observation_space.n
num_actions = env.action_space.n
Q = np.zeros((num_states, num_actions))
Q = np.zeros((env.observation_space.n, env.action_space.n))
# Run the Q-learning algorithm
for i in range(num_episodes):
# Reset the environment
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
    # Choose an action using the epsilon-greedy policy
        if np.random.uniform(0, 1) < 0.5:
            action = env.action_space.sample() # random action
        else:
            action = np.argmax(Q[state, :]) # action with maximum Q-value

    # Take the action and observe the next state and reward
        next_state, reward, done, info = env.step(action)

    # Update the Q-value of the (state, action) pair
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])

    # Update the state and the total reward
        state = next_state
        total_reward += reward

    # Print the total reward obtained during the episode
    print(f"Episode {i}: Total reward = {total_reward}")


# Test the learned policy
num_test_episodes = 100
num_test_steps = 100
num_successes = 0

for i in range(num_test_episodes):
    state = env.reset()
    done = False
    steps = 0

    while not done and steps < num_test_steps:
# Choose the action with the highest Q-value
        action = np.argmax(Q[state, :])

# Take the action and observe the next state and reward
        next_state, reward, done, info = env.step(action)

# Update the state and the step count
        state = next_state
        steps += 1

    if state == 15:
        num_successes += 1

print("Success rate:", num_successes/num_test_episodes)