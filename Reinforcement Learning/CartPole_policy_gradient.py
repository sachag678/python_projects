"""Learn how to play cartpole with policy gradient and keras."""

import keras
from keras.models import Sequential
from keras.layers.core import Dense

import numpy as np
import gym

env = gym.make('CartPole-v0')
env = env.unwrapped
env.seed(1)

# Environement states
state_size = 4
action_size = env.action_space.n


def discount_and_normalize_rewards(episode_rewards):
    """Discount and normalize rewards."""
    discounted_episode_rewards = np.zeros_like(episode_rewards)
    cumulative = 0.0
    for i in reversed(range(len(episode_rewards))):
        cumulative = cumulative * gamma + episode_rewards[i]
        discounted_episode_rewards[i] = cumulative

    mean = np.mean(discounted_episode_rewards)
    std = np.std(discounted_episode_rewards)
    discounted_episode_rewards = (discounted_episode_rewards - mean) / (std)

    return discounted_episode_rewards


def create_model(input_size, alpha, output_size):
    """Create Model."""
    model = Sequential()
    model.add(Dense(10, input_shape=(input_size, ), activation='relu', kernel_initializer='glorot_normal'))
    # self.model.add(Dropout(0.2))

    model.add(Dense(2, activation='relu', kernel_initializer='glorot_normal'))
    # self.model.add(Dropout(0.2))

    model.add(Dense(output_size, activation='softmax', kernel_initializer='glorot_normal'))

    adam = keras.optimizers.Adam(lr=alpha, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=adam)

    return model

# Hyperparams
max_episodes = 1000
alpha = 0.01
gamma = 0.95

# initialize
allRewards, episode_states, episode_dlogloss, episode_rewards = [], [], [], []

for episode in range(max_episodes):

    state = env.reset()

    model = create_model(state_size, alpha, action_size)

    # env.render()

    while True:
        action_prob = model.predict(state.reshape([1, 4]), batch_size=1)

        action = np.random.choice(range(action_size), p=action_prob.ravel())

        new_state, reward, done, info = env.step(action)

        episode_states.append(state)

        action_ = np.zeros(action_size)
        action_[action] = 1

        episode_dlogloss.append(- np.log(action_ * action_prob))
        episode_rewards.append(reward)

        if done:
            # Calculate sum reward
            episode_rewards_sum = np.sum(episode_rewards)

            allRewards.append(episode_rewards_sum)

            total_rewards = np.sum(allRewards)

            # Mean reward
            mean_reward = np.divide(total_rewards, episode + 1)

            maximumRewardRecorded = np.amax(allRewards)

            print("==========================================")
            print("Episode: ", episode)
            print("Reward: ", episode_rewards_sum)
            print("Mean Reward", mean_reward)
            print("Max reward so far: ", maximumRewardRecorded)

            discounted_episode_rewards = discount_and_normalize_rewards(episode_rewards)

            nd_episode_dlogloss = np.array(episode_dlogloss)
            for i in range(len(episode_dlogloss)):
                nd_episode_dlogloss[i] = episode_dlogloss[i] * np.array(discounted_episode_rewards)[i]

            x_train = np.vstack(np.array(episode_states))
            y_train = np.vstack(nd_episode_dlogloss)

            model.fit(x_train, y_train, batch_size=len(x_train), epochs=1, verbose=0)

            # Reset the transition stores
            episode_states, episode_dlogloss, episode_rewards = [], [], []

            break

        state = new_state
