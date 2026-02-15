import gymnasium as gym
env = gym.make('CartPole-v1')
print(env.action_space)
print(env.observation_space)

print(env.observation_space.high)
print(env.observation_space.low)
