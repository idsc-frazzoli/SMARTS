# from gym import spaces
#
# a = spaces.Tuple((spaces.Discrete(2), spaces.Discrete(3)))
#
# print(a)
# print(a.sample())
# print(a.contains((0,0)))
# print(a.contains((2,3)))


# import gym
# from ray.rllib.utils.numpy import one_hot
# import numpy as np
#
# class OneHotEnv(gym.core.ObservationWrapper):
#     # Override `observation` to custom process the original observation
#     # coming from the env.
#     def observation(self, observation):
#         # E.g. one-hotting a float obs [0.0, 5.0[.
#         return one_hot(observation, depth=5)
#
#
# class ClipRewardEnv(gym.core.RewardWrapper):
#     def __init__(self, env, min_, max_):
#         super().__init__(env)
#         self.min = min_
#         self.max = max_
#
#     # Override `reward` to custom process the original reward coming
#     # from the env.
#     def reward(self, reward):
#         # E.g. simple clipping between min and max.
#         return np.clip(reward, self.min, self.max)


# import gym.spaces as spaces
# from gym import ObservationWrapper
#
#
# class FlattenObservation(ObservationWrapper):
#     r"""Observation wrapper that flattens the observation."""
#
#     def __init__(self, env):
#         super().__init__(env)
#         self.observation_space = spaces.flatten_space(env.observation_space)
#
#     def observation(self, observation):
#         return spaces.flatten(self.env.observation_space, observation)


# import time
#
# def tracer(func):
#     def wrapper(*args, **kwargs):
#         print('enter')
#         func(*args, **kwargs)
#         print('exit')
#     return wrapper
#
# def timer(func):
#     def wrapper(*args, **kwargs):
#         start = time.time()
#         answer = func(*args, **kwargs)
#         end = time.time()
#         print(f'Time elapsed: {end - start}')
#         return answer
#     return wrapper
#
# @timer
# @tracer
# def hello_world():
#     print('Hello world!')
#
# hello_world()

class testclass():
    inst = 0
    def __init__(self, var):

        self.var = var

a = testclass(8)

print(a)