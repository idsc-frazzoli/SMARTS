from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete
import matplotlib.pyplot as plt
import numpy as np


class MultiAgentEnv(Env):
    
    def __init__(self, world, reward_callback):
        
        # World object
        self.world = world
        
        self.agents = world.agents
        self.n = len(world.agents)
        self.desired_velocity = world.desired_velocity
        self.goal = self.world.goal
        self.initial_position = world.start_state['position'].copy()
        self.initial_velocity = world.start_state['velocity'].copy()
        self.dt = world.dt
        self.dones = [False] * self.n
        self.max_timesteps = world.max_timesteps
        self.limits = world.limits
        
        # Resources object
        self.resources = world.resources
        
        # initialize time
        self.time_step = 0
        
        # set reward callback
        self.reward_callback = reward_callback
        
        # set state
        self.state = {'position': self.initial_position.copy(), 'velocity': self.initial_velocity.copy()}
        
        # acceleration / deceleration of the car
        self.action_space = Box(-10,10,shape=(self.n,))
        
        self.observation_space = Dict({'position': Box(0, self.n*max(self.goal), shape=(self.n,), dtype=float),
                                       'velocity': Box(-100, 100, shape=(self.n,), dtype=float)})
    
    
    def step(self, action):
        # Apply action
        # action = acceleration
        # update velocity and position of the car
        for agent in range(self.n):
            self.state['velocity'][agent] += action[agent] * self.dt
            self.state['position'][agent] += self.state['velocity'][agent] * self.dt
            # update resources
            self.resources.set_resources_from_state(self.time_step, agent, self.state['position'][agent])
            
        self.resources.compute_loads(self.time_step)
        
        reward = 0
        for agent in range(self.n):
            reward += self._get_reward(agent)
            
        self.time_step += 1
        
        # Check if done
        for agent in range(self.n):
            if not self.dones[agent]:
                if self.state['position'][agent] > self.goal[agent]:
                    self.dones[agent] = True
                    reward += 100
                if self.time_step >= self.max_timesteps: 
                    self.dones = [True]*self.n
                if self.state['position'][agent] > self.limits[1] or self.state['position'][agent] < self.limits[0]:
                    self.dones[agent] = True
        

        # Set placeholder for info
        info = {}
        
        done = all(self.dones)
        
        # Return step information
        return self.state, float(reward), done, info
    
    
    def render(self, mode='human'):
        plt.plot([0, max(self.goal) + 0.1*max(self.goal)], [-1, -1], color='k')
        plt.plot([0, max(self.goal) + 0.1*max(self.goal)], [1, 1], color='k')
        for agent in range(self.n):
            if not self.dones[agent]:
                plt.scatter(self.goal[agent], 0, color='red')
                plt.scatter(self.state['position'][agent], 0, color='green',
                            label="{}".format(self.state['velocity'][agent]))
        plt.gca().set_aspect('equal', adjustable='box')
        plt.axis('off')
        # plt.legend()
        plt.show()
    
    
    def reset(self):
        # reset position and velocity
        self.state = {'position': self.initial_position.copy(), 'velocity': self.initial_velocity.copy()}
        self.time_step = 0
        self.dones = [False]*self.n
        self.resources.reset()
        return self.state
    
    
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, 
                                    self.state, 
                                    self.desired_velocity, 
                                    self.dones, 
                                    self.resources,
                                    self.time_step)
    
    
    
    
    
    
    
    
    