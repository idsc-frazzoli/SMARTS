
# TODO: Use AUCG cost formulation and train model to see if good results can be obtained

from world import World
from world import Resources
from env import MultiAgentEnv
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

def quadspeedcost(agent, state, desired_velocity, dones, resources, time_step):
    reward = 0.0
    speed_coeff = 0.1
    if not dones[agent]:
        reward -= speed_coeff * np.power(state['velocity'][agent] - desired_velocity[agent], 2)
    return reward

def aucg_cost(agent, state, desired_velocity, dones, resources, time_step):
    
    # resource cost function of degree d = 2
    def resource_cost(load, total_load, neighborhood_factor):
        degree = 5
        return neighborhood_factor * load * total_load ** degree
    
    cost = 0
    for neigh, neigh_fact in enumerate(resources.neighborhood_factors):
        cost += sum([resource_cost(resources.resource_map[agent][neigh][time_step][x],
                                   resources.resource_loads[neigh][time_step][x],
                                   neigh_fact) 
                     for x in range(resources.nx)])
    
    return -cost



n_agents = 2
limits = np.array([-10, 50])
agents = np.arange(n_agents)
goal = np.array([40, 40])
start_state = {'position': np.array([0.0, 5.0]),
               'velocity': np.array([11.0, 9.0])}
dt = 0.1
max_timesteps = 50
desired_velocity = np.array([15.0, 5.0])

# set up resources
dx = 1
neighborhood_factors = np.array([1, 0.1, 0.01, 0.001])
neighborhood_sizes = np.array([1, 1, 2, 3])
resources = Resources(n_agents, limits, dx, max_timesteps, neighborhood_factors, neighborhood_sizes)

world = World(limits, agents, goal, start_state, dt, max_timesteps, desired_velocity, resources)

env = MultiAgentEnv(world, reward_callback=aucg_cost)

env.observation_space.sample()
env.reset()

env.render()

# print(env.world.check_goal_reached({'position': 40, 'velocity': env.state['velocity'][0]}, 0))

#%%

model = PPO("MultiInputPolicy", env, verbose=1)

#%%

for _ in range(1):
    model.learn(total_timesteps=100000)
    
    evaluate_policy(model, env, n_eval_episodes=1, render=True)


#%%
evaluate_policy(model, env, n_eval_episodes=1, render=True)

