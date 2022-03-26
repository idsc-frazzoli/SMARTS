
from world import World
from world import Resources
from env import MultiAgentEnv
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from pathlib import Path

import os
import shutil
import moviepy.video.io.ImageSequenceClip

from time import gmtime, strftime


def quadspeedcost(agent, state, desired_velocity, dones, resources, time_step):
    reward = 0.0
    speed_coeff = 0.1
    if not dones[agent]:
        reward -= speed_coeff * np.power(state['velocity'][agent] - desired_velocity[agent], 2)
    return reward


def aucg_cost(agent, state, desired_velocity, dones, resources, time_step):
    # resource cost function of degree d
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


def render_video(save_path, fps=30):

    image_folder = Path('tmp')

    image_files = [os.path.join(image_folder, img)
                   for img in os.listdir(image_folder)
                   if img.endswith(".png")]
    image_files.sort()
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    print(save_path)
    clip.write_videofile(save_path + '.mp4')

    shutil.rmtree(image_folder)


if __name__ == "__main__":
    # set up 1D road environment
    n_agents = 2
    agents = np.arange(n_agents)

    # road limits
    limits = np.array([-10, 50])

    # goals and start states of the agents
    goal = np.array([40, 40])
    start_state = {'position': np.array([0.0, 5.0]),
                   'velocity': np.array([11.0, 8.0])}

    # time step size (0.1 seconds)
    dt = 0.1

    max_timesteps = 50
    desired_velocity = np.array([15.0, 5.0])

    # render_path = Path('log', 'visualization')

    # space discretization
    dx = 1
    # factors (weights) for the different neighborhoods
    neighborhood_factors = np.array([1, 0.1, 0.01, 0.001])
    # neighborhood sizes, 0 means that only the currently occupied space resource is used, one means that all resources
    # within a one-neighborhood are used (3 resources each time step)
    neighborhood_sizes = np.array([1, 2, 3, 4])
    resources = Resources(n_agents, limits, dx, max_timesteps, neighborhood_factors, neighborhood_sizes)

    world = World(limits, agents, goal, start_state, dt, max_timesteps, desired_velocity, resources)

    # set up environment with aucg cost
    env = MultiAgentEnv(world, reward_callback=aucg_cost)

    # set up PPO model for training
    model = PPO("MultiInputPolicy", env, verbose=1)

    # training process
    num_checkpoints = 20
    time_steps_per_checkpoint = 50000
    datetime = strftime("%Y%m%d_%H%M%S", gmtime())
    model_path = Path('training', 'saved_models', datetime)
    for checkpoint in range(1, num_checkpoints + 1):
        cp_path = Path(model_path, 'checkpoints', 'checkpoint_{0:04}'.format(checkpoint))
        model.learn(total_timesteps=time_steps_per_checkpoint)
        model.save(cp_path)


    # valuate policies
    video_path = Path(model_path, 'videos')
    for checkpoint in range(1, num_checkpoints + 1):
        cp_path = Path(model_path, 'checkpoints', 'checkpoint_{0:04}'.format(checkpoint))
        model = PPO.load(cp_path, env)
        evaluate_policy(model, env, n_eval_episodes=1, render=True)
        video_path.mkdir(parents=True, exist_ok=True)
        render_video(str(video_path) + '/checkpoint_{0:04}'.format(checkpoint), fps=20)


