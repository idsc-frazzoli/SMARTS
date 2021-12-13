
#%% 

from smarts.core.agent_interface import AgentInterface
from smarts.core.controllers import ActionSpaceType

agent_interface = AgentInterface(
    max_episode_steps=1000,
    waypoints=True,
    neighborhood_vehicles=True,
    drivable_area_grid_map=True,
    ogm=True,
    rgb=True,
    lidar=False,
    action=ActionSpaceType.Continuous,
)



#%% for further customization
from smarts.core.agent_interface import AgentInterface, NeighborhoodVehicles, DrivableAreaGridMap, OGM, RGB, Waypoints
from smarts.core.controllers import ActionSpaceType

agent_interface = AgentInterface(
    max_episode_steps=1000,
    waypoints=Waypoints(lookahead=50), # lookahead 50 meters
    neighborhood_vehicles=NeighborhoodVehicles(radius=50), # only get neighborhood info with 50 meters.
    drivable_area_grid_map=True,
    ogm=True,
    rgb=True,
    lidar=False,
    action=ActionSpaceType.Continuous,
)


#%% 

from smarts.env.rllib_hiway_env import RLlibHiWayEnv
env = RLlibHiWayEnv(
    config={
        "scenarios": [scenario_path], # scenarios list
        "agent_specs": {AGENT_ID: agent_spec}, # add agent specs
        "headless": False, # enable envision gui, set False to enable.
        "seed": 42, # RNG Seed, seeds are set at the start of simulation, and never automatically re-seeded.
    }
)

# reset env and build agent
observations = env.reset()
agent = agent_spec.build_agent()

# step env
agent_obs = observations[AGENT_ID]
agent_action = agent.act(agent_obs)
observations, rewards, dones, _ = env.step({AGENT_ID: agent_action})

# close env
env.close()

















