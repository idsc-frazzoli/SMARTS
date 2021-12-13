import logging

import gym

from examples.argument_parser import default_argument_parser
from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.sensors import Observation
from smarts.core.utils.episodes import episodes

logging.basicConfig(level=logging.INFO)

AGENT_ID = "Agent-007"


# NK
# from random import randrange
from smarts.core.bezier_motion_planner import BezierMotionPlanner
import numpy as np

class ChaseViaPointsAgent(Agent):
    def act(self, obs: Observation):
        # print(obs.ego_vehicle_state)
        if (
            len(obs.via_data.near_via_points) < 1
            or obs.ego_vehicle_state.edge_id != obs.via_data.near_via_points[0].edge_id
        ):
            return (obs.waypoint_paths[0][0].speed_limit, 0)

        nearest = obs.via_data.near_via_points[0]
        if nearest.lane_index == obs.ego_vehicle_state.lane_index:
            return (nearest.required_speed, 0)

        return (
            nearest.required_speed,
            1 if nearest.lane_index > obs.ego_vehicle_state.lane_index else -1,
        )
        
        # NK
        # return (obs.waypoint_paths[0][0].speed_limit, randrange(-1,2))


# class ExampleAgent(Agent):
#     def __init__(self, target_speed = 10):
#         self.motion_planner = BezierMotionPlanner()
#         self.target_speed = target_speed

#     def act(self, obs):
#         ego = obs.ego_vehicle_state
#         current_pose = np.array([*ego.position[:2], ego.heading])

#         # lookahead (at most) 10 waypoints
#         target_wp = obs.waypoint_paths[0][:10][-1]
#         dist_to_wp = target_wp.dist_to(obs.ego_vehicle_state.position)
#         target_time = dist_to_wp / self.target_speed

#         # Here we've computed the pose we want to hold given our target
#         # speed and the distance to the target waypoint.
#         target_pose_at_t = np.array(
#             [*target_wp.pos, target_wp.heading, target_time]
#         )

#         # The generated motion planner trajectory is compatible
#         # with the `ActionSpaceType.Trajectory`
#         traj = self.motion_planner.trajectory(
#             current_pose, target_pose_at_t, n=10, dt=0.5
#         )
#         return traj


def main(scenarios, sim_name, headless, num_episodes, seed, max_episode_steps=None):
    agent_spec = AgentSpec(
        interface=AgentInterface.from_type(
            AgentType.LanerWithSpeed, max_episode_steps=max_episode_steps
        ),
        agent_builder=ChaseViaPointsAgent,
    )
    # agent_spec = AgentSpec(
    #     interface=AgentInterface.from_type(AgentType.Tracker),
    #     # params are passed to the agent_builder when we build the agent
    #     agent_params={"target_speed": 5},
    #     agent_builder=ExampleAgent
    # )

    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=scenarios,
        agent_specs={AGENT_ID: agent_spec},
        sim_name=sim_name,
        headless=headless,
        visdom=False,
        timestep_sec=0.1,
        sumo_headless=True,
        seed=seed,
        # zoo_addrs=[("10.193.241.236", 7432)], # Sample server address (ip, port), to distribute social agents in remote server.
        envision_record_data_replay_path="./data_replay",
    )

    for episode in episodes(n=num_episodes):
        agent = agent_spec.build_agent()
        observations = env.reset()
        episode.record_scenario(env.scenario_log)

        dones = {"__all__": False}
        while not dones["__all__"]:
            agent_obs = observations[AGENT_ID]
            agent_action = agent.act(agent_obs)
            observations, rewards, dones, infos = env.step({AGENT_ID: agent_action})
            episode.record_step(observations, rewards, dones, infos)

    env.close()


if __name__ == "__main__":
    parser = default_argument_parser("single-agent-example")
    args = parser.parse_args()

    main(
        scenarios=args.scenarios,
        sim_name=args.sim_name,
        headless=args.headless,
        num_episodes=args.episodes,
        seed=args.seed,
    )
