# MIT License
#
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
from collections import deque
from typing import Sequence

import gym
import numpy as np
from ray import logger
from ray.rllib.models import Preprocessor
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.utils.annotations import override
from scipy.spatial import distance

from baselines.marl_benchmark.common import ActionAdapter, cal_obs, cal_obs_centralized
from baselines.marl_benchmark.wrappers.rllib.wrapper import Wrapper

import math


def get_ttc(env_obs):
    ego = env_obs.ego_vehicle_state
    neighbor_vehicle_states = env_obs.neighborhood_vehicle_states

    heading_angle = ego.heading + math.pi / 2.0
    ego_heading_vec = np.asarray([math.cos(heading_angle), math.sin(heading_angle)])
    ttc = []
    for v in neighbor_vehicle_states:
        rel_pos = np.asarray(
            list(map(lambda x: x[0] - x[1], zip(v.position[:2], ego.position[:2])))
        )

        rel_dist = np.sqrt(rel_pos.dot(rel_pos))

        v_heading_angle = math.radians(v.heading)
        v_heading_vec = np.asarray(
            [math.cos(v_heading_angle), math.sin(v_heading_angle)]
        )

        ego_heading_norm_2 = ego_heading_vec.dot(ego_heading_vec)
        rel_pos_norm_2 = rel_pos.dot(rel_pos)
        v_heading_norm_2 = v_heading_vec.dot(v_heading_vec)

        ego_cosin = ego_heading_vec.dot(rel_pos) / np.sqrt(
            ego_heading_norm_2 + rel_pos_norm_2
        )

        v_cosin = v_heading_vec.dot(rel_pos) / np.sqrt(
            v_heading_norm_2 + rel_pos_norm_2
        )

        rel_speed = 0
        if ego_cosin <= 0 and v_cosin > 0:
            rel_speed = 0
        else:
            rel_speed = ego.speed * ego_cosin - v.speed * v_cosin

        ttc.append(float(min(rel_dist / max(1e-5, rel_speed), 5.0)))

    return ttc


def _get_preprocessor(space: gym.spaces.Space):
    if isinstance(space, gym.spaces.Tuple):
        preprocessor = TupleStackingPreprocessor
    else:
        preprocessor = get_preprocessor(space)
    return preprocessor


class TupleStackingPreprocessor(Preprocessor):
    @override(Preprocessor)
    def _init_shape(self, obs_space: gym.Space, options: dict):
        assert isinstance(self._obs_space, gym.spaces.Tuple)
        size = None
        self.preprocessors = []
        for i in range(len(self._obs_space.spaces)):
            space = self._obs_space.spaces[i]
            logger.debug("Creating sub-preprocessor for {}".format(space))
            preprocessor = _get_preprocessor(space)(space, self._options)
            self.preprocessors.append(preprocessor)
            if size is not None:
                assert size == preprocessor.size
            else:
                size = preprocessor.size
        return len(self._obs_space.spaces), size

    @override(Preprocessor)
    def transform(self, observation):
        self.check_shape(observation)
        array = np.zeros(self.shape[0] * self.shape[1])
        self.write(observation, array, 0)
        array.reshape(self.shape)
        return array

    @override(Preprocessor)
    def write(self, observation, array, offset):
        assert len(observation) == len(self.preprocessors), observation
        for o, p in zip(observation, self.preprocessors):
            p.write(o, array, offset)
            offset += p.size


class FrameStack(Wrapper):
    """ By default, this wrapper will stack 3 consecutive frames as an agent observation"""

    def __init__(self, config):
        super(FrameStack, self).__init__(config)
        config = config["custom_config"]
        self.num_stack = config["num_stack"]

        self.observation_adapter = config["observation_adapter"]
        self.action_adapter = config["action_adapter"]
        self.info_adapter = config["info_adapter"]
        self.reward_adapter = config["reward_adapter"]

        self.observation_space = config["observation_space"]
        self.action_space = config["action_space"]

        self.frames = {
            agent_id: deque(maxlen=self.num_stack) for agent_id in self._agent_keys
        }

    @staticmethod
    def get_observation_space(observation_space, wrapper_config):
        frame_num = wrapper_config["num_stack"]
        if isinstance(observation_space, gym.spaces.Box):
            return gym.spaces.Tuple([observation_space] * frame_num)
        elif isinstance(observation_space, gym.spaces.Dict):
            # inner_spaces = {}
            # for k, space in observation_space.spaces.items():
            #     inner_spaces[k] = FrameStack.get_observation_space(space, wrapper_config)
            # dict_space = gym.spaces.Dict(spaces)
            return gym.spaces.Tuple([observation_space] * frame_num)
        else:
            raise TypeError(
                f"Unexpected observation space type: {type(observation_space)}"
            )

    @staticmethod
    def get_action_space(action_space, wrapper_config=None):
        return action_space

    @staticmethod
    def get_observation_adapter(
            observation_space, feature_configs, wrapper_config=None
    ):
        def func(env_obs_seq):
            assert isinstance(env_obs_seq, Sequence)
            observation = cal_obs(env_obs_seq, observation_space, feature_configs)
            # observation = cal_obs_centralized(env_obs_seq, observation_space, feature_configs)
            return observation

        return func

    @staticmethod
    def get_action_adapter(action_type, action_space, wrapper_config=None):
        return ActionAdapter.from_type(action_type)

    @staticmethod
    def stack_frames(frames):
        proto = frames[0]

        if isinstance(proto, dict):
            res = dict()
            for key in proto.keys():
                res[key] = np.stack([frame[key] for frame in frames], axis=0)
        elif isinstance(proto, np.ndarray):
            res = np.stack(frames, axis=0)
        else:
            raise NotImplementedError

        return res

    @staticmethod
    def get_preprocessor():
        return TupleStackingPreprocessor

    def _get_observations(self, raw_frames):
        """Update frame stack with given single frames,
        then return nested array with given agent ids
        """

        for k, frame in raw_frames.items():
            self.frames[k].append(frame)

        agent_ids = list(raw_frames.keys())
        observations = dict.fromkeys(agent_ids)

        for k in agent_ids:
            observation = list(self.frames[k])
            observation = self.observation_adapter(observation)
            observations[k] = observation

        return observations

    def _get_rewards(self, env_observations, env_rewards):
        agent_ids = list(env_rewards.keys())
        rewards = dict.fromkeys(agent_ids, None)

        for k in agent_ids:
            rewards[k] = self.reward_adapter(list(self.frames[k]), env_rewards[k])
        return rewards

    def _get_infos(self, env_obs, rewards, infos):
        if self.info_adapter is None:
            return infos

        res = {}
        agent_ids = list(env_obs.keys())
        for k in agent_ids:
            res[k] = self.info_adapter(env_obs[k], rewards[k], infos[k])
        return res

    def step(self, agent_actions):
        agent_actions = {
            agent_id: self.action_adapter(action)
            for agent_id, action in agent_actions.items()
        }
        env_observations, env_rewards, dones, infos = super(FrameStack, self).step(
            agent_actions
        )

        observations = self._get_observations(env_observations)
        rewards = self._get_rewards(env_observations, env_rewards)
        # NK: 04.04.2022: changed env_rewards to rewards to gain access to the step rewards
        infos = self._get_infos(env_observations, rewards, infos)
        self._update_last_observation(self.frames)

        return observations, rewards, dones, infos

    def reset(self):
        observations = super(FrameStack, self).reset()
        for k, observation in observations.items():
            _ = [self.frames[k].append(observation) for _ in range(self.num_stack)]
        self._update_last_observation(self.frames)
        return self._get_observations(observations)

    # # benchmark cost function
    # @staticmethod
    # def get_reward_adapter(observation_adapter):
    #     def func(env_obs_seq, env_reward):
    #         penalty, bonus = 0.0, 0.0
    #         obs_seq = observation_adapter(env_obs_seq)
    #
    #         # ======== Penalty: too close to neighbor vehicles
    #         # if the mean ttc or mean speed or mean dist is higher than before, get penalty
    #         # otherwise, get bonus
    #         last_env_obs = env_obs_seq[-1]
    #         neighbor_features_np = np.asarray([e.get("neighbor") for e in obs_seq])
    #         if neighbor_features_np is not None:
    #             new_neighbor_feature_np = neighbor_features_np[-1].reshape((-1, 5))
    #             mean_dist = np.mean(new_neighbor_feature_np[:, 0])
    #             mean_ttc = np.mean(new_neighbor_feature_np[:, 2])
    #
    #             last_neighbor_feature_np = neighbor_features_np[-2].reshape((-1, 5))
    #             mean_dist2 = np.mean(last_neighbor_feature_np[:, 0])
    #             # mean_speed2 = np.mean(last_neighbor_feature[:, 1])
    #             mean_ttc2 = np.mean(last_neighbor_feature_np[:, 2])
    #             penalty += (
    #                 0.03 * (mean_dist - mean_dist2)
    #                 # - 0.01 * (mean_speed - mean_speed2)
    #                 + 0.01 * (mean_ttc - mean_ttc2)
    #             )
    #
    #         # ======== Penalty: distance to goal =========
    #         goal = last_env_obs.ego_vehicle_state.mission.goal
    #         ego_2d_position = last_env_obs.ego_vehicle_state.position[:2]
    #         if hasattr(goal, "position"):
    #             goal_position = goal.position
    #         else:
    #             goal_position = ego_2d_position
    #         goal_dist = distance.euclidean(ego_2d_position, goal_position)
    #         penalty += -0.01 * goal_dist
    #
    #         old_obs = env_obs_seq[-2]
    #         old_goal = old_obs.ego_vehicle_state.mission.goal
    #         old_ego_2d_position = old_obs.ego_vehicle_state.position[:2]
    #         if hasattr(old_goal, "position"):
    #             old_goal_position = old_goal.position
    #         else:
    #             old_goal_position = old_ego_2d_position
    #         old_goal_dist = distance.euclidean(old_ego_2d_position, old_goal_position)
    #         penalty += 0.1 * (old_goal_dist - goal_dist)  # 0.05
    #
    #         # ======== Penalty: distance to the center
    #         distance_to_center_np = np.asarray(
    #             [e["distance_to_center"] for e in obs_seq]
    #         )
    #         diff_dist_to_center_penalty = np.abs(distance_to_center_np[-2]) - np.abs(
    #             distance_to_center_np[-1]
    #         )
    #         penalty += 0.01 * diff_dist_to_center_penalty[0]
    #
    #         # ======== Penalty & Bonus: event (collision, off_road, reached_goal, reached_max_episode_steps)
    #         ego_events = last_env_obs.events
    #         # ::collision
    #         penalty += -50.0 if len(ego_events.collisions) > 0 else 0.0
    #         # ::off road
    #         penalty += -50.0 if ego_events.off_road else 0.0
    #         # ::reach goal
    #         if ego_events.reached_goal:
    #             bonus += 20.0
    #
    #         # ::reached max_episode_step
    #         if ego_events.reached_max_episode_steps:
    #             penalty += -0.5
    #         else:
    #             bonus += 0.5
    #
    #         # ======== Penalty: heading error penalty
    #         # if obs.get("heading_errors", None):
    #         #     heading_errors = obs["heading_errors"][-1]
    #         #     penalty_heading_errors = -0.03 * heading_errors[:2]
    #         #
    #         #     heading_errors2 = obs["heading_errors"][-2]
    #         #     penalty_heading_errors += -0.01 * (heading_errors[:2] - heading_errors2[:2])
    #         #     penalty += np.mean(penalty_heading_errors)
    #
    #         # ======== Penalty: penalise sharp turns done at high speeds =======
    #         if last_env_obs.ego_vehicle_state.speed > 60:
    #             steering_penalty = -pow(
    #                 (last_env_obs.ego_vehicle_state.speed - 60)
    #                 / 20
    #                 * last_env_obs.ego_vehicle_state.steering
    #                 / 4,
    #                 2,
    #             )
    #         else:
    #             steering_penalty = 0
    #         penalty += 0.1 * steering_penalty
    #
    #         # ========= Bonus: environment reward (distance travelled) ==========
    #         bonus += 0.05 * env_reward
    #         return bonus + penalty
    #
    #     return func

    # # modified cost function
    # @staticmethod
    # def get_reward_adapter(observation_adapter):
    #     def func(env_obs_seq, env_reward):
    #         penalty, bonus = 0.0, 0.0
    #         obs_seq = observation_adapter(env_obs_seq)
    #
    #         last_env_obs = env_obs_seq[-1]
    #
    #         # ======== Penalty: distance to goal =========
    #         goal = last_env_obs.ego_vehicle_state.mission.goal
    #         ego_2d_position = last_env_obs.ego_vehicle_state.position[:2]
    #         if hasattr(goal, "position"):
    #             goal_position = goal.position
    #         else:
    #             goal_position = ego_2d_position
    #         goal_dist = distance.euclidean(ego_2d_position, goal_position)
    #         penalty += -0.01 * goal_dist
    #
    #         old_obs = env_obs_seq[-2]
    #         old_goal = old_obs.ego_vehicle_state.mission.goal
    #         old_ego_2d_position = old_obs.ego_vehicle_state.position[:2]
    #         if hasattr(old_goal, "position"):
    #             old_goal_position = old_goal.position
    #         else:
    #             old_goal_position = old_ego_2d_position
    #         old_goal_dist = distance.euclidean(old_ego_2d_position, old_goal_position)
    #         penalty += 0.05 * (old_goal_dist - goal_dist)  # 0.1
    #
    #         # ======== Penalty: distance to the center
    #         distance_to_center_np = np.asarray(
    #             [e["distance_to_center"] for e in obs_seq]
    #         )
    #         diff_dist_to_center_penalty = np.abs(distance_to_center_np[-2]) - np.abs(
    #             distance_to_center_np[-1]
    #         )
    #         penalty += 0.01 * diff_dist_to_center_penalty[0]
    #
    #         # ======== Penalty & Bonus: event (collision, off_road, reached_goal, reached_max_episode_steps)
    #         ego_events = last_env_obs.events
    #         # ::collision
    #         penalty += -50.0 if len(ego_events.collisions) > 0 else 0.0
    #         # ::off road
    #         penalty += -50.0 if ego_events.off_road else 0.0
    #         # ::reach goal
    #         if ego_events.reached_goal:
    #             bonus += 20.0
    #
    #         # each time step there is a negative reward to encourage faster mission completion
    #         if not ego_events.reached_goal:
    #             penalty += -0.1
    #
    #         return bonus + penalty
    #
    #     return func

    # # sparse cost function
    # @staticmethod
    # def get_reward_adapter(observation_adapter):
    #     def func(env_obs_seq, env_reward):
    #         penalty, bonus = 0.0, 0.0
    #         last_env_obs = env_obs_seq[-1]
    #
    #         # ======== Penalty & Bonus: event (collision, off_road, reached_goal, time_penalty)
    #         ego_events = last_env_obs.events
    #         # ::collision
    #         penalty += -100.0 if len(ego_events.collisions) > 0 else 0.0
    #         # ::off road
    #         penalty += -100.0 if ego_events.off_road else 0.0
    #         # ::reach goal: large payoff for achieving objective
    #         if ego_events.reached_goal:
    #             bonus += 250.0
    #
    #         # each time step there is a negative reward to encourage faster mission completion
    #         if not ego_events.reached_goal:
    #             penalty += -0.4
    #
    #         return bonus + penalty
    #
    #     return func

    # # sparse + safety distance penalty cost function (communal and personal cost)
    # @staticmethod
    # def get_reward_adapter(observation_adapter):
    #     def func(env_obs_seq, env_reward):
    #         cost_com, cost_per, reward = 0.0, 0.0, 0.0
    #
    #         # get observation of most recent time step
    #         last_obs = env_obs_seq[-1]
    #         # env_obs = observation_adapter(env_obs_seq)
    #
    #         # get ego vehicle information
    #         ego_position = last_obs.ego_vehicle_state.position
    #         # ego_speed = last_obs.ego_vehicle_state.speed
    #         # ego_steering = last_obs.ego_vehicle_state.steering
    #         # ego_yaw_rate = last_obs.ego_vehicle_state.yaw_rate
    #         # ego_linear_velocity = last_obs.ego_vehicle_state.linear_velocity
    #         # ego_linear_acceleration = last_obs.ego_vehicle_state.linear_acceleration
    #         # ego_linear_jerk = last_obs.ego_vehicle_state.linear_jerk
    #         # ego_angular_velocity = last_obs.ego_vehicle_state.angular_velocity
    #         # ego_angular_acceleration = last_obs.ego_vehicle_state.angular_acceleration
    #         # ego_angular_jerk = last_obs.ego_vehicle_state.angular_jerk
    #         # ego_id: str = last_obs.ego_vehicle_state.id
    #         #
    #         # # Number of vehicle, for two vehicles this should be either 0 or 1
    #         # ego_vehicle_nr = int(ego_id[6])
    #
    #         neighborhood_vehicle_states = last_obs.neighborhood_vehicle_states
    #
    #         safety_dist_coeff = 5
    #         for nvs in neighborhood_vehicle_states:
    #             # calculate distance to neighbor vehicle
    #             neigh_position = nvs.position
    #             # neigh_speed = nvs.speed
    #             dist = np.linalg.norm(ego_position - neigh_position)
    #             if dist <= 10:
    #                 cost_com += safety_dist_coeff * np.power(np.power(dist, 2), -1)
    #
    #         # ======== Penalty & Bonus: event (collision, off_road, reached_goal, reached_max_episode_steps)
    #         ego_events = last_obs.events
    #         # ::off-road increases personal cost
    #         cost_per += 50.0 if ego_events.off_road else 0.0
    #         # ::reach goal decreases personal cost
    #         if ego_events.reached_goal:
    #             reward += 100.0
    #
    #         # each time step there is a negative reward to encourage faster mission completion
    #         if not ego_events.reached_goal:
    #             cost_per += 0.4
    #
    #         total_reward = -cost_com - cost_per + reward
    #         return total_reward
    #
    #     return func

    # # bad cost function: designed to crash as fast as possible
    # @staticmethod
    # def get_reward_adapter(observation_adapter):
    #     def func(env_obs_seq, env_reward):
    #         cost_com, cost_per, reward = 0.0, 0.0, 0.0
    #
    #         # get observation of most recent time step
    #         last_obs = env_obs_seq[-1]
    #         ego_events = last_obs.events
    #
    #         # ::collision
    #         reward += 100.0 if len(ego_events.collisions) > 0 else 0.0
    #
    #         # each time step there is a negative reward to encourage faster mission completion
    #         if not ego_events.reached_goal:
    #             cost_per += 0.4
    #
    #         total_reward = -cost_com - cost_per + reward
    #         return total_reward
    #
    #     return func

    # # zero cost function
    # @staticmethod
    # def get_reward_adapter(observation_adapter):
    #     def func(env_obs_seq, env_reward):
    #         return 0.0
    #     return func

    # # safety distance and desired velocity
    # @staticmethod
    # def get_reward_adapter(observation_adapter):
    #     def func(env_obs_seq, env_reward):
    #         cost_com, cost_per, reward = 0.0, 0.0, 0.0
    #
    #         vel_des = {0: 8, 1:14}
    #
    #         # get observation of most recent time step
    #         last_obs = env_obs_seq[-1]
    #         # env_obs = observation_adapter(env_obs_seq)
    #
    #         # get ego vehicle information
    #         ego_position = last_obs.ego_vehicle_state.position
    #         ego_speed = last_obs.ego_vehicle_state.speed
    #         # ego_steering = last_obs.ego_vehicle_state.steering
    #         # ego_yaw_rate = last_obs.ego_vehicle_state.yaw_rate
    #         # ego_linear_velocity = last_obs.ego_vehicle_state.linear_velocity
    #         # ego_linear_acceleration = last_obs.ego_vehicle_state.linear_acceleration
    #         # ego_linear_jerk = last_obs.ego_vehicle_state.linear_jerk
    #         # ego_angular_velocity = last_obs.ego_vehicle_state.angular_velocity
    #         # ego_angular_acceleration = last_obs.ego_vehicle_state.angular_acceleration
    #         # ego_angular_jerk = last_obs.ego_vehicle_state.angular_jerk
    #         ego_id: str = last_obs.ego_vehicle_state.id
    #
    #         # Number of vehicle, for two vehicles this should be either 0 or 1
    #         ego_vehicle_nr = int(ego_id[6])
    #
    #         speed_coeff = 5e-2
    #         cost_per += speed_coeff * np.power(ego_speed - vel_des[ego_vehicle_nr], 2)
    #
    #         neighborhood_vehicle_states = last_obs.neighborhood_vehicle_states
    #
    #         safety_dist_coeff = 5
    #         for nvs in neighborhood_vehicle_states:
    #             # calculate distance to neighbor vehicle
    #             neigh_position = nvs.position
    #             # neigh_speed = nvs.speed
    #             dist = np.linalg.norm(ego_position - neigh_position)
    #             if dist <= 10:
    #                 cost_com += safety_dist_coeff * np.power(np.power(dist, 2), -1)
    #
    #         # ======== Penalty & Bonus: event (collision, off_road, reached_goal, reached_max_episode_steps)
    #         ego_events = last_obs.events
    #         # ::off-road increases personal cost
    #         cost_per += 50.0 if ego_events.off_road else 0.0
    #         # ::reach goal decreases personal cost
    #         if ego_events.reached_goal:
    #             reward += 100.0
    #
    #         # each time step there is a negative reward to encourage faster mission completion
    #         if not ego_events.reached_goal:
    #             cost_per += 0.4
    #
    #         total_reward = -cost_com - cost_per + reward
    #         return total_reward
    #
    #     return func

    # # safety distance and desired velocity
    # @staticmethod
    # def get_reward_adapter(observation_adapter):
    #     def func(env_obs_seq, env_reward):
    #         cost_com, cost_per, reward = 0.0, 0.0, 0.0
    #
    #         vel_des = {0: 10, 1: 16}
    #
    #         # get observation of most recent time step
    #         last_obs = env_obs_seq[-1]
    #         # env_obs = observation_adapter(env_obs_seq)
    #
    #         # get ego vehicle information
    #         ego_position = last_obs.ego_vehicle_state.position
    #         ego_speed = last_obs.ego_vehicle_state.speed
    #         # ego_steering = last_obs.ego_vehicle_state.steering
    #         # ego_yaw_rate = last_obs.ego_vehicle_state.yaw_rate
    #         # ego_linear_velocity = last_obs.ego_vehicle_state.linear_velocity
    #         # ego_linear_acceleration = last_obs.ego_vehicle_state.linear_acceleration
    #         # ego_linear_jerk = last_obs.ego_vehicle_state.linear_jerk
    #         # ego_angular_velocity = last_obs.ego_vehicle_state.angular_velocity
    #         # ego_angular_acceleration = last_obs.ego_vehicle_state.angular_acceleration
    #         # ego_angular_jerk = last_obs.ego_vehicle_state.angular_jerk
    #         ego_id: str = last_obs.ego_vehicle_state.id
    #
    #         # Number of vehicle, for two vehicles this should be either 0 or 1
    #         ego_vehicle_nr = int(ego_id[6])
    #
    #         speed_coeff = 0.4
    #         cost_per += speed_coeff * np.power(ego_speed - vel_des[ego_vehicle_nr], 2)
    #
    #         # ======== Penalty & Bonus: event (collision, off_road, reached_goal, reached_max_episode_steps)
    #         ego_events = last_obs.events
    #         # ::collision
    #         cost_com += 100.0 if len(ego_events.collisions) > 0 else 0.0
    #         # ::off-road increases personal cost
    #         cost_per += 50.0 if ego_events.off_road else 0.0
    #         # ::reach goal decreases personal cost
    #         if ego_events.reached_goal:
    #             reward += 100.0
    #
    #         # each time step there is a negative reward to encourage faster mission completion
    #         if not ego_events.reached_goal:
    #             cost_per += 0.4
    #
    #         total_reward = -cost_com - cost_per + reward
    #         return total_reward
    #
    #     return func

    # # safety distance and desired velocity
    # @staticmethod
    # def get_reward_adapter(observation_adapter):
    #     def func(env_obs_seq, env_reward):
    #         cost_com, cost_per, reward = 0.0, 0.0, 0.0
    #
    #         # get observation of most recent time step
    #         last_obs = env_obs_seq[-1]
    #         # env_obs = observation_adapter(env_obs_seq)
    #
    #         # get ego vehicle information
    #         ego_position = last_obs.ego_vehicle_state.position
    #         ego_speed = last_obs.ego_vehicle_state.speed
    #         # ego_steering = last_obs.ego_vehicle_state.steering
    #         # ego_yaw_rate = last_obs.ego_vehicle_state.yaw_rate
    #         # ego_linear_velocity = last_obs.ego_vehicle_state.linear_velocity
    #         # ego_linear_acceleration = last_obs.ego_vehicle_state.linear_acceleration
    #         # ego_linear_jerk = last_obs.ego_vehicle_state.linear_jerk
    #         # ego_angular_velocity = last_obs.ego_vehicle_state.angular_velocity
    #         # ego_angular_acceleration = last_obs.ego_vehicle_state.angular_acceleration
    #         # ego_angular_jerk = last_obs.ego_vehicle_state.angular_jerk
    #         ego_id: str = last_obs.ego_vehicle_state.id
    #
    #         # Number of vehicle, for two vehicles this should be either 0 or 1
    #         ego_vehicle_nr = int(ego_id[6])
    #
    #         # for agent 0, desired speed is in [9,11]
    #         # for agent 1, desired speed is in [15,17]
    #
    #         if ego_vehicle_nr == 0:
    #             if ego_speed < 9:
    #                 cost_per += 1
    #             if ego_speed > 11:
    #                 cost_per += 3
    #
    #         if ego_vehicle_nr == 1:
    #             if ego_speed > 17:
    #                 cost_per += 1
    #             if ego_speed < 15:
    #                 cost_per += 3
    #
    #
    #         # ======== Penalty & Bonus: event (collision, off_road, reached_goal, reached_max_episode_steps)
    #         ego_events = last_obs.events
    #         # ::collision
    #         cost_com += 100.0 if len(ego_events.collisions) > 0 else 0.0
    #         # ::off-road increases personal cost
    #         cost_per += 50.0 if ego_events.off_road else 0.0
    #         # ::reach goal decreases personal cost
    #         if ego_events.reached_goal:
    #             reward += 100.0
    #
    #         # each time step there is a negative reward to encourage faster mission completion
    #         if not ego_events.reached_goal:
    #             cost_per += 0.4
    #
    #         total_reward = -cost_com - cost_per + reward
    #         return total_reward
    #
    #     return func

    # # safety distance and desired velocity
    # @staticmethod
    # def get_reward_adapter(observation_adapter):
    #     def func(env_obs_seq, env_reward):
    #         cost_com, cost_per, reward = 0.0, 0.0, 0.0
    #
    #         # get observation of most recent time step
    #         last_obs = env_obs_seq[-1]
    #         # env_obs = observation_adapter(env_obs_seq)
    #
    #         # get ego vehicle information
    #         # ego_position = last_obs.ego_vehicle_state.position
    #         ego_speed = last_obs.ego_vehicle_state.speed
    #         # ego_steering = last_obs.ego_vehicle_state.steering
    #         # ego_yaw_rate = last_obs.ego_vehicle_state.yaw_rate
    #         # ego_linear_velocity = last_obs.ego_vehicle_state.linear_velocity
    #         # ego_linear_acceleration = last_obs.ego_vehicle_state.linear_acceleration
    #         # ego_linear_jerk = last_obs.ego_vehicle_state.linear_jerk
    #         # ego_angular_velocity = last_obs.ego_vehicle_state.angular_velocity
    #         # ego_angular_acceleration = last_obs.ego_vehicle_state.angular_acceleration
    #         # ego_angular_jerk = last_obs.ego_vehicle_state.angular_jerk
    #         ego_id: str = last_obs.ego_vehicle_state.id
    #
    #         # Number of vehicle, for two vehicles this should be either 0 or 1
    #         ego_vehicle_nr = int(ego_id[6])
    #
    #         # for agent 0, desired speed is in [4,6]
    #         # for agent 1, desired speed is in [9,11]
    #         # for agent 2, desired speed is in [14,16]
    #         # for agent 3, desired speed is in [19,21]
    #
    #         if ego_vehicle_nr == 0:
    #             if ego_speed < 4 or ego_speed > 6:
    #                 cost_per += 1
    #
    #         if ego_vehicle_nr == 1:
    #             if ego_speed < 9 or ego_speed > 11:
    #                 cost_per += 1
    #
    #         if ego_vehicle_nr == 2:
    #             if ego_speed < 14 or ego_speed > 16:
    #                 cost_per += 1
    #
    #         if ego_vehicle_nr == 3:
    #             if ego_speed < 19 or ego_speed > 21:
    #                 cost_per += 1
    #
    #         # ======== Penalty & Bonus: event (collision, off_road, reached_goal, reached_max_episode_steps)
    #         ego_events = last_obs.events
    #         # ::collision
    #         cost_com += 100.0 if len(ego_events.collisions) > 0 else 0.0
    #         # ::off-road increases personal cost
    #         cost_per += 50.0 if ego_events.off_road else 0.0
    #         # ::reach goal decreases personal cost
    #         if ego_events.reached_goal:
    #             reward += 100.0
    #
    #         # NO TIME PENALTY
    #
    #         total_reward = -cost_com - cost_per + reward
    #         return total_reward
    #
    #     return func

    # # safety distance and desired velocity
    # @staticmethod
    # def get_reward_adapter(observation_adapter):
    #     def func(env_obs_seq, env_reward):
    #         cost_com, cost_per, reward = 0.0, 0.0, 0.0
    #
    #         # get observation of most recent time step
    #         last_obs = env_obs_seq[-1]
    #         # env_obs = observation_adapter(env_obs_seq)
    #
    #         # desired velocities for the agents
    #         # vel_des = {0: 5,
    #         #            1: 10,
    #         #            2: 15,
    #         #            3: 20}
    #
    #         vel_des = {0: 5, 1: 15}
    #
    #         # get ego vehicle information
    #         # ego_position = last_obs.ego_vehicle_state.position
    #         ego_speed = last_obs.ego_vehicle_state.speed
    #         # ego_steering = last_obs.ego_vehicle_state.steering
    #         # ego_yaw_rate = last_obs.ego_vehicle_state.yaw_rate
    #         # ego_linear_velocity = last_obs.ego_vehicle_state.linear_velocity
    #         # ego_linear_acceleration = last_obs.ego_vehicle_state.linear_acceleration
    #         # ego_linear_jerk = last_obs.ego_vehicle_state.linear_jerk
    #         # ego_angular_velocity = last_obs.ego_vehicle_state.angular_velocity
    #         # ego_angular_acceleration = last_obs.ego_vehicle_state.angular_acceleration
    #         # ego_angular_jerk = last_obs.ego_vehicle_state.angular_jerk
    #         ego_id: str = last_obs.ego_vehicle_state.id
    #
    #         # Number of vehicle, for two vehicles this should be either 0 or 1
    #         ego_vehicle_nr = int(ego_id[6])
    #
    #         speed_coeff = 0.1
    #         cost_per += speed_coeff * np.power(ego_speed - vel_des[ego_vehicle_nr], 2)
    #
    #         # ======== Penalty & Bonus: event (collision, off_road, reached_goal, reached_max_episode_steps)
    #         ego_events = last_obs.events
    #         # ::collision
    #         cost_com += 10000.0 if len(ego_events.collisions) > 0 else 0.0
    #         # ::off-road increases personal cost
    #         cost_per += 5000.0 if ego_events.off_road else 0.0
    #         # ::reach goal decreases personal cost
    #         if ego_events.reached_goal:
    #             reward += 1000.0
    #
    #         # NO TIME PENALTY
    #
    #         total_reward = -cost_com - cost_per + reward
    #         return total_reward
    #
    #     return func

    # # safety distance and desired velocity
    # @staticmethod
    # def get_reward_adapter(observation_adapter):
    #     def func(env_obs_seq, env_reward):
    #         cost_com, cost_per, reward = 0.0, 0.0, 0.0
    #
    #         vel_des = {0: 7, 1: 15}
    #
    #         # get observation of most recent time step
    #         last_obs = env_obs_seq[-1]
    #
    #         # get ego vehicle information
    #         ego_position = last_obs.ego_vehicle_state.position
    #         ego_speed = last_obs.ego_vehicle_state.speed
    #
    #         ego_id: str = last_obs.ego_vehicle_state.id
    #
    #         # Number of vehicle, for two vehicles this should be either 0 or 1
    #         ego_vehicle_nr = int(ego_id[6])
    #
    #         speed_coeff = 1
    #         cost_per += speed_coeff * np.power(ego_speed - vel_des[ego_vehicle_nr], 2)
    #
    #         neighborhood_vehicle_states = last_obs.neighborhood_vehicle_states
    #
    #         safety_dist_coef = 10 * 750
    #         safety_dist = float(10)  # [m]
    #         for nvs in neighborhood_vehicle_states:
    #             # calculate distance to neighbor vehicle
    #             neigh_position = nvs.position
    #             dist = float(np.linalg.norm(ego_position - neigh_position))
    #             if dist <= safety_dist:
    #                 cost_com += min(safety_dist_coef * (np.power(dist**2, -1) - np.power(safety_dist**2, -1)), 6000)
    #
    #         # ======== Penalty & Bonus: event (collision, off_road, reached_goal, reached_max_episode_steps)
    #         ego_events = last_obs.events
    #         # ::off-road increases personal cost
    #         cost_per += 5000.0 if ego_events.off_road else 0.0
    #
    #         # collisions are dealt with by safety distance cost
    #         # no reward for reaching the goal
    #
    #         # give reward of average cost when both cars drive at average desired
    #         # velocity to offset desires to complete the mission faster
    #         if not ego_events.reached_goal:
    #             reward += 4 ** 2 + 5
    #
    #         total_reward = -cost_com - cost_per + reward
    #         return total_reward
    #
    #     return func

    # # lexicographic cost: <com_cost, per_cost>, where per_cost = <off-road, goal reached, time cost>
    # @staticmethod
    # def get_reward_adapter(observation_adapter):
    #     def func(env_obs_seq, env_reward):
    #         cost_com, cost_per, reward = 0.0, 0.0, 0.0
    #
    #         # get observation of most recent time step
    #         last_obs = env_obs_seq[-1]
    #
    #         # ======== Penalty & Bonus: event (collision, off_road, reached_goal, reached_max_episode_steps)
    #         ego_events = last_obs.events
    #         # ::collision
    #         cost_com += 1000.0 if len(ego_events.collisions) > 0 else 0.0
    #         # ::off-road increases personal cost
    #         cost_per += 500.0 if ego_events.off_road else 0.0
    #         # ::reach goal decreases personal cost
    #         if ego_events.reached_goal:
    #             reward += 300.0
    #         else:
    #             # time penalty increases personal cost
    #             cost_per += 1.0
    #
    #         total_reward = -cost_com - cost_per + reward
    #         return total_reward
    #
    #     return func

    # # lexicographic cost: <com_cost, per_cost>, where
    # # per_cost = <off-road, goal reached, (time cost, closer to goal)>
    # @staticmethod
    # def get_reward_adapter(observation_adapter):
    #     def func(env_obs_seq, env_reward):
    #         cost_com, cost_per, reward = 0.0, 0.0, 0.0
    #
    #         # get observation of most recent time step
    #         current_obs = env_obs_seq[-1]
    #         last_obs = env_obs_seq[-2]
    #
    #         # ======== Penalty & Bonus: event (collision, off_road, reached_goal, reached_max_episode_steps)
    #         ego_events = current_obs.events
    #         # ::collision
    #         cost_com += 1000.0 if len(ego_events.collisions) > 0 else 0.0
    #         # ::off-road increases personal cost
    #         cost_per += 500.0 if ego_events.off_road else 0.0
    #         # ::reach goal decreases personal cost
    #         if ego_events.reached_goal:
    #             reward += 300.0
    #         else:
    #             # time penalty increases personal cost
    #             cost_per += 2.0
    #
    #         x_gc = current_obs.ego_vehicle_state.mission.goal.position
    #         x_c = current_obs.ego_vehicle_state.position
    #         x_gl = last_obs.ego_vehicle_state.mission.goal.position
    #         x_l = last_obs.ego_vehicle_state.position
    #         current_dist_to_goal = np.sqrt((x_c[0]-x_gc[0])**2 + (x_c[1]-x_gc[1])**2)
    #         last_dist_to_goal = np.sqrt((x_l[0]-x_gl[0])**2 + (x_l[1]-x_gl[1])**2)
    #
    #         goal_improvement = last_dist_to_goal - current_dist_to_goal
    #
    #         if goal_improvement > 0:
    #             # lexicost1
    #             reward += 3 * min(goal_improvement, 1)
    #             # # lexicost2
    #             # reward += 0.5 * min(goal_improvement, 3)
    #         elif goal_improvement < 0:
    #             # lexicost1
    #             reward += 3 * max(goal_improvement, -1)
    #             # # lexicost2
    #             # reward += 0.5 * max(goal_improvement, -3)
    #
    #         total_reward = -cost_com - cost_per + reward
    #         return total_reward
    #
    #     return func

    # # pseudo-lexicographic cost (additional clearance cost): <com_cost, per_cost>, where
    # # per_cost = <off-road, goal reached, (time cost, closer to goal)>
    # @staticmethod
    # def get_reward_adapter(observation_adapter):
    #     def func(env_obs_seq, env_reward):
    #         cost_com, cost_per, reward = 0.0, 0.0, 0.0
    #
    #         # get observation of most recent time step
    #         current_obs = env_obs_seq[-1]
    #         last_obs = env_obs_seq[-2]
    #
    #         # ======== Penalty & Bonus: event (collision, off_road, reached_goal, reached_max_episode_steps)
    #         ego_events = current_obs.events
    #         # ::collision
    #         cost_com += 1000.0 if len(ego_events.collisions) > 0 else 0.0
    #         # ::off-road increases personal cost
    #         cost_per += 500.0 if ego_events.off_road else 0.0
    #         # ::reach goal decreases personal cost
    #         if ego_events.reached_goal:
    #             reward += 300.0
    #         else:
    #             # time penalty increases personal cost
    #             cost_per += 2.0
    #
    #         x_gc = current_obs.ego_vehicle_state.mission.goal.position
    #         x_c = current_obs.ego_vehicle_state.position
    #         x_gl = last_obs.ego_vehicle_state.mission.goal.position
    #         x_l = last_obs.ego_vehicle_state.position
    #         current_dist_to_goal = np.sqrt((x_c[0] - x_gc[0]) ** 2 + (x_c[1] - x_gc[1]) ** 2)
    #         last_dist_to_goal = np.sqrt((x_l[0] - x_gl[0]) ** 2 + (x_l[1] - x_gl[1]) ** 2)
    #
    #         goal_improvement = last_dist_to_goal - current_dist_to_goal
    #
    #         if goal_improvement > 0:
    #             # lexicost1
    #             reward += 3 * min(goal_improvement, 1)
    #             # # lexicost2
    #             # reward += 0.5 * min(goal_improvement, 3)
    #         elif goal_improvement < 0:
    #             # lexicost1
    #             reward += 3 * max(goal_improvement, -1)
    #             # # lexicost2
    #             # reward += 0.5 * max(goal_improvement, -3)
    #
    #         neighborhood_vehicle_states = last_obs.neighborhood_vehicle_states
    #         safety_dist_coef = 100
    #         safety_dist = float(10)  # [m]
    #         for nvs in neighborhood_vehicle_states:
    #             # calculate distance to neighbor vehicle
    #             neigh_position = nvs.position
    #             dist = float(np.linalg.norm(x_c - neigh_position))
    #             if dist <= safety_dist:
    #                 cost_com += min(safety_dist_coef * (np.power(dist ** 2, -1) - np.power(safety_dist ** 2, -1)), 5)
    #
    #         total_reward = -cost_com - cost_per + reward
    #         return total_reward
    #
    #     return func

    # pseudo-lexicographic cost (additional clearance cost): <com_cost, per_cost>, where
    # per_cost = <off-road, goal reached, (time cost, closer to goal)>
    @staticmethod
    def get_reward_adapter(observation_adapter):
        def func(env_obs_seq, env_reward):
            cost_com, cost_per, reward = 0.0, 0.0, 0.0

            # get observation of most recent time step
            current_obs = env_obs_seq[-1]
            last_obs = env_obs_seq[-2]

            # ======== Penalty & Bonus: event (collision, off_road, reached_goal, reached_max_episode_steps)
            ego_events = current_obs.events
            # ::collision
            cost_com += 1000.0 if len(ego_events.collisions) > 0 else 0.0
            # ::off-road increases personal cost
            cost_per += 500.0 if ego_events.off_road else 0.0
            # ::reach goal decreases personal cost
            if ego_events.reached_goal:
                reward += 300.0
            else:
                # time penalty increases personal cost
                cost_per += 2.0

            x_gc = current_obs.ego_vehicle_state.mission.goal.position
            x_c = current_obs.ego_vehicle_state.position
            x_gl = last_obs.ego_vehicle_state.mission.goal.position
            x_l = last_obs.ego_vehicle_state.position
            current_dist_to_goal = np.sqrt((x_c[0] - x_gc[0]) ** 2 + (x_c[1] - x_gc[1]) ** 2)
            last_dist_to_goal = np.sqrt((x_l[0] - x_gl[0]) ** 2 + (x_l[1] - x_gl[1]) ** 2)

            goal_improvement = last_dist_to_goal - current_dist_to_goal

            if goal_improvement > 0:
                reward += 1.0 * min(goal_improvement, 2)
            elif goal_improvement < 0:
                reward += 1.0 * max(goal_improvement, -2)

            ttcs = get_ttc(current_obs)
            ttc_cutoff = 3.0  # [s]
            ttc_scaling = 1.0
            ttc_coeff = 5.625
            for ttc in ttcs:
                if ttc < ttc_cutoff:
                    if ttc > 1e-5:
                        cost_com += ttc_scaling * min(ttc_coeff * (np.power(ttc ** 2, -1) - np.power(ttc_cutoff ** 2, -1)), 5.0)
                    else:
                        cost_com += ttc_scaling * 5.0

            total_reward = -cost_com - cost_per + reward
            return total_reward

        return func
