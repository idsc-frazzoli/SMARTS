#%% test

class A:
    name = 'A'
    
    
print(A.name)

a = A

print(a.name)


#%% Agent Class

from typing import Any, Callable, Optional

from pathlib import Path

import gym
import numpy as np

# ray[rllib] is not the part of main dependency of the SMARTS package. It needs to be installed separately
# as a part of the smarts[train] dependency using the command "pip install -e .[train]. The following try block checks
# whether ray[rllib] was installed by user and raises an Exception warning the user to install it if not so.
try:
    from ray.rllib.models import ModelCatalog
    from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
    from ray.rllib.utils import try_import_tf
except Exception as e:
    from examples import RayException

    raise RayException.required_to("rllib_agent.py")


from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.env.custom_observations import lane_ttc_observation_adapter

# Tries importing tf and returns the module (or None). (from RLlib)
tf = try_import_tf()

class Agent:
    """The base class for agents"""

    @classmethod
    def from_function(cls, agent_function: Callable):
        """A utility function to create an agent from a lambda or other callable object.

        .. code-block:: python

            keep_lane_agent = Agent.from_function(lambda obs: "keep_lane")
        """
        assert callable(agent_function)

        class FunctionAgent(Agent):
            def act(self, obs):
                return agent_function(obs)

        return FunctionAgent()

    def act(self, obs):
        """The agent action. See documentation on observations, `AgentSpec`, and `AgentInterface`.

        Expects an adapted observation and returns an unadapted action.
        """

        raise NotImplementedError
        
# keep_lane_agent = Agent.from_function(lambda obs: "keep_lane")


class RLLibTFSavedModelAgent(Agent):
    def __init__(self, path_to_model, observation_space):
        path_to_model = str(path_to_model)  # might be a str or a Path, normalize to str
        self._prep = ModelCatalog.get_preprocessor_for_space(observation_space)
        self._sess = tf.compat.v1.Session(graph=tf.Graph())
        tf.compat.v1.saved_model.load(
            self._sess, export_dir=path_to_model, tags=["serve"]
        )
        self._output_node = self._sess.graph.get_tensor_by_name("default_policy/add:0")
        self._input_node = self._sess.graph.get_tensor_by_name(
            "default_policy/observation:0"
        )

    def __del__(self):
        self._sess.close()

    def act(self, obs):
        obs = self._prep.transform(obs)
        # These tensor names were found by inspecting the trained model
        res = self._sess.run(self._output_node, feed_dict={self._input_node: [obs]})
        action = res[0]
        return action









