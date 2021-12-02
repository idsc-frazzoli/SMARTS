import logging
from typing import Any, Callable, Dict, Sequence

from envision.client import Client as Envision
from smarts.core import seed as random_seed
from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.controllers import ActionSpaceType
from smarts.core.scenario import Scenario
from smarts.core.smarts import SMARTS
from smarts.core.utils.episodes import episodes

from examples.argument_parser import default_argument_parser

logging.basicConfig(level=logging.INFO)
AGENT_ID = "Agent 1"


class KeepLaneAgent(Agent):
    def act(self, obs):
        return "keep_lane"


def main(
    script, scenarios, sim_name, headless, num_episodes, seed, max_episode_steps=None
):
    logger = logging.getLogger(script)
    logger.setLevel(logging.INFO)
    logger.debug("initializing SMARTS")
    agent_specs = {
        AGENT_ID: AgentSpec(
            interface=AgentInterface.from_type(
                AgentType.Laner, max_episode_steps=max_episode_steps
            ),
            agent_builder=KeepLaneAgent,
        )
    }

    envision_client = None
    smarts = SMARTS(
        agent_interfaces={},
        traffic_sim=None,
        envision=envision_client,
    )
    random_seed(seed)

    scenarios_iterator = Scenario.scenario_variations(scenarios, [])
    scenario = next(scenarios_iterator)

    for episode in episodes(n=num_episodes):
        logger.info(f"starting episode {episode}...")
        observations = smarts.reset(scenario)
        agents = {
            agent_id: agent_spec.build_agent()
            for agent_id, agent_spec in agent_specs.items()
        }
        dones = {}
        while not dones or not all(dones.values()):
            actions = {
                agent_id: agents[agent_id].act(agent_obs)
                for agent_id, agent_obs in observations.items()
            }

            observations, rewards, dones, infos = smarts.step(actions)
            episode.record_step(observations, rewards, dones, infos)

    smarts.destroy()


if __name__ == "__main__":
    parser = default_argument_parser("dummy-agent-example")
    args = parser.parse_args()

    main(
        script=parser.prog,
        scenarios=args.scenarios,
        sim_name=args.sim_name,
        headless=args.headless,
        num_episodes=args.episodes,
        seed=args.seed,
    )
