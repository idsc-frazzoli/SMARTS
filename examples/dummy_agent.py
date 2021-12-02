import logging
from typing import Sequence

from envision.client import Client as Envision
from smarts.core import seed as random_seed
from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import AgentInterface
from smarts.core.controllers import ActionSpaceType
from smarts.core.scenario import Scenario
from smarts.core.smarts import SMARTS

from examples.argument_parser import default_argument_parser

logging.basicConfig(level=logging.INFO)


class BasicAgent(Agent):
    def act(self, obs):
        return "keep_lane"


def main(
    script: str,
    scenarios: Sequence[str],
    headless: bool,
    envision_record_data_replay_path: str,
    seed: int,
    episodes: int,
):
    assert episodes > 0
    logger = logging.getLogger(script)
    logger.setLevel(logging.INFO)
    logger.debug("initializing SMARTS")

    envision_client = None
    if not headless or envision_record_data_replay_path:
        envision_client = Envision(output_dir=envision_record_data_replay_path)

    smarts = SMARTS(
        agent_interfaces={},
        traffic_sim=None,
        envision=envision_client,
    )
    random_seed(seed)

    scenarios_iterator = Scenario.scenario_variations(scenarios, [])
    scenario = next(scenarios_iterator)
    agent_spec = AgentSpec(
        interface=AgentInterface(waypoints=True, action=ActionSpaceType.Lane),
        agent_builder=BasicAgent,
    )
    agent_id = "agent-007"
    for episode in range(episodes):
        logger.info(f"starting episode {episode}...")
        agents = {agent_id: agent_spec.build_agent()}

        dones = {}
        observations = smarts.reset(scenario)
        while not dones or not all(dones.values()):

            # Step simulation
            actions = {
                agent_id: agents[agent_id].act(agent_obs)
                for agent_id, agent_obs in observations.items()
            }
            logger.debug(
                f"stepping @ sim_time={smarts.elapsed_sim_time} for agents={list(observations.keys())}..."
            )
            observations, rewards, dones, infos = smarts.step(actions)

            for agent_id in agents.keys():
                if dones.get(agent_id, False):
                    if not observations[agent_id].events.reached_goal:
                        logger.warning(
                            f"agent_id={agent_id} exited @ sim_time={smarts.elapsed_sim_time}"
                        )
                        logger.warning(f"   ... with {observations[agent_id].events}")
                    else:
                        logger.info(
                            f"agent_id={agent_id} reached goal @ sim_time={smarts.elapsed_sim_time}"
                        )
                        logger.debug(f"   ... with {observations[agent_id].events}")
                    del observations[agent_id]

    smarts.destroy()


if __name__ == "__main__":
    parser = default_argument_parser("dummy_agent.py")
    parser.add_argument(
        "--envision_record_data_path",
        help="Envisions data replay output directory where the recording will be stored.",
        type=str,
        default=None,
    )

    args = parser.parse_args()
    main(
        script=parser.prog,
        scenarios=args.scenarios,
        headless=args.headless,
        envision_record_data_replay_path=args.envision_record_data_path,
        seed=args.seed,
        episodes=args.episodes,
    )
