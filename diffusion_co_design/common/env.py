from abc import abstractmethod

from diffusion_co_design.common.pydra import Config


class ScenarioConfig(Config):
    @abstractmethod
    def get_name(self) -> str:
        """
        Get the name of the scenario.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_episode_steps(self) -> int:
        """
        Get the maximum number of steps in the scenario.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_num_agents(self) -> int:
        """
        Get the number of agents in the scenario.
        """
        raise NotImplementedError()
