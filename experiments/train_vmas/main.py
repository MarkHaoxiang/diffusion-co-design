import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from diffusion_co_design.common.rl.mappo import MAPPOCoDesign
from diffusion_co_design.common.design import DesignerParams
from diffusion_co_design.vmas.model.rl import vmas_models
from diffusion_co_design.vmas.scenario.env import create_env
from diffusion_co_design.vmas import design, schema
from diffusion_co_design.vmas.static import GROUP_NAME


class Trainer(
    MAPPOCoDesign[
        schema.DesignerConfig,
        schema.ScenarioConfigType,
        schema.ActorCriticConfig,
        schema.TrainingConfig,
    ]
):
    support_vmap = False

    def __init__(self, cfg: schema.TrainingConfig):
        super().__init__(cfg, f"diffusion-co-design-vmas-{cfg.scenario.name}")

    def create_designer(self, scenario, designer, ppo, artifact_dir, device):
        return design.create_designer(
            scenario=scenario,
            designer=designer,
            ppo_cfg=ppo,
            artifact_dir=artifact_dir,
            device=device,
        )

    def create_batched_env(self, make_design_consumer, n_envs, mode):
        return create_env(
            mode=mode,
            scenario=self.cfg.scenario,
            designer=make_design_consumer(),
            num_environments=n_envs,
            device=self.device.env_device,
        )

    def create_env(self, mode, scenario, designer, device, render=False):
        return create_env(
            mode=mode,
            scenario=scenario,
            designer=designer,
            num_environments=1,
            device=device,
        )

    def create_actor_critic_models(self, reference_env, actor_critic_config, device):
        return vmas_models(
            env=reference_env,
            scenario=self.cfg.scenario,
            actor_critic_cfg=actor_critic_config,
            device=device,
        )

    def create_placeholder_designer(self, scenario):
        return design.RandomDesigner(
            designer_setting=DesignerParams.placeholder(scenario=scenario)
        )

    def post_sample_hook(self, sampling_td):
        sampling_td.set(
            ("next", self.group_name, "done"),
            sampling_td.get(("next", "done"))
            .unsqueeze(-1)
            .expand(sampling_td.get_item_shape(("next", "agents", "reward"))),
        )

        sampling_td.set(
            ("next", self.group_name, "terminated"),
            sampling_td.get(("next", "terminated"))
            .unsqueeze(-1)
            .expand(sampling_td.get_item_shape(("next", "agents", "reward"))),
        )

        return sampling_td

    @property
    def group_name(self):
        return GROUP_NAME


@hydra.main(version_base=None, config_path="conf", config_name="random")
def run(config: DictConfig):
    print(f"Running job {HydraConfig.get().job.name}")
    trainer = Trainer(schema.TrainingConfig.from_raw(config))
    trainer.run()


if __name__ == "__main__":
    run()
