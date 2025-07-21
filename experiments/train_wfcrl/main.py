import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


from diffusion_co_design.common.rl.mappo import MAPPOCoDesign
from diffusion_co_design.wfcrl.static import GROUP_NAME
import diffusion_co_design.wfcrl.schema as schema
import diffusion_co_design.wfcrl.design as design
from diffusion_co_design.wfcrl.env import create_env
from diffusion_co_design.wfcrl.model.rl import wfcrl_models


class Trainer(
    MAPPOCoDesign[
        schema.DesignerConfig,
        schema.ScenarioConfig,
        schema.ActorCriticConfig,
        schema.TrainingConfig,
    ]
):
    def __init__(self, cfg: schema.TrainingConfig):
        super().__init__(cfg, "diffusion-co-design-wfcrl")

    def create_designer(self, scenario, designer, ppo, artifact_dir, device):
        return design.create_designer(
            scenario=scenario,
            designer=designer,
            ppo_cfg=ppo,
            normalisation_statistics=self.cfg.normalisation,
            artifact_dir=artifact_dir,
            device=device,
        )

    def create_env(self, mode, scenario, designer, device, render=False):
        return create_env(
            mode=mode,
            scenario=scenario,
            designer=designer,
            device=device,
            render=render,
        )

    def create_actor_critic_models(self, reference_env, actor_critic_config, device):
        return wfcrl_models(
            env=reference_env,
            cfg=actor_critic_config,
            normalisation=self.cfg.normalisation,
            device=device,
        )

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
