import torch
import torch.nn as nn

from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal
from e3nn.o3 import Irreps

from diffusion_co_design.wfcrl.model.segnn import WeightBalancedIrreps, SEGNN
from diffusion_co_design.wfcrl.schema import RLConfig


class MLPPolicy(nn.Module):
    def __init__(self, cfg: RLConfig, env):
        super().__init__()
        self.model = MultiAgentMLP(
            n_agent_inputs=env.observation_spec["turbine", "observation_vec"].shape[-1],
            n_agent_outputs=env.action_spec.shape[-1],
            n_agents=env.num_agents,
            centralised=False,
            share_params=False,
            depth=cfg.policy_depth,
            num_cells=cfg.policy_hidden_size,
            activation_class=torch.nn.Tanh,
        )

        self.std = nn.Parameter(torch.zeros(env.action_spec.shape[-1]))

    def forward(self, x):
        mu = self.model(x)
        std = (
            torch.ones_like(mu) * self.std
        )  # NormalParamExtractor manages transformation
        return torch.cat((mu, std), dim=-1)


def segnn_model(hidden_features: int = 128, lmax_h: int = 2):
    lmax_attr = 3

    input_irreps = Irreps("2x1o")  # Wind vector + yaw
    node_attr_irreps = Irreps.spherical_harmonics(lmax_attr)

    hidden_irreps = WeightBalancedIrreps(
        Irreps("{}x0e".format(hidden_features)),
        node_attr_irreps,
        sh=True,
        lmax=lmax_h,
    )
    pass


def wfcrl_models(env, cfg: RLConfig, device: str):
    policy_net = nn.Sequential(
        MLPPolicy(cfg, env).to(device=device), NormalParamExtractor()
    )

    policy_module = TensorDictModule(
        policy_net,
        in_keys=[("turbine", "observation_vec")],
        out_keys=[("turbine", "loc"), ("turbine", "scale")],
    )
    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec_unbatched,
        in_keys=[("turbine", "loc"), ("turbine", "scale")],
        out_keys=[env.action_key],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env.full_action_spec_unbatched[env.action_key].space.low,
            "high": env.full_action_spec_unbatched[env.action_key].space.high,
        },
        return_log_prob=True,
        log_prob_key=("turbine", "sample_log_prob"),
    )

    critic_net = MultiAgentMLP(
        n_agent_inputs=env.observation_spec["turbine", "observation_vec"].shape[-1],
        n_agent_outputs=1,  # 1 value per agent
        n_agents=env.num_agents,
        centralised=cfg.mappo,
        share_params=False,
        device=device,
        depth=cfg.critic_depth,
        num_cells=cfg.critic_hidden_size,
        activation_class=torch.nn.Tanh,
    )

    critic = TensorDictModule(
        module=critic_net,
        in_keys=[("turbine", "observation_vec")],
        out_keys=[("turbine", "state_value")],
    )

    # Initialise
    td = env.reset().to(device)
    with torch.no_grad():
        policy(td)
        critic(td)

    return policy, critic
