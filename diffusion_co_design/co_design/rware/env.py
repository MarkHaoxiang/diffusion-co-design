from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule
from torchrl.envs import PettingZooWrapper

from rware.layout import Layout

from diffusion_co_design.diffusion.datasets.rware.transform import image_to_layout


class RwareCoDesignWrapper(PettingZooWrapper):
    def __init__(
        self,
        env=None,
        reset_policy: TensorDictModule | None = None,
        environment_objective=None,
        return_state=False,
        group_map=None,
        use_mask=False,
        categorical_actions=True,
        seed=None,
        done_on_any=None,
        **kwargs,
    ):
        super().__init__(
            env,
            return_state,
            group_map,
            use_mask,
            categorical_actions,
            seed,
            done_on_any,
            **kwargs,
        )
        # Hack: TorchRL messes with object attributes, so need to set in inner environment
        # Also, it's difficult to rewrite sync
        self._env._reset_policy = reset_policy
        self._env._environment_objective = environment_objective

    def _reset(self, tensordict: TensorDictBase | None = None, **kwargs):
        """Extract the layout from tensordict and pass to env"""
        reset_policy_output = None

        if tensordict is not None and self._env._environment_objective is not None:
            tensordict[("environment_design", "objective")] = (
                self._env._environment_objective
            )

        if (
            tensordict is not None
            and "environment_design" in tensordict
            and self._env._reset_policy is not None
        ):
            reset_policy_output = self._env._reset_policy(tensordict)
            tensordict.update(
                reset_policy_output, keys_to_update=reset_policy_output.keys()
            )
            layout = image_to_layout(
                tensordict.get(("environment_design", "layout_image"))
                .detach()
                .cpu()
                .numpy()
            )
            options = {"layout": layout}
        else:
            options = None

        # Maybe add the newest layout to tensordict out
        tensordict_out = super()._reset(tensordict, options=options, **kwargs)
        return tensordict_out
